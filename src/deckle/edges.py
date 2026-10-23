"""Find card edges inside a jig window.

Scanlines march inward from the window wall using different strategies:

- brightest: largest luma rise into the card. (use with a pad)
- innermost: follow the shadow outside the card (when no pad)
- outermost: the inner end of the last run of void (pad with a dark-colored card)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np

from .geometry import Line, LineFit, fit_line_trimmed
from .units import DEFAULT_DPI, DEFAULT_STRATEGY, EdgeStrategy, mm_to_px, px_to_mm

INWARD = {"top": (0, 1), "bottom": (0, -1), "left": (1, 0), "right": (-1, 0)}

# Default search depth past the wall
BAND_MM = 3.5

# Start slightly outside the wall (in case card is flush with the jig)
LEAD_MM = 0.35

# Smallest step that can be a card edge (in luma).
STEP_FLOOR = 12.0

# The card must still be there this far past the edge.
SUSTAIN_MM = 0.8


# Half-width of the gradient-centroid window for subpixel placement.
REFINE_HALF = 4

# the brightest a sample can be and still count as void (outermost)
VOID_CEIL = 8.0

# shortest void run up to a card edge (outermost)
VOID_RUN_MM = 0.15

# how far either side of the void run's end to look for the step's peak (outermost)
VOID_EDGE_SEARCH_PX = 4

# fewest hits that make an edge worth fitting a line to
MIN_HITS = 16


class Reason(IntEnum):
    """What happened for one scanline."""

    HIT = 0
    OFF_IMAGE = 1
    NO_ROOM = 2
    NO_VOID = 3
    WEAK_STEP = 4
    NO_STEP = 5
    VOID_AFTER = 6
    NOT_FLAT = 7

    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class Trace:
    """Per-scanline record of one edge search, one element per scanline."""

    scanline: np.ndarray  # int32 px along the edge: x for top/bottom, y for left/right
    anchor: np.ndarray  # float32 px, where the search started
    reason: np.ndarray  # uint8 Reason
    d: np.ndarray  # int16, chosen step sample (-1 where none was chosen)
    dist_px: np.ndarray  # float32, px inward from the anchor (NaN unless hit)

    def __len__(self) -> int:
        return len(self.scanline)

    @property
    def hit(self) -> np.ndarray:
        return self.reason == Reason.HIT


@dataclass(frozen=True)
class EdgeFit:
    """One fitted card edge."""

    edge: str
    fit: LineFit
    gap_mm: np.ndarray  # per-scanline distance from wall to card edge
    step: np.ndarray  # per-scanline step height in luma
    n_scanlines: int

    @property
    def line(self) -> Line:
        return self.fit.line

    @property
    def median_gap_mm(self) -> float:
        return float(np.median(self.gap_mm))

    @property
    def median_step(self) -> float:
        return float(np.median(np.abs(self.step)))

    @property
    def yield_frac(self) -> float:
        return len(self.gap_mm) / max(1, self.n_scanlines)

    @property
    def residual_sd_mm(self) -> float:
        """For debugging."""
        return px_to_mm(self.fit.residual_sd_px)


@dataclass(frozen=True)
class EdgeAttempt:
    edge: str
    trace: Trace
    n_scanlines: int
    band_mm: float
    lead_mm: float
    anchor_offset_mm: float
    strategy: EdgeStrategy
    dpi: float
    fit: EdgeFit | None  # None when too few scanlines hit

    @property
    def n_hit(self) -> int:
        return int(np.count_nonzero(self.trace.hit))

    @property
    def yield_frac(self) -> float:
        return self.n_hit / max(1, self.n_scanlines)

    def counts(self) -> dict[Reason, int]:
        """Scanlines per reason, every reason included."""
        n = np.bincount(self.trace.reason, minlength=len(Reason))
        return {r: int(n[r]) for r in Reason}


def _line_at(line: Line, coord: float, edge: str) -> float:
    """Evaluate a wall line at one scanline."""
    p, d = line.point, line.direction
    if edge in ("top", "bottom"):
        return float(p[1] + (coord - p[0]) / d[0] * d[1])
    return float(p[0] + (coord - p[1]) / d[1] * d[0])


def _step_response(prof: np.ndarray, gap: int = 2, win: int = 5) -> np.ndarray:
    """Difference of medians across a short baseline, one value per profile sample.

    Short, so a shadow ramp does not outscore the card's step; medians, so dust cannot
    make one.
    """
    n = len(prof)
    r = np.zeros(n)
    span = gap + win
    if n < 2 * span + 1:
        return r
    sw = np.lib.stride_tricks.sliding_window_view(prof, win)
    med = np.median(sw, axis=1)  # med[i] = median(prof[i:i+win])
    # after[d] = median(prof[d+gap : d+gap+win]); before[d] = median(prof[d-gap-win : d-gap])
    idx = np.arange(span, n - span)
    r[idx] = med[idx + gap] - med[idx - span]
    return r


def _innermost_run(above: np.ndarray, min_len: int = 1) -> tuple[int, int] | None:
    """Last contiguous run of True in above at least min_len long"""
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return None

    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    keep = np.flatnonzero(ends - starts + 1 >= min_len)
    if keep.size == 0:
        return None
    return int(starts[keep[-1]]), int(ends[keep[-1]])


def _subpixel(prof: np.ndarray, d: int, half: int = 4, search: int = 4) -> float:
    """Refine a step location to subpixel by gradient centroid.

    The window is centred on the gradient peak near `d`, not on `d`: an off-centre window
    clips one flank and pulls the centroid toward its middle.
    """
    n = len(prof)
    grad = np.zeros(n)
    grad[1:-1] = np.abs(prof[2:] - prof[:-2])
    lo, hi = max(1, d - search), min(n - 1, d + search + 1)
    if hi - lo < 2:
        return float(d)
    peak = lo + int(np.argmax(grad[lo:hi]))
    a, b = max(1, peak - half), min(n - 1, peak + half + 1)
    g = grad[a:b]
    total = g.sum()
    if total <= 0:
        return float(peak)
    return float(np.arange(a, b) @ g / total)


def _profile(
    gray: np.ndarray, edge: str, scanline: int, anchor: float, n: int, lead_px: float
) -> tuple[np.ndarray, float] | None:
    """Luma profile walking inward from the wall. Returns (profile, distance of sample 0)."""
    dx, dy = INWARD[edge]
    if edge in ("top", "bottom"):
        start = int(round(anchor - lead_px * dy))
        stop = start + n * dy
        if min(start, stop) < 0 or max(start, stop) >= gray.shape[0]:
            return None
        prof = gray[start:stop:dy, scanline]
        d0 = (start - anchor) * dy
    else:
        start = int(round(anchor - lead_px * dx))
        stop = start + n * dx
        if min(start, stop) < 0 or max(start, stop) >= gray.shape[1]:
            return None
        prof = gray[scanline, start:stop:dx]
        d0 = (start - anchor) * dx
    return (prof.astype(np.float32), float(d0)) if len(prof) > 12 else None


def fit_card_edge(
    gray: np.ndarray,
    wall: Line,
    edge: str,
    scanlines: np.ndarray,
    dpi: float = DEFAULT_DPI,
    band_mm: float = BAND_MM,
    lead_mm: float = LEAD_MM,
    step_floor: float = STEP_FLOOR,
    sustain_mm: float = SUSTAIN_MM,
    anchor_offset_mm: float = 0.0,
    strategy: EdgeStrategy | str = DEFAULT_STRATEGY,
) -> EdgeAttempt:
    """Main edge finding algorithm"""
    strategy = EdgeStrategy(strategy)

    band_px = mm_to_px(band_mm, dpi)
    lead_px = mm_to_px(lead_mm, dpi)
    sustain_px = mm_to_px(sustain_mm, dpi)
    void_run_px = int(round(mm_to_px(VOID_RUN_MM, dpi)))
    offset_px = mm_to_px(anchor_offset_mm, dpi)

    n_samples = int(round(lead_px + band_px + sustain_px))
    search_limit = int(round(lead_px + band_px))

    n = len(scanlines)
    anchors = np.full(n, np.nan, dtype=np.float32)
    reasons = np.full(n, 255, dtype=np.uint8)  # every path below overwrites this
    chosen = np.full(n, -1, dtype=np.int16)
    dists = np.full(n, np.nan, dtype=np.float32)

    pts, gaps, steps = [], [], []
    for i, scanline in enumerate(scanlines):
        dx, dy = INWARD[edge]
        step_dir = dy if edge in ("top", "bottom") else dx

        # find the wall
        wall_at = _line_at(wall, float(scanline), edge)

        anchor = wall_at + offset_px * step_dir
        anchors[i] = anchor

        # compute the luma of samples on this line
        got = _profile(gray, edge, int(scanline), anchor, n_samples, lead_px)
        if got is None:
            reasons[i] = Reason.OFF_IMAGE
            continue

        prof, d0 = got
        smooth = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 1.2).ravel()

        # find if there are brightness steps between samples
        r = _step_response(smooth)

        limit = min(search_limit, len(smooth) - int(sustain_px))
        if limit <= 0:
            reasons[i] = Reason.NO_ROOM
            continue

        if strategy is EdgeStrategy.BRIGHTEST:
            d = int(np.argmax(r[:limit]))
            if r[d] < step_floor:
                reasons[i], chosen[i] = Reason.WEAK_STEP, d
                continue
        elif strategy is EdgeStrategy.OUTERMOST:
            run = _innermost_run(prof[:limit] <= VOID_CEIL, void_run_px)
            if run is None:
                reasons[i] = Reason.NO_VOID
                continue
            _, b = run
            lo = max(0, b - VOID_EDGE_SEARCH_PX)
            hi = min(limit, b + VOID_EDGE_SEARCH_PX + 1)
            if hi - lo < 1:
                reasons[i] = Reason.NO_ROOM
                continue
            d = lo + int(np.argmax(r[lo:hi]))
            if r[d] < step_floor:
                reasons[i], chosen[i] = Reason.WEAK_STEP, d
                continue
        else:  # innermost
            above = np.abs(r[:limit]) >= step_floor
            run = _innermost_run(above)
            if run is None:
                reasons[i] = Reason.NO_STEP
                continue
            a, b = run
            d = a + int(np.argmax(np.abs(r[a : b + 1])))

        chosen[i] = d
        after = smooth[d + 7 : d + 7 + int(sustain_px)]
        if after.size < 4:
            reasons[i] = Reason.NO_ROOM
            continue
        if strategy is EdgeStrategy.OUTERMOST:
            # dark-bordered card
            if after.min() <= VOID_CEIL:
                reasons[i] = Reason.VOID_AFTER
                continue
        elif np.std(after) > 12.0:
            reasons[i] = Reason.NOT_FLAT
            continue

        # Detect on the smoothed profile, place on the raw one.
        sub = _subpixel(prof, d, half=REFINE_HALF)
        dist = d0 + sub  # px inward from the anchor
        reasons[i], dists[i] = Reason.HIT, dist
        if edge in ("top", "bottom"):
            pts.append((float(scanline), anchor + dist * dy))
        else:
            pts.append((anchor + dist * dx, float(scanline)))
        gaps.append(px_to_mm(dist + offset_px, dpi))
        steps.append(float(r[d]))

    trace = Trace(scanline=scanlines, anchor=anchors, reason=reasons, d=chosen, dist_px=dists)

    fit = None
    if len(pts) >= MIN_HITS:
        fit = EdgeFit(
            edge=edge,
            fit=fit_line_trimmed(np.array(pts)),
            gap_mm=np.array(gaps),
            step=np.array(steps),
            n_scanlines=n,
        )
    return EdgeAttempt(
        edge=edge,
        trace=trace,
        n_scanlines=n,
        band_mm=band_mm,
        lead_mm=lead_mm,
        anchor_offset_mm=anchor_offset_mm,
        strategy=strategy,
        dpi=dpi,
        fit=fit,
    )
