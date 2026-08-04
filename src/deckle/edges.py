"""Find the card's edges inside a jig window.

This is the module RFC-001's §"The one bad number" was written about. The naive readings
of the problem have all been measured and all fail:

* Anchoring a fixed search band to *design* coordinates put one card's true edge outside
  the band, and the fitter returned the strongest gradient it could find — an edge inside
  the card's own artwork, 376um wrong, reported with a 6.8um residual.
* Re-anchoring to the green mask but keeping "strongest gradient in the band" then broke
  the other card by +1654um, locking onto the window wall across a 1.65mm gap.
* Adding a hue / Lab-a* channel was tried and measured *worse* (-376um to -424um). On the
  failing edge the peak gradient was 2.1 in luma and 0.1 in a*: neither channel had a
  signal, so channel choice could not have been the problem.

What the profile actually looks like, walking inward from the wall (measured, 600dpi):

    green wall | dark line | shadow ramp, up to ~1.7mm | flat plateau | STEP | flat card

The shadow the 3mm-thick frame casts into the clearance gap is a *ramp*, and it is
steeper than the card's own step — up to 35 luma/px against the card's 12-24. So the card
edge can never be "the strongest gradient". But it is always the **innermost** step: past
it there is nothing but flat card. That is the rule implemented here, and it costs nothing
to be right about the ramp because the ramp is always further out.

Consequently the gates in `detect` are absolute-plausibility gates — gap width, step
height, opposite edges agreeing — and never residual. Collinearity only proves the points
are on *a* line.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Line, LineFit, fit_line_trimmed
from .units import DEFAULT_DPI, mm_to_px, px_to_mm

# Inward direction in image coordinates for each window edge.
INWARD = {"top": (0, 1), "bottom": (0, -1), "left": (1, 0), "right": (-1, 0)}

# How far inside the wall to look for the card, when a caller does not derive it. Callers
# should: the honest bound is the window's own slack (opening minus card), because a card
# shoved against one wall puts *all* of the slack on the opposite edge. On the assembled
# jig that is 2.94mm and a card was measured sitting 3.007mm off its top wall.
#
# The band must not be widened "to be safe". The rule takes the innermost step inside it,
# and at a 9mm band the same edge locks onto the boundary of the card's own dark artwork
# 7.8mm in — which is the -376um failure of RFC-001 wearing a different hat.
BAND_MM = 3.5

# Start fractionally *outside* the wall so a card sitting flush against it is not skipped.
# One measured edge has a gap of ~0.04mm.
LEAD_MM = 0.35

# A step must clear this to be a card edge. The smallest true card step measured is ~20
# luma (a white border against a fully-lit shadow plateau); card interiors wobble by ~5.
STEP_FLOOR = 12.0

# The card must still be there after the edge, or it was not an edge.
SUSTAIN_MM = 0.8

# Half-width of the gradient-centroid window used to place the edge to subpixel. The card
# edge transitions over 4-5px (scanner PSF plus the shadow the card's own 0.4mm thickness
# casts). Widening this to 12 was measured and changes every dimension by under 45um while
# making the far edges worse, so the estimator is not sensitive to it — which is the
# reassuring result, since it means the residual bias reported in the PR is a property of
# the image and not of this number.
REFINE_HALF = 4


@dataclass(frozen=True)
class EdgeFit:
    """One fitted card edge, with the evidence that justifies it."""

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
        """Reported for diagnostics only. It is NOT a validity check — see module docs."""
        return px_to_mm(self.fit.residual_sd_px)


def _line_at(line: Line, coord: float, edge: str) -> float:
    """Evaluate a wall line at one scanline: y for a horizontal edge, x for a vertical."""
    p, d = line.point, line.direction
    if edge in ("top", "bottom"):
        return float(p[1] + (coord - p[0]) / d[0] * d[1])
    return float(p[0] + (coord - p[1]) / d[1] * d[0])


def _step_response(prof: np.ndarray, gap: int = 2, win: int = 5) -> np.ndarray:
    """Difference of medians across a short baseline, one value per profile sample.

    Short windows on purpose: a long baseline turns the shadow ramp into a larger
    "step" than the card edge. Medians rather than means so a speck of dust in the
    clearance gap cannot manufacture an edge.
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


def _innermost_run(above: np.ndarray) -> tuple[int, int] | None:
    """Last contiguous run of True in `above`, as inclusive (start, end).

    The shadow ramp and the card edge each produce their own run, separated by the flat
    plateau between them. Taking the innermost run is what makes the card edge findable
    without ever having to out-score the ramp, which is the steeper of the two.
    """
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    start = idx[breaks[-1] + 1] if breaks.size else idx[0]
    return int(start), int(idx[-1])


def _subpixel(prof: np.ndarray, d: int, half: int = 4, search: int = 4) -> float:
    """Refine a step location to subpixel by gradient centroid.

    The centroid window is centred on the gradient *peak*, not on `d`. The step-response
    plateau only locates the edge to within a couple of pixels, and an off-centre window
    clips one flank of the gradient and drags the centroid toward the window's middle —
    worth ~1px per edge, which is 40um and half the error budget.
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
    gray: np.ndarray, edge: str, at: int, anchor: float, n: int, lead_px: float
) -> tuple[np.ndarray, float] | None:
    """Luma profile walking inward from the wall. Returns (profile, distance of sample 0)."""
    dx, dy = INWARD[edge]
    if edge in ("top", "bottom"):
        start = int(round(anchor - lead_px * dy))
        stop = start + n * dy
        if min(start, stop) < 0 or max(start, stop) >= gray.shape[0]:
            return None
        prof = gray[start:stop:dy, at] if dy > 0 else gray[start:stop:dy, at]
        d0 = (start - anchor) * dy
    else:
        start = int(round(anchor - lead_px * dx))
        stop = start + n * dx
        if min(start, stop) < 0 or max(start, stop) >= gray.shape[1]:
            return None
        prof = gray[at, start:stop:dx]
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
) -> EdgeFit:
    """Fit one card edge by taking the innermost sustained step on every scanline.

    `anchor_offset_mm` displaces the search anchor inward; it exists so tests can aim the
    detector wrongly on purpose and confirm the plausibility gates fire, which is the one
    failure this project has already paid for once.
    """
    band_px = mm_to_px(band_mm, dpi)
    lead_px = mm_to_px(lead_mm, dpi)
    sustain_px = mm_to_px(sustain_mm, dpi)
    offset_px = mm_to_px(anchor_offset_mm, dpi)
    # The profile runs lead + band + sustain, and only the first lead + band of it is
    # searched. Reserving the sustain room out of the band instead would silently shorten
    # the reach by 0.8mm, which is exactly how a card sitting 3.007mm off its wall went
    # undetected behind a nominally 3.5mm band.
    n_samples = int(round(lead_px + band_px + sustain_px))
    search_limit = int(round(lead_px + band_px))

    pts, gaps, steps = [], [], []
    for at in scanlines:
        dx, dy = INWARD[edge]
        step_dir = dy if edge in ("top", "bottom") else dx
        wall_at = _line_at(wall, float(at), edge)
        # Everything below is measured relative to `anchor`, including where the edge is
        # finally placed. Measuring from the anchor but reporting against the undisplaced
        # wall would make every dimension track the anchor 1:1 — which is precisely the
        # anchor-dependence this whole approach exists to remove.
        anchor = wall_at + offset_px * step_dir
        got = _profile(gray, edge, int(at), anchor, n_samples, lead_px)
        if got is None:
            continue
        prof, d0 = got
        smooth = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 1.2).ravel()
        r = _step_response(smooth)

        # Only steps with room for card behind them can be card edges.
        limit = min(search_limit, len(smooth) - int(sustain_px))
        above = np.abs(r[:limit]) >= step_floor if limit > 0 else np.zeros(0, bool)
        run = _innermost_run(above)
        if run is None:
            continue
        a, b = run
        # The step response is a plateau centred on the edge, not a spike at it: it stays
        # elevated for several px on either side. Take the peak within the innermost run,
        # never the run's last index — that lands ~7px inside the card and biased every
        # measured dimension short by 200-500um.
        d = a + int(np.argmax(np.abs(r[a : b + 1])))

        # The level after the step must hold — a transient is dust, not a card edge.
        after = smooth[d + 7 : d + 7 + int(sustain_px)]
        if after.size < 4 or np.std(after) > 12.0:
            continue

        # Detect on the smoothed profile, but *place* the edge using the raw one: the
        # blur is there to stop noise inventing steps, and refining on it as well widens
        # the gradient and drags the centroid.
        sub = _subpixel(prof, d, half=REFINE_HALF)
        dist = d0 + sub  # px inward from the anchor
        if edge in ("top", "bottom"):
            pts.append((float(at), anchor + dist * dy))
        else:
            pts.append((anchor + dist * dx, float(at)))
        # Reported gap is always wall-to-card, whatever the anchor was displaced by.
        gaps.append(px_to_mm(dist + offset_px, dpi))
        steps.append(float(r[d]))

    if len(pts) < 16:
        raise ValueError(
            f"card edge {edge!r}: only {len(pts)} of {len(scanlines)} scanlines found a "
            f"sustained step — no card edge here"
        )
    return EdgeFit(
        edge=edge,
        fit=fit_line_trimmed(np.array(pts)),
        gap_mm=np.array(gaps),
        step=np.array(steps),
        n_scanlines=len(scanlines),
    )
