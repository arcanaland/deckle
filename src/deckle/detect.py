"""Detect cards + geometry from an image"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .edges import STEP_FLOOR, EdgeAttempt, EdgeFit, Reason, fit_card_edge
from .geometry import Quad, corners_from_lines
from .jig import DEFAULT_FRAME_COLOR, EDGES, FrameColor, JigError, Window, find_windows
from .units import (
    DEFAULT_ASPECT,
    DEFAULT_CARD_MM,
    DEFAULT_DPI,
    DEFAULT_OPPOSITE_TOL_MM,
    DEFAULT_STRATEGY,
    EdgeStrategy,
    px_to_mm,
)


class DetectionError(RuntimeError):
    """Not card found."""

    def __init__(self, message: str, report: DetectionReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class CardSpec:
    """Card + physical metadata."""

    width_mm: float = DEFAULT_CARD_MM[0]
    height_mm: float = DEFAULT_CARD_MM[1]
    aspect: float = DEFAULT_ASPECT
    aspect_tol: float = 0.01
    size_tol_mm: float = 1.5

    # how closely opposite edges must agree (if it's not a perfect rectangle)
    opposite_tol_mm: float = DEFAULT_OPPOSITE_TOL_MM

    # slack within a window
    gap_margin_mm: float = 0.70

    gap_min_mm: float = -0.10
    min_yield: float = 0.80

    @property
    def card_mm(self) -> tuple[float, float]:
        return (self.width_mm, self.height_mm)


@dataclass(frozen=True)
class Card:
    window: Window
    corners: Quad
    edges: dict[str, EdgeFit]
    dpi: float
    warnings: list[str] = field(default_factory=list)

    @property
    def top_w_mm(self) -> float:
        return px_to_mm(self.corners.top, self.dpi)

    @property
    def bottom_w_mm(self) -> float:
        return px_to_mm(self.corners.bottom, self.dpi)

    @property
    def left_h_mm(self) -> float:
        return px_to_mm(self.corners.left, self.dpi)

    @property
    def right_h_mm(self) -> float:
        return px_to_mm(self.corners.right, self.dpi)

    @property
    def width_mm(self) -> float:
        return 0.5 * (self.top_w_mm + self.bottom_w_mm)

    @property
    def height_mm(self) -> float:
        return 0.5 * (self.left_h_mm + self.right_h_mm)

    @property
    def aspect(self) -> float:
        return self.width_mm / self.height_mm

    @property
    def skew_deg(self) -> float:
        """Rotation of the card's top edge from horizontal in degrees."""
        return self.edges["top"].line.angle_deg()

    def to_dict(self) -> dict:
        return {
            "index": self.window.index,
            "row": self.window.row,
            "col": self.window.col,
            "width_mm": round(self.width_mm, 4),
            "height_mm": round(self.height_mm, 4),
            "aspect": round(self.aspect, 5),
            "skew_deg": round(self.skew_deg, 4),
            "opposite_width_delta_mm": round(abs(self.top_w_mm - self.bottom_w_mm), 5),
            "opposite_height_delta_mm": round(abs(self.left_h_mm - self.right_h_mm), 5),
            "corners_px": [[round(float(v), 3) for v in c] for c in np.asarray(self.corners)],
            "window_opening_mm": [
                round(self.window.opening_w_mm, 4),
                round(self.window.opening_h_mm, 4),
            ],
            "edges": {
                e: {
                    "gap_mm": round(f.median_gap_mm, 4),
                    "step_luma": round(f.median_step, 2),
                    "scanline_yield": round(f.yield_frac, 3),
                    "residual_sd_mm": round(f.residual_sd_mm, 5),
                }
                for e, f in self.edges.items()
            },
            "warnings": list(self.warnings),
        }


def edge_bands(window: Window, spec: CardSpec) -> dict[str, float]:
    """How deep to search for the card on each edge of this window."""
    slack_w = max(0.0, window.opening_w_mm - spec.width_mm) + spec.gap_margin_mm
    slack_h = max(0.0, window.opening_h_mm - spec.height_mm) + spec.gap_margin_mm
    return {"top": slack_h, "bottom": slack_h, "left": slack_w, "right": slack_w}


def _scanlines(window: Window, edge: str, inset: float = 0.15) -> np.ndarray:
    """Scanline coordinates across an edge (inset past the card's corner radius)"""
    x, y, w, h = window.bbox
    lo, hi = (x, x + w) if edge in ("top", "bottom") else (y, y + h)
    span = hi - lo
    return np.arange(int(lo + inset * span), int(hi - inset * span))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _scanline_yield(attempt: EdgeAttempt, spec: CardSpec) -> str:
    """How many scanlines hit"""
    misses = sorted(
        ((r, n) for r, n in attempt.counts().items() if r is not Reason.HIT and n),
        key=lambda rn: -rn[1],
    )

    text = (
        f"{attempt.edge}: {attempt.n_hit} of {attempt.n_scanlines} scanlines hit "
        f"({100 * attempt.yield_frac:.0f}%, need {100 * spec.min_yield:.0f}%)"
    )

    if misses:
        text += " — " + ", ".join(f"{r} {n}" for r, n in misses)

    return text


def _check(
    window: Window, attempts: dict[str, EdgeAttempt], card: Card | None, spec: CardSpec
) -> list[str]:
    """Check if we found a card in a window."""
    bad = []
    bands = edge_bands(window, spec)
    for e, a in attempts.items():
        f = a.fit
        if f is None:
            bad.append(_scanline_yield(a, spec))
            continue

        if not spec.gap_min_mm <= f.median_gap_mm <= bands[e]:
            bad.append(
                f"{e} edge sits {f.median_gap_mm:.3f}mm from the wall, outside "
                f"[{spec.gap_min_mm:.2f}, {bands[e]:.2f}]mm"
            )

        if f.median_step < STEP_FLOOR:
            bad.append(
                f"{e} edge step is only {f.median_step:.1f} luma "
                f"(need at least {STEP_FLOOR:.1f})"
            )

        if f.yield_frac < spec.min_yield:
            bad.append(_scanline_yield(a, spec))

    if card is None:
        return bad

    dw = abs(card.top_w_mm - card.bottom_w_mm)
    dh = abs(card.left_h_mm - card.right_h_mm)
    if dw > spec.opposite_tol_mm:
        bad.append(
            f"top and bottom edges disagree on width by {1000 * dw:.0f}um "
            f"(limit {1000 * spec.opposite_tol_mm:.0f}um) — this is not a rectangle"
        )

    if dh > spec.opposite_tol_mm:
        bad.append(
            f"left and right edges disagree on height by {1000 * dh:.0f}um "
            f"(limit {1000 * spec.opposite_tol_mm:.0f}um) — this is not a rectangle"
        )

    if abs(card.aspect - spec.aspect) > spec.aspect_tol:
        bad.append(f"aspect {card.aspect:.4f} is outside {spec.aspect:.4f} +/- {spec.aspect_tol}")

    if abs(card.width_mm - spec.width_mm) > spec.size_tol_mm:
        bad.append(
            f"width {card.width_mm:.3f}mm is more than {spec.size_tol_mm}mm from the "
            f"configured {spec.width_mm}mm"
        )

    if abs(card.height_mm - spec.height_mm) > spec.size_tol_mm:
        bad.append(
            f"height {card.height_mm:.3f}mm is more than {spec.size_tol_mm}mm from the "
            f"configured {spec.height_mm}mm"
        )

    return bad


@dataclass(frozen=True)
class WindowResult:
    """Result of searching for card in a window"""

    window: Window
    edges: dict[str, EdgeAttempt]
    card: Card | None
    problems: list[str]


@dataclass(frozen=True)
class DetectionReport:
    """Result of a detect."""

    windows: list[WindowResult]
    jig_error: str | None
    problems: list[str]  # every problem in the scan, window-labelled

    # the settings it ran with
    spec: CardSpec
    dpi: float
    num_windows: int | None
    anchor_offset_mm: float
    strategy: EdgeStrategy
    color: FrameColor

    @property
    def cards(self) -> list[Card]:
        return [w.card for w in self.windows if w.card is not None]

    def format(self) -> str:
        """Human readable message of problems."""
        if self.jig_error is not None:
            return self.jig_error
        failed = sum(1 for w in self.windows if w.problems)
        head = (
            f"{_plural(len(self.problems), 'problem')} in {failed} of "
            f"{_plural(len(self.windows), 'window')}"
        )
        return "\n".join([head, *(f"  {p}" for p in self.problems)])


def detect_card(
    gray: np.ndarray,
    window: Window,
    spec: CardSpec,
    dpi: float = DEFAULT_DPI,
    anchor_offset_mm: float = 0.0,
    strategy: EdgeStrategy | str = DEFAULT_STRATEGY,
) -> WindowResult:
    """Search all four edges of a window, then check what was found."""
    bands = edge_bands(window, spec)
    attempts = {
        e: fit_card_edge(
            gray,
            window.wall_lines[e].line,
            e,
            _scanlines(window, e),
            dpi=dpi,
            band_mm=bands[e],
            anchor_offset_mm=anchor_offset_mm,
            strategy=strategy,
        )
        for e in EDGES
    }

    card, problems = None, []
    fits = {e: a.fit for e, a in attempts.items() if a.fit is not None}
    if len(fits) == len(EDGES):
        try:
            corners = corners_from_lines({e: f.line for e, f in fits.items()})
        except ValueError as exc:
            problems.append(f"the four edges do not meet at four corners: {exc}")
        else:
            card = Card(window=window, corners=corners, edges=fits, dpi=dpi)

    problems = _check(window, attempts, card, spec) + problems
    return WindowResult(window=window, edges=attempts, card=card, problems=problems)


def detect_report(
    bgr: np.ndarray,
    spec: CardSpec | None = None,
    dpi: float = DEFAULT_DPI,
    num_windows: int | None = None,
    anchor_offset_mm: float = 0.0,
    strategy: EdgeStrategy | str = DEFAULT_STRATEGY,
    color: FrameColor = DEFAULT_FRAME_COLOR,
) -> DetectionReport:
    """Measure all windows in a scan."""
    spec = spec or CardSpec()
    try:
        strategy = EdgeStrategy(strategy)
    except ValueError as exc:
        raise DetectionError(str(exc)) from exc
    settings = dict(
        spec=spec,
        dpi=dpi,
        num_windows=num_windows,
        anchor_offset_mm=anchor_offset_mm,
        strategy=strategy,
        color=color,
    )

    # look for windows from the jig
    try:
        windows = find_windows(bgr, spec.card_mm, dpi=dpi, color=color)
    except JigError as exc:
        return DetectionReport(windows=[], jig_error=str(exc), problems=exc.problems, **settings)

    problems = []
    if num_windows is not None and len(windows) != num_windows:
        problems.append(f"expected {num_windows} windows, found {len(windows)}")

    # measure every window found
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results = [
        detect_card(gray, w, spec, dpi=dpi, anchor_offset_mm=anchor_offset_mm, strategy=strategy)
        for w in windows
    ]
    for r in results:
        problems.extend(f"{r.window.label}: {p}" for p in r.problems)

    return DetectionReport(windows=results, jig_error=None, problems=problems, **settings)


def detect(
    bgr: np.ndarray,
    spec: CardSpec | None = None,
    dpi: float = DEFAULT_DPI,
    num_windows: int | None = None,
    anchor_offset_mm: float = 0.0,
    strategy: EdgeStrategy | str = DEFAULT_STRATEGY,
    color: FrameColor = DEFAULT_FRAME_COLOR,
) -> list[Card]:
    """Detect cards in a jig scan."""
    report = detect_report(
        bgr,
        spec=spec,
        dpi=dpi,
        num_windows=num_windows,
        anchor_offset_mm=anchor_offset_mm,
        strategy=strategy,
        color=color,
    )

    if report.problems:
        raise DetectionError(report.format(), report)

    return report.cards
