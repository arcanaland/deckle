"""Turn a jig scan into per-card geometry, or fail loudly.

The failure policy is RFC-001's and is not negotiable: a wrong window count, or any card
outside tolerance, is a hard error. Never a guess, never a best-effort crop. A silently
mis-detected card is the one error class that survives all the way into the emitted deck,
and the review screen will not catch it — a person clicking through thumbnails cannot see
that a card is 0.4mm short.

Note what is *not* used as a gate: fit residual. The known-bad fit in RFC-001 reported a
6.8um residual sd on an edge that was 376um wrong and physically absent. Residual is
reported as a diagnostic and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .edges import EdgeFit, fit_card_edge
from .geometry import intersect
from .jig import EDGES, JigError, Window, find_windows
from .units import DEFAULT_DPI, px_to_mm


class DetectionError(RuntimeError):
    """Raised when a scan yields no trustworthy card geometry."""


@dataclass(frozen=True)
class CardSpec:
    """The card this project is packaging. Every gate is relative to these."""

    width_mm: float = 70.0
    height_mm: float = 120.0
    # RFC-001: the sample deck is 0.583, not the deck spec's 0.5789 default. Configurable
    # per project; the detector must never hardcode the spec default.
    aspect: float = 0.5843
    aspect_tol: float = 0.01
    size_tol_mm: float = 1.5
    # Opposite edges of one card must agree this closely, or the "rectangle" is not one.
    opposite_tol_mm: float = 0.150
    gap_min_mm: float = -0.10
    # Added to a window's measured slack to get the search band and the gap gate. It
    # absorbs wall-fit error and card trim variation, and nothing else: every extra
    # millimetre here is another millimetre in which the detector may find artwork.
    gap_margin_mm: float = 0.70
    min_yield: float = 0.80


@dataclass(frozen=True)
class Card:
    window: Window
    corners: np.ndarray  # (4,2) TL, TR, BR, BL in image coordinates
    edges: dict[str, EdgeFit]
    dpi: float
    warnings: list[str] = field(default_factory=list)

    @property
    def top_w_mm(self) -> float:
        return px_to_mm(float(np.linalg.norm(self.corners[1] - self.corners[0])), self.dpi)

    @property
    def bottom_w_mm(self) -> float:
        return px_to_mm(float(np.linalg.norm(self.corners[2] - self.corners[3])), self.dpi)

    @property
    def left_h_mm(self) -> float:
        return px_to_mm(float(np.linalg.norm(self.corners[3] - self.corners[0])), self.dpi)

    @property
    def right_h_mm(self) -> float:
        return px_to_mm(float(np.linalg.norm(self.corners[2] - self.corners[1])), self.dpi)

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
        """Rotation of the card's top edge from horizontal, in degrees."""
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
            "corners_px": [[round(float(v), 3) for v in c] for c in self.corners],
            "window_opening_mm": [
                round(self.window.opening_w_mm, 4),
                round(self.window.opening_h_mm, 4),
            ],
            "edges": {
                e: {
                    "gap_mm": round(f.median_gap_mm, 4),
                    "step_luma": round(f.median_step, 2),
                    "scanline_yield": round(f.yield_frac, 3),
                    # Diagnostic only. Never a gate. See module docstring.
                    "residual_sd_mm": round(f.residual_sd_mm, 5),
                }
                for e, f in self.edges.items()
            },
            "warnings": list(self.warnings),
        }


def edge_bands(window: Window, spec: CardSpec) -> dict[str, float]:
    """How deep to search for the card on each edge of this window.

    Derived from the window's own measured opening, never from a constant: a card is free
    to sit against one wall, which puts the whole of the window's slack on the opposite
    edge. The reference jig's slack is ~2.1mm and the assembled jig's ~2.9mm, and a fixed
    band sized for the first cannot see the card in the second.
    """
    slack_w = max(0.0, window.opening_w_mm - spec.width_mm) + spec.gap_margin_mm
    slack_h = max(0.0, window.opening_h_mm - spec.height_mm) + spec.gap_margin_mm
    return {"top": slack_h, "bottom": slack_h, "left": slack_w, "right": slack_w}


def _scanlines(window: Window, edge: str, inset: float = 0.15) -> np.ndarray:
    """Scanline coordinates across an edge, insetting past the card's corner radius."""
    x, y, w, h = window.bbox
    lo, hi = (x, x + w) if edge in ("top", "bottom") else (y, y + h)
    span = hi - lo
    return np.arange(int(lo + inset * span), int(hi - inset * span))


def _check(card: Card, spec: CardSpec) -> list[str]:
    """Absolute plausibility gates. Returns a list of reasons the card is not believable."""
    bad = []
    bands = edge_bands(card.window, spec)
    for e, f in card.edges.items():
        if not spec.gap_min_mm <= f.median_gap_mm <= bands[e]:
            bad.append(
                f"{e} edge sits {f.median_gap_mm:.3f}mm from the wall, outside "
                f"[{spec.gap_min_mm:.2f}, {bands[e]:.2f}]mm — that is not a clearance gap"
            )
        if f.median_step < 0:
            bad.append(f"{e} edge step is negative")
        if f.yield_frac < spec.min_yield:
            bad.append(
                f"{e} edge found a step on only {100 * f.yield_frac:.0f}% of scanlines "
                f"(need {100 * spec.min_yield:.0f}%)"
            )
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


def detect_card(
    gray: np.ndarray,
    window: Window,
    spec: CardSpec,
    dpi: float = DEFAULT_DPI,
    anchor_offset_mm: float = 0.0,
) -> Card:
    fits: dict[str, EdgeFit] = {}
    bands = edge_bands(window, spec)
    for e in EDGES:
        try:
            fits[e] = fit_card_edge(
                gray,
                window.wall_lines[e].line,
                e,
                _scanlines(window, e),
                dpi=dpi,
                band_mm=bands[e],
                anchor_offset_mm=anchor_offset_mm,
            )
        except ValueError as exc:
            raise DetectionError(f"window {window.index}: {exc}") from exc

    lines = {e: f.line for e, f in fits.items()}
    corners = np.array(
        [
            intersect(lines["top"], lines["left"]),
            intersect(lines["top"], lines["right"]),
            intersect(lines["bottom"], lines["right"]),
            intersect(lines["bottom"], lines["left"]),
        ]
    )
    return Card(window=window, corners=corners, edges=fits, dpi=dpi)


def detect(
    bgr: np.ndarray,
    spec: CardSpec | None = None,
    dpi: float = DEFAULT_DPI,
    expect: int | None = None,
    anchor_offset_mm: float = 0.0,
) -> list[Card]:
    """Detect every card in a jig scan.

    `expect` hard-fails on a window count other than the one given. It is None by default
    because the reference jig is a single printed half with two windows; RFC-001's four is
    a property of the assembled jig, not of the pipeline.
    """
    spec = spec or CardSpec()
    try:
        windows = find_windows(bgr, dpi=dpi)
    except JigError as exc:
        raise DetectionError(str(exc)) from exc

    if expect is not None and len(windows) != expect:
        raise DetectionError(f"expected {expect} windows, found {len(windows)}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cards, problems = [], []
    for w in windows:
        card = detect_card(gray, w, spec, dpi=dpi, anchor_offset_mm=anchor_offset_mm)
        bad = _check(card, spec)
        if bad:
            problems.extend(f"window {w.index}: {b}" for b in bad)
        cards.append(card)
    if problems:
        raise DetectionError("; ".join(problems))
    return cards
