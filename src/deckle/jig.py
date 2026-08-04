"""Locate the green jig frame and its card windows.

RFC-001's jig does not register: it has no lip and no fiducials, and its only job is to
be green around each card. Detection is therefore *per window* — find the frame, find the
holes in it, and everything after that is bounded by known geometry. "What is inside this
window" is a far easier question than "what is a card anywhere in this image".

Every dimension here is derived from the mask. The print measures ~0.6% under nominal and
the two windows of one printed half disagree on wall position by 21px, so design constants
are a debugging aid only — never an anchor. See RFC-001 §"The one bad number".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Line, LineFit, fit_line_trimmed, intersect
from .units import DEFAULT_DPI, mm_to_px_int, px_to_mm

# Measured on the printed jig (RFC-001 §"The jig, measured"): H 69.6±2.4, S 195.5±31.6,
# V 159.9±11.5. This range separates frame from card, artwork and platen with no tuning.
# Deliberately NOT auto-tuned — TASK-002 forbids it.
GREEN_LO = (35, 60, 40)
GREEN_HI = (90, 255, 255)

# The finger notches are 8mm deep and 16mm wide and connect to the windows, which inflates
# a naive window bbox by 8mm. Opening removes a protrusion only when the kernel cannot fit
# inside it, so the kernel must exceed the notch *width*, not its depth: a 10mm kernel
# leaves the window reading 80.9mm instead of 72.9mm. A rectangular kernel is used because
# an ellipse of the same size still protrudes into the notch (measured: 77.4mm), and
# because a rect kernel is separable and so far cheaper on a 4960x6460 mask.
NOTCH_OPEN_MM = 20.0

# A window is 73 x 123mm nominal. Anything under a third of that area is debris.
MIN_WINDOW_AREA_MM2 = 0.33 * 73.0 * 123.0

EDGES = ("top", "bottom", "left", "right")


class JigError(RuntimeError):
    """Raised when the scan does not contain a usable jig."""


@dataclass(frozen=True)
class Window:
    """One card window: a hole in the green frame, measured from the mask."""

    index: int
    row: int
    col: int
    mask: np.ndarray  # uint8 0/255, this window only
    bbox: tuple[int, int, int, int]  # x, y, w, h — a coarse handle, never a measurement
    centroid: tuple[float, float]
    wall_lines: dict[str, LineFit]
    corners: np.ndarray  # (4,2) TL, TR, BR, BL — intersections of the wall lines
    dpi: float

    @property
    def opening_w_mm(self) -> float:
        tl, tr, br, bl = self.corners
        return px_to_mm(0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)), self.dpi)

    @property
    def opening_h_mm(self) -> float:
        tl, tr, br, bl = self.corners
        return px_to_mm(0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)), self.dpi)


def green_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, GREEN_LO, GREEN_HI)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        raise JigError("no green pixels found — is the jig in the scan?")
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == best).astype(np.uint8) * 255, int(stats[best, cv2.CC_STAT_AREA])


def _holes(frame: np.ndarray) -> np.ndarray:
    """Pixels enclosed by the frame — i.e. the windows.

    Green artwork inside a window is a *separate* component from the frame, so it is
    swallowed by the hole rather than punching a sub-hole in it. That matters: the sample
    deck has a 59x31mm green blob inside one window.
    """
    filled = frame.copy()
    h, w = filled.shape
    cv2.floodFill(filled, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_and(cv2.bitwise_not(frame), cv2.bitwise_not(filled) | frame)


def _scanline_boundary(mask: np.ndarray, edge: str, at: int) -> float | None:
    """Where `mask` ends along one scanline, on the half-pixel convention.

    The point sits between the last mask pixel and the first non-mask one, so a window
    spanning rows [a, b] measures b - a + 1 px rather than b - a.
    """
    line = mask[:, at] if edge in ("top", "bottom") else mask[at, :]
    hit = np.flatnonzero(line)
    if not hit.size:
        return None
    return float(hit.min() - 0.5) if edge in ("top", "left") else float(hit.max() + 0.5)


def _wall_boundary_points(
    hole: np.ndarray,
    opened: np.ndarray,
    bbox: tuple[int, int, int, int],
    edge: str,
    corridor_mm: float = 2.0,
    dpi: float = DEFAULT_DPI,
    inset: float = 0.1,
) -> np.ndarray:
    """Boundary points of one window wall, sampled across its middle.

    Measured on the *un-opened* hole: eroding-then-dilating a rectangle that is a fifth of
    a degree off axis clips it, and clips the more-skewed window harder, which showed up
    as the two windows of one printed half disagreeing on height by 247um where they
    should agree to tens.

    The notches come back with the un-opened hole, so they are excluded geometrically
    rather than statistically — a notch is 8mm deep and 13% of the edge, which lands
    almost exactly on a 2.5-sigma trim and so survives it about as often as not. Any
    scanline whose hole boundary is more than `corridor_mm` from the opened mask's
    boundary is simply not a wall sample.
    """
    x, y, w, h = bbox
    corridor = corridor_mm * dpi / 25.4
    lo, hi = (
        (int(x + inset * w), int(x + (1 - inset) * w))
        if edge in ("top", "bottom")
        else (int(y + inset * h), int(y + (1 - inset) * h))
    )

    pts = []
    for at in range(lo, hi):
        b = _scanline_boundary(hole, edge, at)
        ref = _scanline_boundary(opened, edge, at)
        if b is None or ref is None or abs(b - ref) > corridor:
            continue
        pts.append((at, b) if edge in ("top", "bottom") else (b, at))
    if len(pts) < 8:
        raise JigError(f"window wall {edge!r} has too few boundary samples")
    return np.array(pts, dtype=np.float64)


def _corners_from_walls(walls: dict[str, Line]) -> np.ndarray:
    return np.array(
        [
            intersect(walls["top"], walls["left"]),
            intersect(walls["top"], walls["right"]),
            intersect(walls["bottom"], walls["right"]),
            intersect(walls["bottom"], walls["left"]),
        ]
    )


def find_windows(bgr: np.ndarray, dpi: float = DEFAULT_DPI) -> list[Window]:
    """Find every card window in a jig scan, ordered row-major by centroid.

    Returns however many windows the jig actually has — the reference scan is a single
    printed half and so has two. Nothing here assumes four.
    """
    mask = green_mask(bgr)
    frame, frame_area = _largest_component(mask)

    # A frame occupies a large, connected fraction of the bed. A scan with no jig still
    # has *some* greenish pixels (artwork, noise), so require real coverage before
    # believing there is a frame at all.
    if px_to_mm(1.0, dpi) ** 2 * frame_area < 0.25 * 200.0 * 136.0:
        raise JigError(
            f"largest green region is only "
            f"{px_to_mm(1.0, dpi) ** 2 * frame_area:.0f}mm^2 — no jig found in this scan"
        )

    holes = _holes(frame)
    k = mm_to_px_int(NOTCH_OPEN_MM, dpi) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8)
    min_area_px = MIN_WINDOW_AREA_MM2 / (px_to_mm(1.0, dpi) ** 2)
    found = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]
    # Map each opened window back onto the hole it came from, so walls are measured on
    # un-eroded geometry. See _wall_boundary_points.
    hole_n, hole_labels, _, _ = cv2.connectedComponentsWithStats(holes, 8)
    if not found:
        raise JigError("green frame found but it has no card windows")

    # Row-major by centroid. Windows are a grid, so cluster rows by y before sorting x.
    ys = np.array([centroids[i][1] for i in found])
    row_gap = mm_to_px_int(60.0, dpi)  # half a card height: unambiguous row separator
    order = np.argsort(ys)
    rows: list[list[int]] = []
    for oi in order:
        i = found[oi]
        if rows and abs(centroids[i][1] - centroids[rows[-1][-1]][1]) < row_gap:
            rows[-1].append(i)
        else:
            rows.append([i])
    for r in rows:
        r.sort(key=lambda i: centroids[i][0])

    windows: list[Window] = []
    idx = 0
    for ri, r in enumerate(rows):
        for ci, i in enumerate(r):
            opened_mask = (labels == i).astype(np.uint8) * 255
            bbox = tuple(int(v) for v in stats[i, :4])  # x, y, w, h
            # The hole this opened window sits inside, notches and all.
            hole_id = int(np.bincount(hole_labels[labels == i], minlength=hole_n).argmax())
            wmask = (hole_labels == hole_id).astype(np.uint8) * 255
            fits = {
                e: fit_line_trimmed(_wall_boundary_points(wmask, opened_mask, bbox, e, dpi=dpi))
                for e in EDGES
            }
            windows.append(
                Window(
                    index=idx,
                    row=ri,
                    col=ci,
                    mask=wmask,
                    bbox=bbox,  # type: ignore[arg-type]
                    centroid=(float(centroids[i][0]), float(centroids[i][1])),
                    wall_lines=fits,
                    corners=_corners_from_walls({e: f.line for e, f in fits.items()}),
                    dpi=dpi,
                )
            )
            idx += 1
    return windows
