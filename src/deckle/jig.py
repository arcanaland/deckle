"""Locate the 3d-printed jig and its card windows."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import LineFit, Quad, corners_from_lines, fit_line_trimmed
from .units import DEFAULT_DPI, mm_to_px, mm_to_px_int, px_to_mm

NOTCH_OPEN_MM = 20.0

MIN_AREA_CARDS = 0.5

EDGES = ("top", "bottom", "left", "right")


class JigError(RuntimeError):
    """Raised when the scan does not contain a usable jig."""


@dataclass(frozen=True)
class FrameColor:
    """The jig's filament as an HSV range (H 0-179, S and V 0-255)."""

    lo: tuple[int, int, int]
    hi: tuple[int, int, int]

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi) = self.lo, self.hi
        if h_lo <= h_hi:
            return cv2.inRange(hsv, self.lo, self.hi)
        return cv2.inRange(hsv, (h_lo, s_lo, v_lo), (179, s_hi, v_hi)) | cv2.inRange(
            hsv, (0, s_lo, v_lo), (h_hi, s_hi, v_hi)
        )


# my green PLA
DEFAULT_FRAME_COLOR = FrameColor(lo=(35, 60, 40), hi=(90, 255, 255))


@dataclass(frozen=True)
class Window:
    """A card window from the jig."""

    index: int
    row: int
    col: int
    mask: np.ndarray  # uint8 0/255, this window only
    bbox: tuple[int, int, int, int]  # x, y, w, h (roughly)
    centroid: tuple[float, float]
    wall_lines: dict[str, LineFit]
    corners: Quad
    dpi: float

    @property
    def opening_w_mm(self) -> float:
        return px_to_mm(self.corners.width, self.dpi)

    @property
    def opening_h_mm(self) -> float:
        return px_to_mm(self.corners.height, self.dpi)


def frame_mask(bgr: np.ndarray, color: FrameColor = DEFAULT_FRAME_COLOR) -> np.ndarray:
    return color.mask(bgr)


def _frame_pieces(mask: np.ndarray, min_area_mm2: float, dpi: float) -> np.ndarray:
    """Union of every frame-colored component big enough to be a printed piece of jig."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        raise JigError("Missing jig: no frame-colored pixels found")
    mm2_per_px = px_to_mm(1.0, dpi) ** 2
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] * mm2_per_px >= min_area_mm2]
    if not keep:
        biggest = float(stats[1:, cv2.CC_STAT_AREA].max()) * mm2_per_px
        raise JigError(f"Missing jig: largest frame-colored region is only {biggest:.0f}mm^2")

    # A frame-colored region enclosed by the rest of the frame is artwork on a card, however
    # large. Pieces of jig never enclose each other.
    if len(keep) > 1:
        frame_pieces = []
        for i in keep:
            others = np.isin(labels, [j for j in keep if j != i]).astype(np.uint8) * 255
            if not _holes(others)[labels == i].all():
                frame_pieces.append(i)
        keep = frame_pieces
    return np.isin(labels, keep).astype(np.uint8) * 255


def _holes(frame: np.ndarray) -> np.ndarray:
    """Pixels enclosed by the frame (i.e. the windows)"""
    h, w = frame.shape
    padded = cv2.copyMakeBorder(frame, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    filled = padded[1:-1, 1:-1]
    return cv2.bitwise_and(cv2.bitwise_not(frame), cv2.bitwise_not(filled) | frame)


def _scanline_boundary(mask: np.ndarray, edge: str, at: int) -> float | None:
    """Where mask ends along one scanline"""
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
    """Boundary points of one window wall sampled across its middle."""
    x, y, w, h = bbox
    corridor = mm_to_px(corridor_mm, dpi)

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


def find_windows(
    bgr: np.ndarray,
    card_mm: tuple[float, float],
    dpi: float = DEFAULT_DPI,
    color: FrameColor = DEFAULT_FRAME_COLOR,
) -> list[Window]:
    """Find every card window in a jig scan. Ordered row-major by centroid."""

    # find the jig roughly
    card_w_mm, card_h_mm = card_mm
    min_area_mm2 = MIN_AREA_CARDS * card_w_mm * card_h_mm
    frame = _frame_pieces(frame_mask(bgr, color), min_area_mm2, dpi)

    # find the windows in the jig
    holes = _holes(frame)
    k = mm_to_px_int(NOTCH_OPEN_MM, dpi) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    opened = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, 8)
    min_area_px = min_area_mm2 / (px_to_mm(1.0, dpi) ** 2)
    found = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]

    # map each window back onto the hole it came from
    hole_n, hole_labels, _, _ = cv2.connectedComponentsWithStats(holes, 8)
    if not found:
        raise JigError("frame found but it has no card windows")

    # sort row-major by centroid
    ys = np.array([centroids[i][1] for i in found])
    row_gap = mm_to_px_int(0.5 * card_h_mm, dpi)  # unambiguous row separator
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

    # for each row and column of our jig
    for ri, r in enumerate(rows):
        for ci, i in enumerate(r):
            # which connected component
            opened_mask = (labels == i).astype(np.uint8) * 255

            bbox = tuple(int(v) for v in stats[i, :4])  # x, y, w, h
            hole_id = int(np.bincount(hole_labels[labels == i], minlength=hole_n).argmax())
            wmask = (hole_labels == hole_id).astype(np.uint8) * 255

            ## for each edge of the window, fit a line from samples we find across its edge
            fits = {
                e: fit_line_trimmed(_wall_boundary_points(wmask, opened_mask, bbox, e, dpi=dpi))
                for e in EDGES
            }
            windows.append(
                Window(
                    index=len(windows),
                    row=ri,
                    col=ci,
                    mask=wmask,
                    bbox=bbox,  # type: ignore[arg-type]
                    centroid=(float(centroids[i][0]), float(centroids[i][1])),
                    wall_lines=fits,
                    corners=corners_from_lines({e: f.line for e, f in fits.items()}),
                    dpi=dpi,
                )
            )

    return windows
