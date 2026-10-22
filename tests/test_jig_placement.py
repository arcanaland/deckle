from __future__ import annotations

import numpy as np
import pytest

from deckle.jig import FrameColor, JigError, find_windows
from deckle.units import mm_to_px_int

DPI = 150.0

BED_W_MM, BED_H_MM = 216.0, 297.0
FRAME_W_MM, FRAME_H_MM = 200.0, 260.0
WINDOW_W_MM, WINDOW_H_MM = 63.8, 114.0
WALL_MM = 8.0  # frame wall between a window and the frame's outer edge
CARD_MM = (59.8, 110.0)  # 2mm clearance per side

FRAME_BGR = (60, 160, 60)  # inside DEFAULT_FRAME_COLOR once converted to HSV
PLATEN_BGR = (250, 250, 250)


def jig_scan(
    origin_mm: tuple[float, float], frame_bgr=FRAME_BGR, rows: int = 2, cols: int = 2
) -> np.ndarray:
    """A rows x cols jig laid on the platen with its top-left corner at `origin_mm`."""
    px = lambda v: mm_to_px_int(v, DPI)  # noqa: E731
    img = np.full((px(BED_H_MM), px(BED_W_MM), 3), PLATEN_BGR, np.uint8)

    ox, oy = px(origin_mm[0]), px(origin_mm[1])
    img[oy : oy + px(FRAME_H_MM), ox : ox + px(FRAME_W_MM)] = frame_bgr

    # Windows spaced evenly to leave a wall between them.
    gap_x = (px(FRAME_W_MM) - cols * px(WINDOW_W_MM)) // (cols + 1)
    gap_y = (px(FRAME_H_MM) - rows * px(WINDOW_H_MM)) // (rows + 1)
    for r in range(rows):
        for c in range(cols):
            x = ox + gap_x + c * (px(WINDOW_W_MM) + gap_x)
            y = oy + gap_y + r * (px(WINDOW_H_MM) + gap_y)
            img[y : y + px(WINDOW_H_MM), x : x + px(WINDOW_W_MM)] = PLATEN_BGR
    return img


def openings(img: np.ndarray) -> list[tuple[float, float]]:
    return [(w.opening_w_mm, w.opening_h_mm) for w in find_windows(img, CARD_MM, DPI)]


def test_jig_away_from_the_edges_finds_four_windows() -> None:
    """The control."""
    assert len(openings(jig_scan((8.0, 8.0)))) == 4


@pytest.mark.parametrize(
    "origin_mm",
    [
        (0.0, 0.0),  # flush into the top-left corner
        (0.0, 8.0),  # left edge only
        (8.0, 0.0),  # top edge only
        (16.0, 37.0),  # flush right and bottom
        (0.0, 37.0),  # left, bottom: two adjacent edges, away from (0,0)
    ],
    ids=["corner", "left", "top", "right-bottom", "left-bottom"],
)
def test_jig_against_the_platen_edges_finds_four_windows(
    origin_mm: tuple[float, float],
) -> None:
    """No placement may invent a window."""
    assert len(openings(jig_scan(origin_mm))) == 4


def test_a_sliver_between_the_frame_and_the_corner_is_not_a_window() -> None:
    """Pixel (0,0) is platen, but frame cuts it off from the rest of the outside."""
    px = lambda v: mm_to_px_int(v, DPI)  # noqa: E731
    img = jig_scan((1.0, 0.0))
    img[px(50.0) : px(51.0), : px(1.0)] = FRAME_BGR  # debris bridging sliver to frame
    assert len(openings(img)) == 4


@pytest.mark.parametrize("rows, cols", [(2, 2), (2, 1), (1, 2)])
def test_windows_are_numbered_row_major_from_zero(rows: int, cols: int) -> None:
    windows = find_windows(jig_scan((8.0, 8.0), rows=rows, cols=cols), CARD_MM, DPI)
    assert [(w.index, w.row, w.col) for w in windows] == [
        (r * cols + c, r, c) for r in range(rows) for c in range(cols)
    ]


def test_placement_does_not_move_the_measured_opening() -> None:
    away = sorted(openings(jig_scan((8.0, 8.0))))
    corner = sorted(openings(jig_scan((0.0, 0.0))))
    for (aw, ah), (cw, ch) in zip(away, corner, strict=True):
        assert aw == pytest.approx(cw, abs=0.05)
        assert ah == pytest.approx(ch, abs=0.05)


def test_a_frame_with_no_holes_is_not_a_window() -> None:
    """The outside is never a window"""
    px = lambda v: mm_to_px_int(v, DPI)  # noqa: E731
    img = np.full((px(BED_H_MM), px(BED_W_MM), 3), PLATEN_BGR, np.uint8)
    img[: px(FRAME_H_MM), : px(FRAME_W_MM)] = FRAME_BGR
    with pytest.raises(Exception, match="no card windows"):
        find_windows(img, CARD_MM, DPI)


def test_a_jig_in_another_color_is_found_with_its_own_color() -> None:
    red = FrameColor(lo=(170, 60, 40), hi=(10, 255, 255))
    img = jig_scan((8.0, 8.0), frame_bgr=(40, 40, 200))
    assert len(find_windows(img, CARD_MM, DPI, color=red)) == 4

    with pytest.raises(JigError, match="no frame-colored pixels"):
        find_windows(img, CARD_MM, DPI)
