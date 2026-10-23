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


@pytest.mark.parametrize("seam", [False, True], ids=["whole", "split-at-seam"])
def test_frame_colored_artwork_inside_a_window_is_not_frame(seam: bool) -> None:
    """Artwork bigger than any piece-size floor still belongs to its window."""
    px = lambda v: mm_to_px_int(v, DPI)  # noqa: E731
    origin = 8.0
    img = jig_scan((origin, origin))
    if seam:
        mid = px(origin + FRAME_W_MM / 2)
        img[:, mid - px(0.5) : mid + px(0.5)] = PLATEN_BGR
    # a blob most of a card in size, clear of the walls of the first window
    gap_x = (px(FRAME_W_MM) - 2 * px(WINDOW_W_MM)) // 3
    gap_y = (px(FRAME_H_MM) - 2 * px(WINDOW_H_MM)) // 3
    x, y = px(origin) + gap_x + px(10.0), px(origin) + gap_y + px(7.0)
    img[y : y + px(CARD_MM[1] - 10.0), x : x + px(CARD_MM[0] - 20.0)] = FRAME_BGR

    windows = find_windows(img, CARD_MM, DPI)
    assert [(w.row, w.col) for w in windows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    clean = find_windows(jig_scan((origin, origin)), CARD_MM, DPI)
    for got, want in zip(windows, clean, strict=True):
        assert got.opening_w_mm == pytest.approx(want.opening_w_mm, abs=0.05)
        assert got.opening_h_mm == pytest.approx(want.opening_h_mm, abs=0.05)


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


def _crack_left_wall(img: np.ndarray, origin_mm: tuple[float, float], r: int, c: int) -> None:
    """A hairline crack in the frame beside one window's left wall, joined to the window.

    It belongs to the window's hole but is too thin to survive the opening, so the hole
    and the opened window disagree about where that wall is on every scanline.
    """
    px = lambda v: mm_to_px_int(v, DPI)  # noqa: E731
    gap_x = (px(FRAME_W_MM) - 2 * px(WINDOW_W_MM)) // 3
    gap_y = (px(FRAME_H_MM) - 2 * px(WINDOW_H_MM)) // 3
    x = px(origin_mm[0]) + gap_x + c * (px(WINDOW_W_MM) + gap_x)
    y = px(origin_mm[1]) + gap_y + r * (px(WINDOW_H_MM) + gap_y)
    crack = x - px(5.0)
    img[y : y + px(WINDOW_H_MM), crack] = PLATEN_BGR
    img[y, crack:x] = PLATEN_BGR


def test_every_failing_wall_is_named_by_its_window() -> None:
    origin = (8.0, 8.0)
    img = jig_scan(origin)
    _crack_left_wall(img, origin, 0, 1)
    _crack_left_wall(img, origin, 1, 0)
    with pytest.raises(JigError) as exc:
        find_windows(img, CARD_MM, DPI)
    assert exc.value.problems == [
        "window 1 (r0c1) wall 'left': too few boundary samples",
        "window 2 (r1c0) wall 'left': too few boundary samples",
    ]
    assert str(exc.value).splitlines() == [
        "2 window walls not found",
        "  window 1 (r0c1) wall 'left': too few boundary samples",
        "  window 2 (r1c0) wall 'left': too few boundary samples",
    ]
