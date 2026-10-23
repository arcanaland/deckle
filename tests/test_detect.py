"""Detection and rectification on a synthetic padded jig scan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from deckle.detect import CardSpec, DetectionError, detect, detect_report
from deckle.edges import Reason
from deckle.geometry import Quad
from deckle.jig import DEFAULT_FRAME_COLOR
from deckle.rectify import master_name, master_size_px, rectify_all, rectify_card
from deckle.units import mm_to_px, px_to_mm

DPI = 300.0
CARD_MM = (40.0, 70.0)
SPEC = CardSpec(width_mm=CARD_MM[0], height_mm=CARD_MM[1], aspect=CARD_MM[0] / CARD_MM[1])
WINDOW_MM = (44.0, 74.0)
WALL_MM = 8.0
MARGIN_MM = 15.0  # platen showing around the frame
BORDER_MM = 3.0

PLATEN_BGR = (250, 250, 250)
FRAME_BGR = (60, 160, 60)
PAD_BGR = (15, 15, 15)
BORDER_BGR = (240, 240, 240)
ART_BGR = (120, 90, 150)

SS = 16  # supersampling factor for drawing cards
SHIFT = 4  # fractional bits for vertices on the supersampled grid


@dataclass(frozen=True)
class Placed:
    """One card in its window."""

    dx_mm: float = 0.0
    dy_mm: float = 0.0
    angle_deg: float = 0.0
    size_mm: tuple[float, float] = CARD_MM

    # Extra width on the bottom edge (to make a card that is not a rectangle)
    bottom_extra_mm: float = 0.0


def _poly(cx: float, cy: float, w: float, h: float, angle: float, bottom_extra: float = 0.0):
    """A rotated quad clockwise from the top-left, in pixels (pixel i spans [i, i+1))."""
    hw, hh, be = w / 2, h / 2, bottom_extra / 2
    local = np.array([[-hw, -hh], [hw, -hh], [hw + be, hh], [-hw - be, hh]])
    t = np.deg2rad(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return local @ rot.T + [cx, cy]


def _fill(img: np.ndarray, pts: np.ndarray, bgr) -> None:
    """Blend a quad in by its exact pixel coverage, found by supersampling."""
    x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 1
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
    fine = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS), np.uint8)
    # fine pixel j's centre is at coarse (j + 0.5) / SS
    local = ((pts - [x0, y0]) * SS - 0.5) * (1 << SHIFT)
    cv2.fillPoly(fine, [np.round(local).astype(np.int32)], 255, shift=SHIFT)
    cover = cv2.resize(fine, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA) / 255.0
    patch = img[y0:y1, x0:x1].astype(np.float64)
    patch += cover[..., None] * (np.array(bgr, np.float64) - patch)
    img[y0:y1, x0:x1] = np.round(patch).astype(np.uint8)


def jig_scene(grid: list[list[Placed | None]]) -> np.ndarray:
    """A padded jig with one card per window; None leaves the window empty."""
    px = lambda v: mm_to_px(v, DPI)  # noqa: E731
    rows, cols = len(grid), len(grid[0])
    frame_w = cols * WINDOW_MM[0] + (cols + 1) * WALL_MM
    frame_h = rows * WINDOW_MM[1] + (rows + 1) * WALL_MM
    img = np.full(
        (round(px(frame_h + 2 * MARGIN_MM)), round(px(frame_w + 2 * MARGIN_MM)), 3),
        PLATEN_BGR,
        np.uint8,
    )
    x0 = y0 = MARGIN_MM
    img[round(px(y0)) : round(px(y0 + frame_h)), round(px(x0)) : round(px(x0 + frame_w))] = (
        FRAME_BGR
    )
    for r, row in enumerate(grid):
        for c, card in enumerate(row):
            wx = x0 + WALL_MM + c * (WINDOW_MM[0] + WALL_MM)
            wy = y0 + WALL_MM + r * (WINDOW_MM[1] + WALL_MM)
            img[round(px(wy)) : round(px(wy + WINDOW_MM[1])), round(px(wx)) : round(px(wx + WINDOW_MM[0]))] = PAD_BGR  # fmt: skip
            if card is None:
                continue
            cx = px(wx + WINDOW_MM[0] / 2 + card.dx_mm)
            cy = px(wy + WINDOW_MM[1] / 2 + card.dy_mm)
            w, h = px(card.size_mm[0]), px(card.size_mm[1])
            be = px(card.bottom_extra_mm)
            _fill(img, _poly(cx, cy, w, h, card.angle_deg, be), BORDER_BGR)
            b = px(BORDER_MM)
            _fill(img, _poly(cx, cy, w - 2 * b, h - 2 * b, card.angle_deg, be), ART_BGR)
    return img


def run(img: np.ndarray, spec: CardSpec = SPEC, **kw):
    return detect(img, spec, dpi=DPI, **kw)


TWO = [
    [Placed(dx_mm=-0.6, dy_mm=0.4, angle_deg=0.3), Placed(dx_mm=0.8, dy_mm=-0.5, angle_deg=-0.2)]
]


@pytest.fixture(scope="module")
def two() -> np.ndarray:
    return jig_scene(TWO)


@pytest.fixture(scope="module")
def cards(two):
    return run(two, num_windows=2)


# --- geometry -------------------------------------------------------------------------


def test_every_card_is_found_in_slot_order(cards) -> None:
    assert [(c.window.index, c.window.row, c.window.col) for c in cards] == [(0, 0, 0), (1, 0, 1)]


def test_cards_measure_their_drawn_size(cards) -> None:
    for c in cards:
        assert c.width_mm == pytest.approx(CARD_MM[0], abs=0.02)
        assert c.height_mm == pytest.approx(CARD_MM[1], abs=0.02)


def test_skew_is_measured(cards) -> None:
    for c, placed in zip(cards, TWO[0], strict=True):
        assert c.skew_deg == pytest.approx(placed.angle_deg, abs=0.01)


def test_a_card_flush_against_its_wall_is_still_measured() -> None:
    slack = (WINDOW_MM[0] - CARD_MM[0]) / 2
    (card,) = run(jig_scene([[Placed(dx_mm=-slack)]]))
    assert card.edges["left"].median_gap_mm == pytest.approx(0.0, abs=0.05)
    assert card.width_mm == pytest.approx(CARD_MM[0], abs=0.02)


def test_card_reports_its_diagnostics(cards) -> None:
    d = cards[0].to_dict()
    assert set(d["edges"]) == {"top", "bottom", "left", "right"}
    assert all("residual_sd_mm" in e for e in d["edges"].values())


# --- the anchor -----------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-0.25, -0.50])
def test_detection_is_independent_of_the_search_anchor(two, cards, offset) -> None:
    """Moving the search start outward must not move the answer."""
    for base, moved in zip(cards, run(two, anchor_offset_mm=offset), strict=True):
        assert moved.width_mm == pytest.approx(base.width_mm, abs=5e-4)
        assert moved.height_mm == pytest.approx(base.height_mm, abs=5e-4)


@pytest.mark.parametrize("offset", [2.5, 4.0])
def test_a_badly_aimed_detector_refuses(two, offset) -> None:
    """Aimed past the card's edge, it must refuse rather than return a plausible crop."""
    with pytest.raises(DetectionError):
        run(two, anchor_offset_mm=offset)


# --- the gates that fire --------------------------------------------------------------


def test_wrong_window_count_is_a_hard_failure(two) -> None:
    with pytest.raises(DetectionError, match="expected 4 windows, found 2"):
        run(two, num_windows=4)


def test_a_scan_with_no_jig_is_refused() -> None:
    img = np.full((600, 600, 3), PLATEN_BGR, np.uint8)
    img[100:400, 100:300] = BORDER_BGR
    with pytest.raises(DetectionError, match="no frame-colored pixels"):
        run(img)


def test_an_empty_window_is_refused() -> None:
    with pytest.raises(DetectionError, match="window 1"):
        run(jig_scene([[Placed(), None]]))


def test_a_card_that_is_not_a_rectangle_is_refused() -> None:
    img = jig_scene([[Placed(bottom_extra_mm=0.4)]])
    with pytest.raises(DetectionError, match="disagree on width .* not a rectangle"):
        run(img)


def test_a_card_of_the_wrong_size_is_refused() -> None:
    img = jig_scene([[Placed(size_mm=(38.0, 66.5))]])
    with pytest.raises(DetectionError, match="height .* from the configured"):
        run(img)


def test_a_card_of_the_wrong_aspect_is_refused(two) -> None:
    with pytest.raises(DetectionError, match="aspect"):
        run(two, replace(SPEC, aspect=SPEC.aspect + 0.02))


def test_every_failing_window_is_reported_at_once() -> None:
    img = jig_scene([[Placed(bottom_extra_mm=0.4), Placed(bottom_extra_mm=0.4)]])
    with pytest.raises(DetectionError, match="(?s)window 0 \\(r0c0\\): .*window 1 \\(r0c1\\): "):
        run(img)


# --- collect, then fail ---------------------------------------------------------------


def test_every_window_and_edge_is_measured_before_failing() -> None:
    """Two broken windows of four: both reported, all sixteen edges searched."""
    img = jig_scene([[Placed(), None], [Placed(bottom_extra_mm=0.4), Placed()]])
    report = detect_report(img, SPEC, dpi=DPI)
    assert [len(w.edges) for w in report.windows] == [4, 4, 4, 4]
    assert [w.card is not None for w in report.windows] == [True, False, True, True]
    assert [bool(w.problems) for w in report.windows] == [False, True, True, False]
    assert len(report.cards) == 3

    with pytest.raises(DetectionError) as exc:
        run(img)
    assert exc.value.report is not None
    lines = str(exc.value).splitlines()
    assert lines[0] == f"{len(lines) - 1} problems in 2 of 4 windows"
    assert all(ln.startswith(("  window 1 (r0c1): ", "  window 2 (r1c0): ")) for ln in lines[1:])
    assert any("not a rectangle" in ln for ln in lines)


def test_an_edge_that_misses_reports_its_reason_counts() -> None:
    report = detect_report(jig_scene([[None]]), SPEC, dpi=DPI)
    (window,) = report.windows
    left = window.edges["left"]
    assert left.fit is None
    counts = {r: n for r, n in left.counts().items() if n}
    assert counts == {Reason.WEAK_STEP: left.n_scanlines}
    assert (
        f"  window 0 (r0c0): left: 0 of {left.n_scanlines} scanlines hit (0%, need 80%) — weak_step {left.n_scanlines}"
        in report.format().splitlines()
    )


def test_a_short_count_is_listed_first_and_every_window_is_still_measured() -> None:
    img = jig_scene([[Placed(), None, Placed()]])
    with pytest.raises(DetectionError) as exc:
        run(img, num_windows=4)
    lines = str(exc.value).splitlines()
    assert lines[1] == "  expected 4 windows, found 3"
    assert lines[2].startswith("  window 1 (r0c1): ")
    assert lines[0] == f"{len(lines) - 1} problems in 1 of 3 windows"
    assert len(exc.value.report.windows) == 3


def test_a_failed_jig_is_reported_with_no_windows() -> None:
    img = np.full((600, 600, 3), PLATEN_BGR, np.uint8)
    report = detect_report(img, SPEC, dpi=DPI)
    assert report.windows == []
    assert report.jig_error is not None and "no frame-colored pixels" in report.jig_error
    assert report.problems == [report.jig_error]


def test_success_returns_the_same_cards_as_the_report(two, cards) -> None:
    report = detect_report(two, SPEC, dpi=DPI, num_windows=2)
    assert report.problems == []
    assert [c.corners.top for c in report.cards] == [c.corners.top for c in cards]


# --- the gate that deliberately does not ----------------------------------------------


def test_fit_residual_is_reported_but_never_gates() -> None:
    """A ragged edge inflates the residual. The card is still the right size."""
    img = jig_scene([[Placed()]])
    px = lambda v: round(mm_to_px(v, DPI))  # noqa: E731
    # notch the card's top edge: every other 2mm, 0.15mm of border goes dark
    x0 = px(MARGIN_MM + WALL_MM + WINDOW_MM[0] / 2 - CARD_MM[0] / 2)
    y_top = px(MARGIN_MM + WALL_MM + WINDOW_MM[1] / 2 - CARD_MM[1] / 2)
    for x in range(x0, x0 + px(CARD_MM[0]), 2 * px(2.0)):
        img[y_top : y_top + px(0.15), x : x + px(2.0)] = PAD_BGR
    (card,) = run(img)
    clean = run(jig_scene([[Placed()]]))[0]
    assert card.edges["top"].residual_sd_mm > 5 * clean.edges["top"].residual_sd_mm
    assert card.height_mm == pytest.approx(CARD_MM[1], abs=0.15)


# --- masters --------------------------------------------------------------------------


def test_masters_are_portrait_at_the_measured_size(two, cards) -> None:
    for c in cards:
        img = rectify_card(two, c)
        h, w = img.shape[:2]
        assert (w, h) == master_size_px(c)
        assert h > w
        assert px_to_mm(w, DPI) == pytest.approx(c.width_mm, abs=px_to_mm(1.0, DPI))


def test_master_edges_are_card_not_jig(two, cards) -> None:
    """A strip just inside each master's edge must be card border, never pad or wall.

    Corners are excluded: a real card's are rounded, and that is the alpha mask's job.
    """
    for c in cards:
        img = rectify_card(two, c)
        h, w = img.shape[:2]
        inset = int(0.08 * w)
        for strip in (
            img[2, inset : w - inset],
            img[h - 3, inset : w - inset],
            img[inset : h - inset, 2],
            img[inset : h - inset, w - 3],
        ):
            assert DEFAULT_FRAME_COLOR.mask(strip.reshape(1, -1, 3)).mean() == 0
            assert strip.min() > 200


def test_a_landscape_card_is_refused_at_rectify(two, cards) -> None:
    """A master must be portrait. Turning the corners a quarter-turn makes one that is not."""
    c = cards[0].corners
    quarter_turn = Quad(c.bottom_left, c.top_left, c.top_right, c.bottom_right)
    turned = replace(cards[0], corners=quarter_turn)
    assert turned.width_mm > turned.height_mm
    with pytest.raises(ValueError, match="not portrait"):
        rectify_card(two, turned)


def test_masters_are_named_by_slot_not_identity(two, cards, tmp_path: Path) -> None:
    written = rectify_all(two, cards, Path("scan_007.png"), tmp_path / "out")
    assert [p.name for p in written] == ["scan_007_r0c0.png", "scan_007_r0c1.png"]
    assert [master_name(Path("x/scan_007.jpg"), c) for c in cards] == [p.name for p in written]
    for p, c in zip(written, cards, strict=True):
        assert cv2.imread(str(p)).shape[:2] == master_size_px(c)[::-1]
