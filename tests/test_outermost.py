"""The outermost edge strategy

Every scene is a dark-bordered card on a padded jig, walking inward from the wall:

    wall ~100 | void 0 | dark card border ~20 | brighter artwork

"""

from __future__ import annotations

import numpy as np
import pytest

from deckle.edges import BAND_MM, VOID_CEIL, VOID_RUN_MM, fit_card_edge
from deckle.geometry import Line
from deckle.units import EdgeStrategy, mm_to_px, px_to_mm

WALL_Y = 40  # where the jig wall ends and the clearance gap begins
WIDTH = 240  # scanlines across the edge
WALL_LUMA = 100.0
BORDER_LUMA = 20.0
ART_LUMA = 200.0


def scene(
    gap_px: int,
    *,
    border_px: int = 60,
    art_luma: float = ART_LUMA,
    void_luma: float = 0.0,
    speck: tuple[int, int] | None = None,
) -> np.ndarray:
    """A synthetic top edge: wall then void then card."""
    h = WALL_Y + gap_px + border_px + 400
    img = np.full((h, WIDTH), void_luma, dtype=np.uint8)
    img[:WALL_Y, :] = WALL_LUMA
    card = WALL_Y + gap_px
    img[card : card + border_px, :] = BORDER_LUMA
    img[card + border_px :, :] = art_luma
    if speck is not None:
        start, length = speck
        img[WALL_Y + start : WALL_Y + start + length, :] = 120
    return img


WALL = Line(point=np.array([0.0, float(WALL_Y)]), direction=np.array([1.0, 0.0]))
SCANLINES = np.arange(20, WIDTH - 20)


def fit(img, strategy=EdgeStrategy.OUTERMOST, **kw):
    return fit_card_edge(img, WALL, "top", SCANLINES, strategy=strategy, **kw)


def test_strategies_parse_from_their_config_spelling():
    assert EdgeStrategy("outermost") is EdgeStrategy.OUTERMOST
    assert f"{EdgeStrategy.OUTERMOST}" == "outermost"


def test_finds_the_edge_at_the_end_of_the_void():
    gap_px = int(mm_to_px(1.5))
    got = fit(scene(gap_px))
    assert got.yield_frac == 1.0
    # Placed on the gradient centroid, so within a pixel of the step this file built.
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < px_to_mm(1.5)


@pytest.mark.parametrize("gap_mm", [0.3, 0.8, 1.5, 2.5])
def test_edge_is_found_wherever_the_card_sits_in_the_band(gap_mm):
    gap_px = int(mm_to_px(gap_mm))
    got = fit(scene(gap_px))
    assert got.yield_frac == 1.0
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < px_to_mm(1.5)


def test_bright_artwork_deeper_in_the_band_does_not_win():
    gap_px = int(mm_to_px(1.0))
    # A 0.4mm border puts the artwork step (+180) inside the band, outscoring the card
    # edge (+20).
    img = scene(gap_px, border_px=int(mm_to_px(0.4)))
    outer = fit(img)
    bright = fit(img, strategy=EdgeStrategy.BRIGHTEST)
    assert abs(outer.median_gap_mm - px_to_mm(gap_px)) < px_to_mm(1.5)
    assert bright.median_gap_mm > outer.median_gap_mm + 0.3


def test_a_flush_card_yields_nothing_rather_than_the_wall():
    """No void run, no edge, and no fallback to the wall line."""
    with pytest.raises(ValueError, match="no card edge here"):
        fit(scene(0))


def test_a_void_run_shorter_than_the_minimum_is_not_an_edge():
    short = int(mm_to_px(VOID_RUN_MM)) - 1
    with pytest.raises(ValueError, match="no card edge here"):
        fit(scene(short))


def test_a_run_at_the_minimum_is_an_edge():
    ok = int(round(mm_to_px(VOID_RUN_MM))) + 1
    got = fit(scene(ok))
    assert got.yield_frac == 1.0


def test_a_gap_above_the_ceiling_yields_nothing():
    """A gap that is dark but not void, as an unpadded or lighter-padded jig gives."""
    gap_px = int(mm_to_px(1.5))
    assert fit(scene(gap_px, void_luma=6.0)).yield_frac == 1.0
    with pytest.raises(ValueError, match="no card edge here"):
        fit(scene(gap_px, void_luma=VOID_CEIL + 4.0))


def test_a_speck_of_dust_in_the_gap_is_not_a_card_edge():
    """A lit run that returns to void fails the sustain check."""
    gap_px = int(mm_to_px(2.0))
    speck_at = int(mm_to_px(0.5))
    got = fit(scene(gap_px, speck=(speck_at, 6)))
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < px_to_mm(1.5)


def test_dark_artwork_deeper_in_the_band_does_not_steal_the_run():
    """A black line inside the card is void by the ceiling's definition and deeper in the
    band than the gap, but too short to count as a run."""
    gap_px = int(mm_to_px(1.2))
    img = scene(gap_px, border_px=int(mm_to_px(0.6)))
    at = WALL_Y + gap_px + int(mm_to_px(0.8))
    img[at : at + 3, :] = 0
    assert BAND_MM > 2.0  # the line really is inside the search band
    got = fit(img)
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < px_to_mm(1.5)


def test_the_step_floor_still_applies():
    """A void run that ends in something too faint to be a card is not a card edge."""
    gap_px = int(mm_to_px(1.5))
    img = scene(gap_px, border_px=400, art_luma=6.0)
    img[WALL_Y + gap_px :, :] = 6  # "card" only 6 luma above the void: under the floor
    with pytest.raises(ValueError, match="no card edge here"):
        fit(img)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="not a valid EdgeStrategy"):
        fit(scene(int(mm_to_px(1.5))), strategy="outermsot")


def test_a_plain_string_is_accepted():
    gap_px = int(mm_to_px(1.5))
    assert fit(scene(gap_px), strategy="outermost").yield_frac == 1.0


def test_innermost_lands_on_the_artwork_in_this_scene():
    """Artwork is innermost, so `innermost` must pick it here."""
    gap_px = int(mm_to_px(1.0))
    img = scene(gap_px, border_px=int(mm_to_px(0.4)))
    got = fit(img, strategy=EdgeStrategy.INNERMOST)
    assert got.median_gap_mm > px_to_mm(gap_px) + 0.3
