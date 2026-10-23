"""The brightest and innermost edge strategies"""

from __future__ import annotations

import numpy as np
import pytest

from deckle.edges import EdgeAttempt, EdgeFit, Reason, fit_card_edge
from deckle.geometry import Line
from deckle.units import EdgeStrategy, mm_to_px, px_to_mm

WALL_Y = 40
WIDTH = 240
WALL_LUMA = 100
CARD_LUMA = 240
ART_LUMA = 200

WALL = Line(point=np.array([0.0, float(WALL_Y)]), direction=np.array([1.0, 0.0]))
SCANLINES = np.arange(20, WIDTH - 20)
TOL_MM = px_to_mm(1.5)


def checked(a: EdgeAttempt) -> EdgeAttempt:
    """The trace's invariants, asserted on every edge search a test makes."""
    counts = a.counts()
    assert sum(counts.values()) == a.n_scanlines == len(a.trace)
    hit = a.trace.hit
    assert np.isfinite(a.trace.dist_px[hit]).all()
    assert np.isnan(a.trace.dist_px[~hit]).all()
    if a.fit is not None:
        assert counts[Reason.HIT] == len(a.fit.gap_mm) == len(a.fit.fit.inlier_mask)
    return a


def dominant(a: EdgeAttempt) -> Reason:
    """The most common reason a scanline missed."""
    misses = {r: n for r, n in a.counts().items() if r is not Reason.HIT}
    return max(misses, key=misses.__getitem__)


def attempt(img, strategy, **kw) -> EdgeAttempt:
    return checked(fit_card_edge(img, WALL, "top", SCANLINES, strategy=strategy, **kw))


def fit(img, strategy, **kw) -> EdgeFit:
    got = attempt(img, strategy, **kw).fit
    assert got is not None
    return got


def padded(gap_px: int, border_px: int = int(mm_to_px(1.5))) -> np.ndarray:
    """Wall, black gap, white border, then darker artwork inside the band."""
    img = np.full((WALL_Y + gap_px + 400, WIDTH), ART_LUMA, dtype=np.uint8)
    img[:WALL_Y] = WALL_LUMA
    img[WALL_Y : WALL_Y + gap_px] = 0
    img[WALL_Y + gap_px : WALL_Y + gap_px + border_px] = CARD_LUMA
    return img


def unpadded(ramp_px: int = 6, plateau_px: int = int(mm_to_px(1.5))) -> tuple[np.ndarray, int]:
    """Wall, a steep shadow ramp, a lit plateau, then a small step up onto the card."""
    gap_px = ramp_px + plateau_px
    img = np.full((WALL_Y + gap_px + 400, WIDTH), 225, dtype=np.uint8)
    img[:WALL_Y] = WALL_LUMA
    img[WALL_Y : WALL_Y + ramp_px] = np.linspace(20, 200, ramp_px, dtype=np.uint8)[:, None]
    img[WALL_Y + ramp_px : WALL_Y + gap_px] = 200
    return img, gap_px


@pytest.mark.parametrize("gap_mm", [0.3, 1.0, 2.5])
def test_brightest_finds_the_rise_out_of_a_black_gap(gap_mm):
    gap_px = int(mm_to_px(gap_mm))
    got = fit(padded(gap_px), EdgeStrategy.BRIGHTEST)
    assert got.yield_frac == 1.0
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < TOL_MM


def test_brightest_measures_a_flush_card():
    """With no gap, the wall-to-card step is still a rise."""
    got = fit(padded(0), EdgeStrategy.BRIGHTEST)
    assert got.yield_frac == 1.0
    assert abs(got.median_gap_mm) < TOL_MM


def test_brightest_never_picks_a_fall():
    """The wall and the artwork boundary are both falls, however large."""
    gap_px = int(mm_to_px(1.0))
    img = padded(gap_px)
    img[:WALL_Y] = 255  # a wall brighter than the card: a -255 fall into the gap
    got = fit(img, EdgeStrategy.BRIGHTEST)
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < TOL_MM


def test_innermost_picks_the_artwork_on_a_padded_jig():
    """Why `brightest` is the default once the pad is in."""
    gap_px = int(mm_to_px(1.0))
    got = fit(padded(gap_px), EdgeStrategy.INNERMOST)
    assert got.median_gap_mm > px_to_mm(gap_px) + 0.3


def test_innermost_finds_the_card_past_the_shadow_ramp():
    img, gap_px = unpadded()
    got = fit(img, EdgeStrategy.INNERMOST)
    assert got.yield_frac == 1.0
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < TOL_MM


def test_brightest_picks_the_shadow_ramp_on_an_unpadded_jig():
    """Why `innermost` still exists."""
    img, gap_px = unpadded()
    got = fit(img, EdgeStrategy.BRIGHTEST)
    assert got.median_gap_mm < px_to_mm(gap_px) - 0.5


@pytest.mark.parametrize("strategy", [EdgeStrategy.BRIGHTEST, EdgeStrategy.INNERMOST])
def test_dust_in_the_gap_is_not_a_card_edge(strategy):
    gap_px = int(mm_to_px(2.0))
    img = padded(gap_px, border_px=200)
    speck = WALL_Y + int(mm_to_px(0.5))
    img[speck : speck + 6] = 150
    got = fit(img, strategy)
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < TOL_MM


def test_a_displaced_anchor_still_reports_the_gap_from_the_wall():
    gap_px = int(mm_to_px(2.0))
    got = fit(padded(gap_px), EdgeStrategy.BRIGHTEST, anchor_offset_mm=0.5)
    assert abs(got.median_gap_mm - px_to_mm(gap_px)) < TOL_MM


def test_brightest_refuses_an_empty_window():
    img = np.zeros((WALL_Y + 400, WIDTH), dtype=np.uint8)
    img[:WALL_Y] = WALL_LUMA
    got = attempt(img, EdgeStrategy.BRIGHTEST)
    assert got.fit is None
    assert dominant(got) is Reason.WEAK_STEP


def test_innermost_reports_the_wall_in_an_empty_window():
    """Unsigned, so the fall into the gap is a step. The caller's gap gate refuses it."""
    img = np.zeros((WALL_Y + 400, WIDTH), dtype=np.uint8)
    img[:WALL_Y] = WALL_LUMA
    assert abs(fit(img, EdgeStrategy.INNERMOST).median_gap_mm) < TOL_MM


def test_brightest_refuses_a_step_into_texture():
    """A rise onto something that does not stay level is not a card."""
    gap_px = int(mm_to_px(1.0))
    img = padded(gap_px, border_px=400)
    stripes = WALL_Y + gap_px + 8
    for y in range(stripes, stripes + 60, 6):
        img[y : y + 3] = 140
    got = attempt(img, EdgeStrategy.BRIGHTEST)
    assert got.fit is None
    assert dominant(got) is Reason.NOT_FLAT


def test_innermost_refuses_a_window_with_no_step():
    img = np.full((WALL_Y + 400, WIDTH), WALL_LUMA, dtype=np.uint8)
    got = attempt(img, EdgeStrategy.INNERMOST)
    assert got.fit is None
    assert dominant(got) is Reason.NO_STEP


def test_a_scanline_that_runs_off_the_image_is_recorded():
    img, _ = unpadded()
    wall = Line(point=np.array([0.0, 2.0]), direction=np.array([1.0, 0.0]))
    got = checked(fit_card_edge(img, wall, "top", SCANLINES, strategy=EdgeStrategy.INNERMOST))
    assert got.counts()[Reason.OFF_IMAGE] == len(SCANLINES)


def test_the_trace_records_where_each_hit_landed():
    gap_px = int(mm_to_px(1.0))
    got = attempt(padded(gap_px), EdgeStrategy.BRIGHTEST)
    assert got.counts()[Reason.HIT] == len(SCANLINES)
    assert (got.trace.at == SCANLINES).all()
    assert np.allclose(got.trace.anchor, WALL_Y)
    assert (got.trace.d >= 0).all()
    assert np.allclose(got.trace.dist_px, gap_px, atol=1.5)
