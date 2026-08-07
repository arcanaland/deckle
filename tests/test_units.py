"""The measured constants, and the property that they are measured in one place.

R2 in RFC-005: `detect`, `project` and `cli` each carried their own copy of the card
geometry and two of them disagreed, so the same command was gated against a different
card inside a project than outside one. These tests pin the values against the calipers
CLAUDE.md records. The *consolidation* is checked where the copies used to be -- layer 4
for `detect`, layer 7 for `project`, layer 10 for `cli` -- since those modules do not
exist yet.
"""

from __future__ import annotations

import pytest

from deckle import units


def test_dpi_is_the_one_that_makes_the_height_pyramid_fit():
    """h2400 is the tallest variant the deck spec asks for; 300dpi cannot reach it."""
    assert units.DEFAULT_DPI == 600.0
    assert units.mm_to_px(120.0) == pytest.approx(2834.6, abs=0.1)
    assert units.mm_to_px(120.0) > 2400


def test_card_size_is_the_calipered_one():
    assert units.DEFAULT_CARD_MM == (70.0, 120.0)


def test_aspect_is_measured_and_not_the_spec_default():
    """0.5789 is the deck spec's default and is *not* this deck. Substituting it would
    put every card 0.6mm out on the long side and the detector would be right to fail."""
    assert units.DEFAULT_ASPECT == 0.583
    assert pytest.approx(0.5789, abs=1e-4) != units.DEFAULT_ASPECT


def test_aspect_agrees_with_the_card_size_it_sits_beside():
    """The two are independent constants that describe one card, so they can drift apart.
    70/120 is 0.5833; the measured 0.583 is that, rounded."""
    w, h = units.DEFAULT_CARD_MM
    assert pytest.approx(w / h, abs=5e-4) == units.DEFAULT_ASPECT


def test_the_default_strategy_is_one_that_exists():
    """R5 moved the default here while the fitter keeps using it, so the two can drift."""
    assert units.DEFAULT_STRATEGY in units.STRATEGIES


def test_the_default_strategy_is_the_one_the_foam_pad_calls_for():
    """The pad is mandatory equipment now (RFC-001), so "brightest" is the live path and
    "innermost" is kept only for the pre-pad scans the suite still pins."""
    assert units.DEFAULT_STRATEGY == "brightest"


@pytest.mark.parametrize("mm", [0.0, 1.0, 70.0, 120.0, 273.5])
def test_px_and_mm_round_trip(mm):
    assert units.px_to_mm(units.mm_to_px(mm)) == pytest.approx(mm)
