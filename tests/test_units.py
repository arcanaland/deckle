from __future__ import annotations

import pytest

from deckle import units


@pytest.mark.parametrize("mm", [0.0, 1.0, 70.0, 120.0, 273.5])
def test_px_and_mm_round_trip(mm):
    assert units.px_to_mm(units.mm_to_px(mm)) == pytest.approx(mm)
