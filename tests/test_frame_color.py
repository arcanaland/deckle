"""`FrameColor` selects the frame and nothing a scan puts beside it."""

from __future__ import annotations

import numpy as np

from deckle.jig import DEFAULT_FRAME_COLOR, FrameColor

GREEN = (60, 160, 60)
RED_LOW_HUE = (40, 40, 200)  # H 0
RED_HIGH_HUE = (60, 20, 200)  # H 173
PLATEN = (250, 250, 250)
VOID = (5, 5, 5)


def selected(color: FrameColor) -> list[bool]:
    img = np.array([[GREEN, RED_LOW_HUE, RED_HIGH_HUE, PLATEN, VOID]], np.uint8)
    return [bool(v) for v in color.mask(img)[0]]


def test_default_selects_only_the_green() -> None:
    assert selected(DEFAULT_FRAME_COLOR) == [True, False, False, False, False]


def test_a_hue_range_that_wraps_selects_both_ends_of_red() -> None:
    red = FrameColor(lo=(170, 60, 40), hi=(10, 255, 255))
    assert selected(red) == [False, True, True, False, False]
