"""The corner-placement failure, on the scans it was found in.

`test_jig_placement.py` has the mechanism on a synthetic jig. This file is the measured
claim, and per CLAUDE.md §Truthfulness it is the half that counts: the synthetic frame is
a clean rectangle, and a clean rectangle only splits the outside when frame covers (0,0)
itself. A real scan is messier — on `20260901162636_001.png` pixel (0,0) is *not* frame,
it sits in a 22px sliver walled off by a frame run starting at x=22, and three frame
specks down the left edge finish the job. The phantom came back at 18577mm^2.

PipSpeak Tarot, 2026-09-01, jig J4. Five scans of the same sheet: the first with the jig
clear of the left platen edge, then four with it pushed into the corner, which is where
it belongs — it slides far less there, and sliding is what kills a run.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from deckle.jig import find_windows

pytestmark = pytest.mark.scans

DECK = "decks/pipspeak-tarot"
DPI = 600.0
CARD_MM = (60.016, 110.232)  # deckle's own survey, JIGS.md 2026-09-01

# Jig clear of the left edge: frame x[409, 4752] of 5100. This one always worked, and is
# here so a fix that broke it could not pass.
CLEAR = "20260901161018_001.png"
# Jig in the platen corner: frame reaches x=0. All four failed `detect --expect 4` with
# "found 5" before the fix.
CORNER = (
    "20260901161438_001.png",
    "20260901161750_001.png",
    "20260901162512_001.png",
    "20260901162636_001.png",
)

# Measured across all five scans after the fix. The jig is the thing being measured here,
# not the cards, so these hold for every scan of the session regardless of placement.
OPENING_W_MM = (63.76, 63.88)
OPENING_H_MM = (113.82, 113.96)


def _scan(scans: Path, name: str) -> np.ndarray:
    path = scans / DECK / "inbox" / name
    if not path.exists():
        path = scans / DECK / "inbox" / "accepted" / name
    assert path.exists(), (
        f"{name} is not under {scans / DECK}/inbox or its accepted/ — this is a failure "
        f"rather than a skip: these five scans are the only evidence that the jig may be "
        f"placed in the platen corner, and that placement is now the operating procedure"
    )
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"{path} did not decode"
    return img


@pytest.mark.parametrize("name", (CLEAR, *CORNER))
def test_every_placement_finds_exactly_four_windows(scans, name) -> None:
    windows = find_windows(_scan(scans, name), CARD_MM, DPI)
    assert len(windows) == 4, (
        f"{name}: found {len(windows)} windows. A fifth is the outside of the frame "
        f"read as a window, and it costs the whole scan — `detect --expect 4` refuses it."
    )


@pytest.mark.parametrize("name", (CLEAR, *CORNER))
def test_the_windows_are_the_jig_whichever_corner_it_sits_in(scans, name) -> None:
    """Count alone is not enough: the four found must be J4's actual openings.

    A phantom that happened to be filtered out by area would still leave the wrong four.
    """
    for w in find_windows(_scan(scans, name), CARD_MM, DPI):
        assert OPENING_W_MM[0] <= w.opening_w_mm <= OPENING_W_MM[1], (
            f"{name} window {w.index}: opening width {w.opening_w_mm:.3f}mm is outside "
            f"J4's measured {OPENING_W_MM}"
        )
        assert OPENING_H_MM[0] <= w.opening_h_mm <= OPENING_H_MM[1], (
            f"{name} window {w.index}: opening height {w.opening_h_mm:.3f}mm is outside "
            f"J4's measured {OPENING_H_MM}"
        )
