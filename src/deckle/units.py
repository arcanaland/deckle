"""What the rig is, in numbers — and pixel/millimetre conversion.

Every geometric number deckle reports is in millimetres; every number OpenCV hands
back is in pixels. Keeping the conversion in one place means the scan DPI appears
exactly once per run and the rest of the code never guesses it.

The measured constants live here for the same reason, and it is worth stating because
they did not start out that way. `detect`, `project` and `cli` each carried their own
copy, and they disagreed: the detector defaulted to an aspect of 0.5843 with no
provenance in any doc, while the project file used the measured 0.583 — so `deckle
detect` outside a project was gated against a different card than the same command
inside one. That is the whole argument for one home. A number obtained by holding
calipers against a physical card should appear once.

Every value here is a fact about the scanner, the jig and the deck in front of them,
not a tuning parameter. That is the test for whether something belongs.
"""

from __future__ import annotations

MM_PER_INCH = 25.4

# RFC-001: scan at 600dpi. A 120mm card is 2835px tall, so the whole h750/h1200/h2400
# pyramid falls out of one master with headroom.
DEFAULT_DPI = 600.0

#: Calipers, 2026-08-04, across eight cards. RFC-001 and CLAUDE.md both record it.
DEFAULT_CARD_MM = (70.0, 120.0)

#: Measured, *not* the deck spec's 0.5789 default — the sample deck is genuinely a
#: different shape, and the detector must never quietly substitute the spec figure.
#: Configurable per project; this is only the fallback when nothing says otherwise.
DEFAULT_ASPECT = 0.583

#: How `edges` decides which step along a scanline is the card boundary. Two exist
#: because two optical situations do: with the foam pad the true edge is a hard
#: 150-248 luma step right at the boundary ("brightest"), while pre-pad scans have a
#: shadow ramp in the clearance gap and need the innermost step instead. Which one a
#: scan needs is a fact about how it was taken, so it is pinned here with the rest of
#: the rig rather than inside the fitter.
STRATEGIES = ("brightest", "innermost")
DEFAULT_STRATEGY = "brightest"


def mm_per_px(dpi: float = DEFAULT_DPI) -> float:
    return MM_PER_INCH / dpi


def px_to_mm(px: float, dpi: float = DEFAULT_DPI) -> float:
    return px * MM_PER_INCH / dpi


def mm_to_px(mm: float, dpi: float = DEFAULT_DPI) -> float:
    return mm * dpi / MM_PER_INCH


def mm_to_px_int(mm: float, dpi: float = DEFAULT_DPI) -> int:
    return int(round(mm_to_px(mm, dpi)))
