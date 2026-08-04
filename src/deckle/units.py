"""Pixel/millimetre conversion.

Every geometric number deckle reports is in millimetres; every number OpenCV hands
back is in pixels. Keeping the conversion in one place means the scan DPI appears
exactly once per run and the rest of the code never guesses it.
"""

from __future__ import annotations

MM_PER_INCH = 25.4

# RFC-001: scan at 600dpi. A 120mm card is 2835px tall, so the whole h750/h1200/h2400
# pyramid falls out of one master with headroom.
DEFAULT_DPI = 600.0


def mm_per_px(dpi: float = DEFAULT_DPI) -> float:
    return MM_PER_INCH / dpi


def px_to_mm(px: float, dpi: float = DEFAULT_DPI) -> float:
    return px * MM_PER_INCH / dpi


def mm_to_px(mm: float, dpi: float = DEFAULT_DPI) -> float:
    return mm * dpi / MM_PER_INCH


def mm_to_px_int(mm: float, dpi: float = DEFAULT_DPI) -> int:
    return int(round(mm_to_px(mm, dpi)))
