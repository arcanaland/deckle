from __future__ import annotations

from enum import StrEnum

MM_PER_INCH = 25.4
DEFAULT_DPI = 600.0
DEFAULT_CARD_MM = (70.0, 120.0)
DEFAULT_ASPECT = 0.583
DEFAULT_OPPOSITE_TOL_MM = 0.150


class EdgeStrategy(StrEnum):
    BRIGHTEST = "brightest"
    INNERMOST = "innermost"
    OUTERMOST = "outermost"


DEFAULT_STRATEGY = EdgeStrategy.BRIGHTEST


def mm_per_px(dpi: float = DEFAULT_DPI) -> float:
    return MM_PER_INCH / dpi


def px_to_mm(px: float, dpi: float = DEFAULT_DPI) -> float:
    return px * MM_PER_INCH / dpi


def mm_to_px(mm: float, dpi: float = DEFAULT_DPI) -> float:
    return mm * dpi / MM_PER_INCH


def mm_to_px_int(mm: float, dpi: float = DEFAULT_DPI) -> int:
    return int(round(mm_to_px(mm, dpi)))
