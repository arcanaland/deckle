"""Warp a detected card to a rectangular, losslessly-cropped master.

One `warpPerspective` performs deskew, crop and scale together. The manual GIMP workflow
this replaces rotates, then crops, then resizes — three resamples, each one softening the
artwork. Here the source pixels are read exactly once.

Masters are emitted at native scale and as lossless PNG. No downscale, no colour
conversion, no profile embedded: RFC-001 defers colour management, and a master that has
already been through a transform cannot un-apply it once that decision is made.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import Card
from .units import mm_to_px


def master_size_px(card: Card) -> tuple[int, int]:
    """Output size in px: the card's measured size at the scan's own resolution."""
    return (
        int(round(mm_to_px(card.width_mm, card.dpi))),
        int(round(mm_to_px(card.height_mm, card.dpi))),
    )


def rectify_card(bgr: np.ndarray, card: Card) -> np.ndarray:
    """Deskew, crop and emit one card as an upright image."""
    w, h = master_size_px(card)
    if h <= w:
        raise ValueError(f"card {card.window.index} is not portrait ({w}x{h}px) — refusing to emit")
    src = card.corners.astype(np.float32)  # TL, TR, BR, BL
    dst = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        bgr, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
    )


def master_name(scan: Path, card: Card) -> str:
    """`<scan-stem>_r<row>c<col>.png`.

    Slot position only. Canonical IDs are the `assign` stage's job and deliberately do not
    appear here — this task must not guess which card is which.
    """
    return f"{scan.stem}_r{card.window.row}c{card.window.col}.png"


def rectify_all(bgr: np.ndarray, cards: list[Card], scan: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for card in cards:
        path = out_dir / master_name(scan, card)
        img = rectify_card(bgr, card)
        if not cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise OSError(f"failed to write {path}")
        written.append(path)
    return written
