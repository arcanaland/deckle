"""Measure how sharp each card edge is, per window, across a set of scans.

This is the acceptance test for the foam pressure pad — RFC-001 §"Card lift". It measures
a *physical* property (is the card flat on the platen?) independently of whether `detect`
succeeds, which matters because the two failed together and only one of them is fixable in
software.

The number that decides it is the `bottom` row of the summary table. On 2026-08-03, over
seven scans and 28 cards with no pad:

    top      gapLuma   154   contrast   100   width    3.0px (127um)
    bottom   gapLuma   178   contrast    76   width   20.0px (847um)
    left     gapLuma   128   contrast   105   width    3.0px (127um)
    right    gapLuma   124   contrast   107   width    3.0px (127um)

Three edges at the scanner's PSF limit and one smeared 7x wider, only along the scan axis,
varying card-to-card. That is a penumbra from cards lifted off the glass. With the pad in,
`bottom` must land at ~3px like the others.

    uv run --with opencv-python-headless python tools/edge_sharpness.py <scan.jpg>...

Deliberately crude: it walks a fixed distance from each window wall and takes fixed-offset
samples for the gap and card plateaus, rather than finding the edge properly. That is the
point — it must not share code with the detector whose failure it is diagnosing. Scanlines
where the gap-to-card contrast is under 8 luma are skipped as unmeasurable, so a card with
dark artwork at its border contributes fewer samples rather than a wrong one.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from deckle.edges import INWARD, _line_at
from deckle.jig import EDGES, find_windows

N_SAMPLES = 140  # px walked inward from the wall; ~5.9mm at 600dpi
GAP_AT = slice(20, 30)  # plateau in the clearance gap, past the wall's dark line
CARD_AT = slice(100, 140)  # well inside the card, past any edge transition
MIN_CONTRAST = 8.0


def _profile(gray: np.ndarray, wall, edge: str, at: int) -> np.ndarray | None:
    """Luma walking inward from the wall, starting 8px outside it."""
    anchor = _line_at(wall, float(at), edge)
    dx, dy = INWARD[edge]
    step = dy if edge in ("top", "bottom") else dx
    start = int(round(anchor - 8 * step))
    stop = start + N_SAMPLES * step
    if edge in ("top", "bottom"):
        if min(start, stop) < 0 or max(start, stop) >= gray.shape[0]:
            return None
        return gray[start:stop:step, at].astype(np.float32)
    if min(start, stop) < 0 or max(start, stop) >= gray.shape[1]:
        return None
    return gray[at, start:stop:step].astype(np.float32)


def _transition_width(prof: np.ndarray) -> tuple[float, float, float] | None:
    """(gap luma, card luma, 10-90% rise width in px) for one scanline."""
    gap = float(np.median(prof[GAP_AT]))
    card = float(np.median(prof[CARD_AT]))
    if card - gap < MIN_CONTRAST:
        return None
    t10, t90 = gap + 0.1 * (card - gap), gap + 0.9 * (card - gap)
    seg = prof[15:90]
    above = np.flatnonzero(seg >= t90)
    if not above.size:
        return None
    below = np.flatnonzero(seg[: above[0]] <= t10)
    if not below.size:
        return None
    return gap, card, float(above[0] - below[-1])


def main(paths: list[str]) -> int:
    hdr = f"{'scan':>12} {'win':>3} {'edge':>6} {'gapLuma':>8} {'cardLuma':>8}"
    print(f"{hdr} {'contrast':>8} {'width_px':>9} {'width_um':>9}")
    per_edge: dict[str, list[tuple[float, float, float]]] = {}

    for path in paths:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"cannot read {path}", file=sys.stderr)
            return 1
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for w in find_windows(bgr):
            x, y, ww, hh = w.bbox
            for edge in EDGES:
                lo, hi = (x, x + ww) if edge in ("top", "bottom") else (y, y + hh)
                got = []
                for frac in np.linspace(0.2, 0.8, 13):
                    prof = _profile(gray, w.wall_lines[edge].line, edge, int(lo + frac * (hi - lo)))
                    if prof is None:
                        continue
                    m = _transition_width(prof)
                    if m is not None:
                        got.append(m)
                if not got:
                    continue
                a = np.array(got)
                gap, card, width = (float(np.median(a[:, i])) for i in range(3))
                print(
                    f"{path[-20:-8]:>12} {w.index:>3} {edge:>6} {gap:8.0f} {card:8.0f} "
                    f"{card - gap:8.0f} {width:9.1f} {width * 25400 / 600:9.0f}"
                )
                per_edge.setdefault(edge, []).append((gap, card - gap, width))

    print(f"\n=== per-edge medians over {sum(len(v) for v in per_edge.values()) // 4} cards ===")
    for edge in EDGES:
        if edge not in per_edge:
            continue
        a = np.array(per_edge[edge])
        w = float(np.median(a[:, 2]))
        print(
            f"{edge:>6}  gapLuma {np.median(a[:, 0]):5.0f}   contrast {np.median(a[:, 1]):5.0f}"
            f"   width {w:5.1f}px ({w * 25400 / 600:.0f}um)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
