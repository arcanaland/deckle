from __future__ import annotations

from pathlib import Path

from .detect import Card
from .rectify import master_size_px


def report(cards: list[Card], scan: Path) -> None:
    print(f"{scan}: {len(cards)} card(s)")
    for c in cards:
        print(
            f"  r{c.window.row}c{c.window.col}  "
            f"{c.width_mm:8.4f} x {c.height_mm:8.4f} mm   "
            f"aspect {c.aspect:.4f}   skew {c.skew_deg:+.3f}deg"
        )
        w, h = master_size_px(c)
        print(
            f"        master {w} x {h}px   window opening "
            f"{c.window.opening_w_mm:.3f} x {c.window.opening_h_mm:.3f} mm"
        )
        for e, f in c.edges.items():
            print(
                f"        {e:6s} gap {f.median_gap_mm:+6.3f} mm   "
                f"step {f.median_step:6.1f}   yield {100 * f.yield_frac:3.0f}%   "
                f"residual {1000 * f.residual_sd_mm:5.1f} um (diagnostic only)"
            )
