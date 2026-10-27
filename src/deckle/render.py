from __future__ import annotations

from pathlib import Path

from .detect import Card
from .rectify import master_size_px
from .status import Status


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


def status_report(st: Status) -> None:
    no_default = set(st.no_default)
    print(f"{st.root}")
    if st.config_problem:
        print(f"  {st.config_problem}")
        print("    detect and rectify will refuse until this is fixed")
    elif st.edge_strategy:
        print(f"  edge strategy: {st.edge_strategy}")
    if st.degraded_reason:
        # every master is named by its card ID, so what is present survives a lost state file
        print(f"  {st.degraded_reason} — roster and provenance unavailable, so no total")
        print(f"  {len(st.present)} card(s) present:")
        for m in st.present:
            print(f"    {m.ref}")
    else:
        print(f"  {st.total - len(st.missing)} of {st.total} roster cards present")
        if st.staging is not None:
            print(f"  staging: {st.staging.summary()}  (deckle assign for detail)")
        if st.missing:
            print(f"  missing ({len(st.missing)}):")
            for ref in st.missing:
                note = "  (has variants, but no default master)" if ref in no_default else ""
                print(f"    {ref}{note}")
        if st.unexpected:
            print(f"  present but not in the roster ({len(st.unexpected)}) — the roster is wrong:")
            for ref in st.unexpected:
                print(f"    {ref}")
    if st.variants:
        print(f"  cards with variants ({len(st.variants)}):")
        for card, refs in st.variants.items():
            note = "  — missing its default master" if card in no_default else ""
            print(f"    {card}  variants: {', '.join(refs)}{note}")
    if st.backs:
        print(f"  card backs ({len(st.backs)}): {', '.join(st.backs)}")
    if st.strays:
        print(f"  unrecognised files under masters/ ({len(st.strays)}):")
        for p in st.strays:
            print(f"    {p.relative_to(st.root)}")
