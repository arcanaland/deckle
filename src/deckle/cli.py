"""deckle's command line.

Two commands so far, sharing one detector: `detect` reports geometry and `rectify` writes
masters. Both exit non-zero the moment anything is not believable — RFC-001's failure
policy is that a silently mis-detected card is the one error that survives into the
emitted deck, so there is no best-effort mode and no --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .detect import CardSpec, DetectionError, detect
from .rectify import master_size_px, rectify_all
from .units import DEFAULT_DPI


def _read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"deckle: cannot read image {path}")
    return img


def _spec(args) -> CardSpec:
    w, h = (float(v) for v in args.card.lower().split("x"))
    return CardSpec(width_mm=w, height_mm=h, aspect=args.aspect)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("scan", type=Path, help="scan of the jig with cards in it")
    p.add_argument("--dpi", type=float, default=DEFAULT_DPI, help="scan resolution")
    p.add_argument("--card", default="70x120", help="card size WxH in mm")
    p.add_argument(
        "--aspect",
        type=float,
        default=CardSpec.aspect,
        help="expected width/height (default %(default)s; the deck spec's 0.5789 is NOT "
        "the right value for every deck)",
    )
    p.add_argument("--expect", type=int, default=None, help="require exactly this many windows")


def _report(cards, scan: Path) -> None:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deckle", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="report card geometry found in a scan")
    _common(d)
    d.add_argument("--json", action="store_true", help="emit machine-readable geometry")

    r = sub.add_parser("rectify", help="write one rectified master per card")
    _common(r)
    r.add_argument("-o", "--out", type=Path, required=True, help="output directory")

    args = ap.parse_args(argv)
    img = _read(args.scan)

    try:
        cards = detect(img, spec=_spec(args), dpi=args.dpi, expect=args.expect)
    except DetectionError as exc:
        print(f"deckle: {args.scan}: {exc}", file=sys.stderr)
        return 1

    if args.cmd == "detect":
        if args.json:
            json.dump(
                {"scan": str(args.scan), "dpi": args.dpi, "cards": [c.to_dict() for c in cards]},
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
        else:
            _report(cards, args.scan)
        return 0

    written = rectify_all(img, cards, args.scan, args.out)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
