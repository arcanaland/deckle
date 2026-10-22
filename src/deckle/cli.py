from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import build_id
from .detect import Card, CardSpec, DetectionError, detect
from .rectify import rectify_all
from .render import report
from .units import DEFAULT_DPI, DEFAULT_STRATEGY, EdgeStrategy


def _read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"deckle: cannot read image {path}")
    return img


def _card_size(text: str) -> tuple[float, float]:
    try:
        w, h = (float(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH in mm, got {text!r}") from None
    return w, h


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("scan", type=Path, help="scan of the jig with cards in it")
    p.add_argument(
        "--dpi", type=float, default=DEFAULT_DPI, help="scan resolution (default %(default)s)"
    )
    p.add_argument(
        "--card",
        type=_card_size,
        default="{:g}x{:g}".format(*CardSpec().card_mm),
        help="card size WxH in mm (default %(default)s)",
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=CardSpec.aspect,
        help="expected width/height (default %(default)s)",
    )
    p.add_argument(
        "--num-windows", type=int, default=None, help="require exactly this many windows"
    )
    p.add_argument(
        "--opposite-tol",
        type=float,
        default=CardSpec.opposite_tol_mm,
        help="how closely opposite edges of one card agree (default "
        "%(default)s).",
    )
    p.add_argument(
        "--edge-strategy",
        type=EdgeStrategy,
        choices=list(EdgeStrategy),
        default=DEFAULT_STRATEGY,
        help="how to find card edges  (default %(default)s;)",
    )


def _detect(args) -> tuple[np.ndarray, list[Card]] | None:
    img = _read(args.scan)
    w, h = args.card
    spec = CardSpec(width_mm=w, height_mm=h, aspect=args.aspect, opposite_tol_mm=args.opposite_tol)
    try:
        cards = detect(
            img,
            spec=spec,
            dpi=args.dpi,
            num_windows=args.num_windows,
            strategy=args.edge_strategy,
        )
    except DetectionError as exc:
        print(f"deckle: {args.scan}: {exc}", file=sys.stderr)
        return None
    return img, cards


def _cmd_detect(args) -> int:
    found = _detect(args)
    if found is None:
        return 1
    _, cards = found
    if args.json:
        json.dump(
            {"scan": str(args.scan), "dpi": args.dpi, "cards": [c.to_dict() for c in cards]},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        report(cards, args.scan)
    return 0


def _cmd_rectify(args) -> int:
    found = _detect(args)

    if found is None:
        return 1

    img, cards = found
    for p in rectify_all(img, cards, args.scan, args.out):
        print(p)

    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deckle", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"deckle {build_id()}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="report card geometry found in a scan")
    _common(d)
    d.add_argument("--json", action="store_true", help="emit machine-readable geometry")
    d.set_defaults(func=_cmd_detect)

    r = sub.add_parser("rectify", help="write one rectified master per card")
    _common(r)
    r.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    r.set_defaults(func=_cmd_rectify)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
