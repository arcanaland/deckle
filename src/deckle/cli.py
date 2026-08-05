"""deckle's command line.

Two families of command. `detect` and `rectify` share one detector and turn a scan into
geometry or into masters; `init`, `assign`, `status` and `emit` are the project directory
of [[ADR-003]] — where masters live, what is recorded about them, and how a deck directory
is generated from them.

`detect` and `rectify` exit non-zero the moment anything is not believable — RFC-001's
failure policy is that a silently mis-detected card is the one error that survives into the
emitted deck, so there is no best-effort mode and no --force. `status` is the exception and
exits 0 whether or not the deck is complete: incompleteness is the normal state of a
project, not an error.

Inside a project, `detect` and `rectify` take their card size, aspect, dpi and edge strategy
from `[project]`, so `deckle rectify <scan>` needs no flags where outside one it needs four.
An explicit flag still wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .assign import AssignError, assign
from .detect import CardSpec, DetectionError, detect
from .edges import DEFAULT_STRATEGY, STRATEGIES
from .emit import EmitError, emit
from .ids import IdError
from .project import (
    Project,
    ProjectError,
    Status,
    find_project,
    init,
    read_staging_index,
    status,
    write_staging_index,
)
from .rectify import master_size_px, rectify_all
from .units import DEFAULT_DPI


def _read(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"deckle: cannot read image {path}")
    return img


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("scan", type=Path, help="scan of the jig with cards in it")
    p.add_argument("--project", type=Path, default=None, help="project directory to use")
    p.add_argument("--dpi", type=float, default=None, help="scan resolution")
    p.add_argument("--card", default=None, help="card size WxH in mm")
    p.add_argument(
        "--aspect",
        type=float,
        default=None,
        help="expected width/height (the deck spec's 0.5789 is NOT the right value for "
        "every deck)",
    )
    p.add_argument("--expect", type=int, default=None, help="require exactly this many windows")
    p.add_argument(
        "--edge-strategy",
        choices=STRATEGIES,
        default=None,
        help="how to pick the card edge out of the candidate steps (default "
        f"{DEFAULT_STRATEGY!r}; use 'innermost' for scans taken WITHOUT the foam pad, "
        "where the frame's shadow ramp outsteps the card edge)",
    )


def _optional_project(args) -> Project | None:
    """The project a detect/rectify run sits in, if any. Absent is not an error: outside a
    project the flag-driven behaviour is exactly what it was."""
    if args.project is not None:
        return Project.discover(args.project)
    root = find_project()
    return Project.load(root) if root else None


def _settings(args, project: Project | None):
    """Flags first, then `[project]`, then the module defaults."""
    cfg = project.config if project and not project.degraded else None
    if args.card is not None:
        w, h = (float(v) for v in args.card.lower().split("x"))
    elif cfg is not None:
        w, h = cfg.card_width_mm, cfg.card_height_mm
    else:
        w, h = 70.0, 120.0
    aspect = args.aspect if args.aspect is not None else (cfg.aspect if cfg else CardSpec.aspect)
    dpi = args.dpi if args.dpi is not None else (cfg.dpi if cfg else DEFAULT_DPI)
    strategy = args.edge_strategy or (cfg.edge_strategy if cfg else DEFAULT_STRATEGY)
    return CardSpec(width_mm=w, height_mm=h, aspect=aspect), dpi, strategy


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


def _print_status(st: Status) -> None:
    print(f"{st.root}")
    if st.degraded_reason:
        # ADR-003's redundancy claim, in the one place it has to hold: identity survives
        # because every master is named by its canonical ID. Say what is unavailable
        # rather than showing a total that would be a guess.
        print(f"  {st.degraded_reason} — roster and provenance unavailable, so no total")
        print(f"  {len(st.present)} card(s) present:")
        for m in st.present:
            print(f"    {m.ref}")
    else:
        print(f"  {len(st.present)} of {st.total} roster cards present")
        if st.missing:
            print(f"  missing ({len(st.missing)}):")
            for ref in st.missing:
                print(f"    {ref}")
        if st.unexpected:
            print(f"  present but not in the roster ({len(st.unexpected)}) — the roster is wrong:")
            for ref in st.unexpected:
                print(f"    {ref}")
    if st.backs:
        print(f"  card backs ({len(st.backs)}): {', '.join(st.backs)}")
    if st.strays:
        print(f"  unrecognised files under masters/ ({len(st.strays)}):")
        for p in st.strays:
            print(f"    {p.relative_to(st.root)}")


def _cmd_detect_or_rectify(args) -> int:
    project = _optional_project(args)
    spec, dpi, strategy = _settings(args, project)
    img = _read(args.scan)
    try:
        cards = detect(img, spec=spec, dpi=dpi, expect=args.expect, strategy=strategy)
    except DetectionError as exc:
        print(f"deckle: {args.scan}: {exc}", file=sys.stderr)
        return 1

    if args.cmd == "detect":
        if args.json:
            json.dump(
                {"scan": str(args.scan), "dpi": dpi, "cards": [c.to_dict() for c in cards]},
                sys.stdout,
                indent=2,
            )
            sys.stdout.write("\n")
        else:
            _report(cards, args.scan)
        return 0

    out = args.out
    if out is None:
        if project is None:
            print("deckle: -o/--out is required outside a project", file=sys.stderr)
            return 1
        out = project.staging_dir

    written = rectify_all(img, cards, args.scan, out)

    # Record what the detector measured so `assign` can carry it into provenance rather
    # than the operator retyping it. It lives in staging/, which is scratch: nothing in it
    # outlives the assignment that consumes it.
    staged = (
        project is not None
        and not project.degraded
        and (out.resolve() == project.staging_dir.resolve())
    )
    if staged:
        index = read_staging_index(project)
        for path, card in zip(written, cards, strict=True):
            index[path.name] = {
                "source_scan": args.scan.name,
                "slot": f"r{card.window.row}c{card.window.col}",
                "width_mm": round(card.width_mm, 4),
                "height_mm": round(card.height_mm, 4),
                "aspect": round(card.aspect, 5),
                "skew_deg": round(card.skew_deg, 4),
                "edge_strategy": strategy,
                "dpi": dpi,
            }
        write_staging_index(project, index)

    for p in written:
        print(p)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deckle", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="report card geometry found in a scan")
    _common(d)
    d.add_argument("--json", action="store_true", help="emit machine-readable geometry")

    r = sub.add_parser("rectify", help="write one rectified master per card")
    _common(r)
    r.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the project's staging/)",
    )

    i = sub.add_parser("init", help="create a deckle project directory")
    i.add_argument("directory", type=Path, nargs="?", default=Path("."), help="project root")
    i.add_argument("--name", default=None, help="the deck's display name")
    i.add_argument("--deck-dir", default=None, help="where `deckle emit` writes the deck")

    s = sub.add_parser("status", help="what the project has, and what the roster still wants")
    s.add_argument("--project", type=Path, default=None, help="project directory to use")

    a = sub.add_parser("assign", help="move a staged master into masters/ under a canonical ID")
    a.add_argument("file", type=Path, help="the staged master to assign")
    a.add_argument("id", nargs="?", default=None, help="canonical ID, e.g. minor_arcana.cups.king")
    a.add_argument("--project", type=Path, default=None, help="project directory to use")
    a.add_argument("--variant", default=None, help="variant key, e.g. two_women (§4.7)")
    a.add_argument("--card-back", default=None, help="assign as a card back design instead")
    a.add_argument("--rotate-180", action="store_true", help="the card was placed upside down")

    e = sub.add_parser("emit", help="generate the deck directory from masters/")
    e.add_argument("--project", type=Path, default=None, help="project directory to use")
    e.add_argument("--deck-dir", type=Path, default=None, help="override the emit target")

    args = ap.parse_args(argv)

    try:
        if args.cmd in ("detect", "rectify"):
            return _cmd_detect_or_rectify(args)

        if args.cmd == "init":
            root = args.directory
            root.mkdir(parents=True, exist_ok=True)
            project = init(root, args.name or root.resolve().name, deck_dir=args.deck_dir)
            print(f"initialised {project.root}")
            print("  edit [deck] and [roster] in deckle.toml before emitting: the roster")
            print("  starts at the canonical 78, which is a starting point, not a fact")
            return 0

        project = Project.discover(args.project)

        if args.cmd == "status":
            _print_status(status(project))
            return 0

        if args.cmd == "assign":
            ref = args.id
            if ref and args.variant:
                ref = f"{ref}:{args.variant}"
            result = assign(
                project, args.file, ref, card_back=args.card_back, rotate_180=args.rotate_180
            )
            note = " (rotated 180)" if result.rotated else ""
            print(f"{result.ref} -> {result.dest.relative_to(project.root)}{note}")
            if result.replaced:
                print(f"  removed previous master {result.replaced.relative_to(project.root)}")
            return 0

        if args.cmd == "emit":
            result = emit(project, args.deck_dir)
            for w in result.warnings:
                print(f"deckle: warning: {w}", file=sys.stderr)
            print(
                f"{result.target}: {result.cards} card(s), {result.backs} back(s), "
                f"{result.images} image(s)"
            )
            # Say this every time. Implying a gate that does not exist is worse than
            # having no gate: nothing implements deck spec 2.0 yet, and §9.4 is
            # libarcana's to implement, not deckle's.
            print("  not validated: nothing implements deck spec 2.0 yet (§9.4 is libarcana's)")
            return 0
    except (ProjectError, AssignError, EmitError, IdError) as exc:
        print(f"deckle: {exc}", file=sys.stderr)
        return 1

    raise AssertionError(f"unhandled command {args.cmd!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
