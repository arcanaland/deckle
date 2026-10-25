from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import build_id
from .assign import AssignError, assign
from .debug import write_bundle
from .detect import Card, CardSpec, DetectionError, detect_report
from .emit import EmitError, emit
from .ids import IdError
from .labels import plan_staging
from .project import Config, Project, ProjectError, find_project, init
from .rectify import rectify_all
from .render import report, status_report
from .staging import record_rectified
from .status import status
from .units import EdgeStrategy


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


def _project_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--project",
        type=Path,
        default=None,
        help="project directory (default: the nearest one above the current directory)",
    )


_PROJECT_DEBUG = object()


def _common(p: argparse.ArgumentParser) -> None:
    # Defaults are None so an explicit flag can be told apart from the project's setting.
    defaults = Config()
    p.add_argument("scan", type=Path, help="scan of the jig with cards in it")
    _project_arg(p)
    p.add_argument(
        "--dpi", type=float, default=None, help=f"scan resolution (default {defaults.dpi:g})"
    )
    p.add_argument(
        "--card",
        type=_card_size,
        default=None,
        help=f"card size WxH in mm (default {defaults.card_width_mm:g}x{defaults.card_height_mm:g})",
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=None,
        help=f"expected width/height (default {defaults.aspect:g})",
    )
    p.add_argument(
        "--num-windows", type=int, default=None, help="require exactly this many windows"
    )
    p.add_argument(
        "--opposite-tol",
        type=float,
        default=None,
        help=f"how closely opposite edges of one card agree (default {defaults.opposite_tol_mm:g})",
    )
    p.add_argument(
        "--edge-strategy",
        type=EdgeStrategy,
        choices=list(EdgeStrategy),
        default=None,
        help=f"how to find card edges (default {defaults.edge_strategy})",
    )
    p.add_argument(
        "--debug",
        type=Path,
        nargs="?",
        const=_PROJECT_DEBUG,
        default=None,
        metavar="DIR",
        help="create a debug image and stats" "(default DIR: debug/)",
    )


def _optional_project(args) -> Project | None:
    """The project we're in."""
    if args.project is not None:
        return Project.discover(args.project)
    root = find_project()
    return Project.load(root) if root else None


def _settings(args, project: Project | None) -> tuple[CardSpec, float, EdgeStrategy]:
    """Flags first, then [project], then the defaults."""
    cfg = project.config if project and not project.degraded else Config()
    w, h = args.card or (cfg.card_width_mm, cfg.card_height_mm)
    spec = CardSpec(
        width_mm=w,
        height_mm=h,
        aspect=cfg.aspect if args.aspect is None else args.aspect,
        opposite_tol_mm=cfg.opposite_tol_mm if args.opposite_tol is None else args.opposite_tol,
    )
    dpi = cfg.dpi if args.dpi is None else args.dpi
    return spec, dpi, args.edge_strategy or cfg.edge_strategy


def _debug_root(args, project: Project | None) -> Path | None:
    """Where to dump debug bundles."""
    if args.debug is not None and args.debug is not _PROJECT_DEBUG:
        return args.debug

    return project.debug_dir if project is not None else None


def _detect(
    args, spec: CardSpec, dpi: float, strategy: EdgeStrategy, project: Project | None
) -> tuple[np.ndarray, list[Card]] | None:
    img = _read(args.scan)
    try:
        report = detect_report(
            img,
            spec=spec,
            dpi=dpi,
            num_windows=args.num_windows,
            strategy=strategy,
        )
    except DetectionError as exc:
        print(f"deckle: {args.scan}: {exc}", file=sys.stderr)
        return None

    failed = bool(report.problems)
    root = _debug_root(args, project)
    notes = []
    if root is not None and (failed or args.debug is not None):
        try:
            notes.append(f"deckle: debug overview: {write_bundle(report, img, args.scan, root)}")
        except Exception as exc:
            notes.append(f"deckle: could not write the debug bundle: {exc}")

    if failed:
        print(f"deckle: {args.scan}: {report.format()}", file=sys.stderr)
    for n in notes:
        print(n, file=sys.stderr)

    return None if failed else (img, report.cards)


def _debug_needs_dir(args, project: Project | None) -> bool:
    if args.debug is _PROJECT_DEBUG and project is None:
        print("deckle: --debug needs a directory outside a project", file=sys.stderr)
        return True
    return False


def _cmd_detect(args) -> int:
    project = _optional_project(args)
    if _debug_needs_dir(args, project):
        return 1
    spec, dpi, strategy = _settings(args, project)
    found = _detect(args, spec, dpi, strategy, project)
    if found is None:
        return 1
    _, cards = found
    if args.json:
        json.dump(
            {"scan": str(args.scan), "dpi": dpi, "cards": [c.to_dict() for c in cards]},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        report(cards, args.scan)
    return 0


def _cmd_rectify(args) -> int:
    project = _optional_project(args)
    if _debug_needs_dir(args, project):
        return 1
    out = args.out
    if out is None:
        if project is None:
            print("deckle: -o/--out is required outside a project", file=sys.stderr)
            return 1
        out = project.staging_dir

    spec, dpi, strategy = _settings(args, project)
    found = _detect(args, spec, dpi, strategy, project)

    if found is None:
        return 1

    img, cards = found
    written = rectify_all(img, cards, args.scan, out)
    if project is not None:
        record_rectified(project, args.scan, cards, written, strategy=strategy, dpi=dpi)

    for p in written:
        print(p)

    return 0


def _cmd_init(args) -> int:
    root = args.directory
    project = init(root, args.name or root.resolve().name, deck_dir=args.deck_dir)
    print(f"initialised {project.root}")
    print("  edit [deck] in deck.in.toml, and [roster] in deckle.toml")
    return 0


def _cmd_status(args) -> int:
    # incompleteness is the normal state of a project, so this exits 0 either way
    status_report(status(Project.discover(args.project)))
    return 0


def _cmd_assign(args) -> int:
    if args.file is None:
        stray = [f for f, on in _FILE_ONLY_FLAGS.items() if on(args)]
        if stray:
            args.usage(f"{', '.join(stray)} need a file; without one, assign reads labels")
        return _assign_from_labels(
            Project.discover(args.project), apply=args.apply, force=args.force
        )
    stray = [f for f, on in _LABEL_ONLY_FLAGS.items() if on(args)]
    if stray:
        args.usage(f"{', '.join(stray)} only apply without a file, when assign reads labels")

    project = Project.discover(args.project)
    ref = args.id
    if ref and args.variant:
        ref = f"{ref}:{args.variant}"
    result = assign(project, args.file, ref, card_back=args.card_back, rotate_180=args.rotate_180)
    note = " (rotated 180)" if result.rotated else ""
    print(f"{result.ref} -> {result.dest.relative_to(project.root)}{note}")
    if result.replaced:
        print(f"  removed previous master {result.replaced.relative_to(project.root)}")
    return 0


_FILE_ONLY_FLAGS = {
    "--variant": lambda a: a.variant is not None,
    "--card-back": lambda a: a.card_back is not None,
    "--rotate-180": lambda a: a.rotate_180,
}
_LABEL_ONLY_FLAGS = {"--apply": lambda a: a.apply, "--force": lambda a: a.force}


def _assign_from_labels(project: Project, *, apply: bool, force: bool) -> int:
    plan = plan_staging(project, force=force)

    for row in plan.rows:
        print(row.describe(project.root))
    for w in plan.warnings:
        print(f"warning: {w}")
    for p in plan.problems:
        print(f"problem: {p}")
    print(f"staging: {plan.summary()}")

    if plan.problems:
        if apply:
            print("deckle: nothing was assigned; fix the problems above first", file=sys.stderr)
        return 1
    if not apply:
        return 0

    for done, row in enumerate(plan.rows):
        try:
            assign(
                project,
                row.source,
                row.label.ref,
                card_back=row.label.card_back,
                rotate_180=row.label.rotate_180,
            )
        except (AssignError, ProjectError, IdError, OSError) as exc:
            print(f"deckle: {row.source.name}: {exc}", file=sys.stderr)
            print(
                f"deckle: {done} of {len(plan.rows)} assigned before this; run again to continue",
                file=sys.stderr,
            )
            return 1
    print(f"assigned {len(plan.rows)} file(s)")
    return 0


def _cmd_emit(args) -> int:
    result = emit(Project.discover(args.project), args.deck_dir)

    for w in result.warnings:
        print(f"deckle: warning: {w}", file=sys.stderr)

    print(
        f"{result.target}: {result.cards} card(s), {result.backs} back(s), "
        f"{result.images} image(s)"
    )

    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deckle", description="Scan tarot cards into a deck")
    ap.add_argument("--version", action="version", version=f"deckle {build_id()}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="report card geometry found in a scan")
    _common(d)
    d.add_argument("--json", action="store_true", help="emit machine-readable geometry")
    d.set_defaults(func=_cmd_detect)

    r = sub.add_parser("rectify", help="write one rectified master per card")
    _common(r)
    r.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output directory (default: the project's staging/)",
    )
    r.set_defaults(func=_cmd_rectify)

    i = sub.add_parser("init", help="create a project directory")
    i.add_argument("directory", type=Path, nargs="?", default=Path("."), help="project root")
    i.add_argument("--name", default=None, help="the deck's display name")
    i.add_argument("--deck-dir", default=None, help="where `deckle emit` writes the deck")
    i.set_defaults(func=_cmd_init)

    s = sub.add_parser("status", help="what the project has, and what the roster still wants")
    _project_arg(s)
    s.set_defaults(func=_cmd_status)

    a = sub.add_parser(
        "assign",
        help="move staged masters into masters/ under card IDs",
        description="With a file, assign it to the given card ID. Without one, read every "
        "label in staging/, print the plan, and with --apply carry it out.",
    )
    a.add_argument("file", type=Path, nargs="?", default=None, help="the staged master to assign")
    a.add_argument("id", nargs="?", default=None, help="card ID, e.g. minor_arcana.cups.king")
    _project_arg(a)
    a.add_argument("--variant", default=None, help="variant key, e.g. two_women")
    a.add_argument("--card-back", default=None, help="assign as a card back design instead")
    a.add_argument("--rotate-180", action="store_true", help="the card was placed upside down")
    a.add_argument("--apply", action="store_true", help="carry out the plan read from labels")
    a.add_argument(
        "--force", action="store_true", help="let the plan replace masters that already exist"
    )
    a.set_defaults(func=_cmd_assign, usage=a.error)

    e = sub.add_parser("emit", help="generate the deck directory from masters/")
    _project_arg(e)
    e.add_argument("--deck-dir", type=Path, default=None, help="override the emit target")
    e.set_defaults(func=_cmd_emit)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (ProjectError, AssignError, EmitError, IdError) as exc:
        print(f"deckle: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
