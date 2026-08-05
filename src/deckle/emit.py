"""Generate a spec-2.0 deck directory from a project's masters.

`emit` owns the deck directory outright and overwrites it, `deck.toml` and `names/`
included. That is [[ADR-003]] as amended: a directory holding *any* content that exists
nowhere else is not disposable, so all human authorship lives in the project and is
rendered from there. The property that buys is worth stating plainly — **`rm -rf` the deck
directory and re-run `emit` and you get byte-identical output** — and it is what makes a
deck directory something you can throw away when a resampler, a resolution or a spec
revision changes, instead of 78 cards back on the scanner bed.

**What this module is not.** It does not validate its output against §9.4. That rule table
is libarcana's, and deckle will link or shell out to it once libarcana implements 2.0;
carrying a second copy would drift from the first, invisibly, until a deck is wrong. What
this module owes instead is narrower and is a property of the generator: **it builds only
what it has files for**, so that the three §9.4 errors a partial deck could trip — a
`ranks` list naming a rank with no files, a `[cards]` row for a card with no files, a
`[card_backs].default` naming a design the deck lacks — do not arise. Those constraints are
tested as regressions on what `emit` writes, not as checks on a `deck.toml` it reads back.

Until libarcana lands, deckle's output has **no conformance gate at all**. That is accepted
debt, recorded in ADR-003, and it is not papered over here.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import cv2

from . import tomlout
from .ids import is_custom_name
from .project import CARD_BACKS_DIR, NAMES_DIR, Master, Project

SCHEMA_VERSION = "2.0"

#: Dropped in the emit target so that `emit` can tell a directory it wrote from one it did
#: not. Adam's `~/.local/share/tarot/decks/` holds real content, and an over-eager delete
#: there is the one failure in this module that destroys data. Its contents are fixed, so
#: it does not disturb the byte-identical property.
MARKER = ".deckle-emit"
MARKER_TEXT = (
    "# Written by `deckle emit`. This directory is generated output and deckle will\n"
    "# overwrite it wholesale on the next run. Delete this marker and deckle will\n"
    "# refuse to touch the directory at all.\n"
)

#: The order `[deck]` keys are written in. Fixed, so the file is byte-stable; roughly
#: §4.1's own order, with identity first and the licensing block together.
DECK_KEY_ORDER = (
    "schema_version",
    "name",
    "version",
    "identifier",
    "signifies",
    "default_language",
    "aspect_ratio",
    "icon",
    "author",
    "packager",
    "publisher",
    "description",
    "license",
    "license_files",
    "copyright",
    "attribution",
    "rights_status",
    "redistribution",
    "derivation",
    "created_date",
    "updated_date",
    "links",
    "tags",
)

PATH_FIELDS = ("icon",)


class EmitError(RuntimeError):
    """Something `emit` will not construct, or a target it will not write to."""


@dataclass
class EmitResult:
    target: Path
    cards: int
    backs: int
    images: int
    warnings: list[str] = field(default_factory=list)


# -- safety ---------------------------------------------------------------------------


def safe_relpath(value: str, what: str) -> str:
    """§2.3 / §10.1: a path field inside `deck.toml` stays inside the deck root."""
    if not value or value.startswith("/") or PurePosixPath(value).is_absolute():
        raise EmitError(f"{what} {value!r} must be relative to the deck root")
    if "\\" in value:
        raise EmitError(f"{what} {value!r} must use / as its separator")
    if ".." in PurePosixPath(value).parts:
        raise EmitError(f"{what} {value!r} must not contain a `..` segment")
    return value


def check_target(target: Path) -> None:
    """Refuse any target `emit` cannot recognise as its own.

    Three things are allowed: a path that does not exist, an empty directory, and a
    directory carrying the marker `emit` drops. Anything else — including a directory
    holding a hand-built deck — is refused rather than deleted.
    """
    if ".." in target.parts:
        raise EmitError(f"emit target {target} must not contain a `..` segment")
    if target.is_symlink():
        raise EmitError(f"emit target {target} is a symlink; deckle will not follow it")
    if not target.exists():
        return
    if not target.is_dir():
        raise EmitError(f"emit target {target} exists and is not a directory")
    entries = list(target.iterdir())
    if not entries:
        return
    if not (target / MARKER).is_file():
        raise EmitError(
            f"refusing to overwrite {target}: it is not empty and carries no {MARKER} "
            "marker, so deckle did not write it. Move it aside, or delete it yourself if "
            "you are sure."
        )


def _wipe(target: Path) -> None:
    for entry in target.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


# -- deck.toml construction -----------------------------------------------------------


def _as_date_string(value: Any, what: str) -> str:
    """§4.1: `created_date` and `updated_date` are strings, never TOML dates.

    A bare `1909-12-01` in the project file parses as a `datetime.date`, so it is coerced
    here rather than passed to the writer — which has no way to write a date at all.
    """
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise EmitError(f"[deck].{what} must be a string, got {type(value).__name__}")
    return value


def _clean_links(links: Any) -> list[dict[str, Any]]:
    if not isinstance(links, list):
        raise EmitError("[deck].links must be an array of tables")
    out = []
    for entry in links:
        if not isinstance(entry, dict):
            raise EmitError("each [deck].links entry must be a table")
        rel, url = entry.get("rel"), entry.get("url")
        if not isinstance(rel, str) or not is_custom_name(rel):
            raise EmitError(f"link rel {rel!r} must be a well-formed custom name (§3.2)")
        parsed = urlparse(url or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise EmitError(f"link url {url!r} must be absolute with an http(s) scheme")
        clean = {"rel": rel, "url": url}
        if entry.get("title"):
            clean["title"] = entry["title"]
        out.append(clean)
    return out


def build_deck_table(project: Project) -> dict[str, Any]:
    src = dict(project.deck)
    for required in ("name", "version"):
        if not src.get(required):
            raise EmitError(f"[deck].{required} is required; set it in {project.root}/deckle.toml")

    out: dict[str, Any] = {}
    src["schema_version"] = SCHEMA_VERSION
    src.setdefault("aspect_ratio", project.config.aspect)
    for key in ("created_date", "updated_date"):
        if key in src:
            src[key] = _as_date_string(src[key], key)
    if "links" in src:
        src["links"] = _clean_links(src["links"])
    for key in PATH_FIELDS:
        if key in src:
            src[key] = safe_relpath(str(src[key]), f"[deck].{key}")
    if "license_files" in src:
        src["license_files"] = [
            safe_relpath(str(p), "[deck].license_files entry") for p in src["license_files"]
        ]

    for key in DECK_KEY_ORDER:
        if key in src:
            out[key] = src.pop(key)
    # Anything left is a key deckle does not model. Carry it rather than drop it: §8
    # reserves unknown top-level names, but a key an author wrote is authorship.
    for key in sorted(src):
        out[key] = src[key]
    return out


def build_deck_doc(project: Project, masters: list[Master], warnings: list[str]) -> dict[str, Any]:
    """Assemble the whole `deck.toml`, projected down to what `masters/` actually holds."""
    fronts = [m for m in masters if not m.is_back]
    backs = sorted({m.design for m in masters if m.is_back and m.design})
    present_ids = {str(m.card) for m in fronts}

    doc: dict[str, Any] = {"deck": build_deck_table(project)}

    # §9.4 E: `[card_backs].default` must name a design the deck has. Omit rather than
    # name an absent one.
    wanted_default = project.config.card_back_default
    if wanted_default:
        if wanted_default in backs:
            doc["card_backs"] = {"default": wanted_default}
        else:
            warnings.append(
                f"card_back_default = {wanted_default!r} has no master under "
                f"masters/{CARD_BACKS_DIR}/; omitting [card_backs].default"
            )

    # §9.4 E: every card declared in `[cards]` is a card the deck has files for. Only
    # emitted cards get a row, so the question does not arise.
    cards = {
        ref: dict(meta)
        for ref, meta in sorted(project.card_meta.items())
        if ref in present_ids and meta
    }
    for ref, meta in project.card_meta.items():
        if ref not in present_ids and meta:
            warnings.append(f"[card_meta.{ref!r}] has no master; not writing a [cards] row for it")
    if cards:
        doc["cards"] = cards

    # §9.4 E: every rank named in a `ranks` list has files in that suit.
    #
    # A `ranks` list *replaces* a canonical suit's sequence (§4.4), so writing one for an
    # unmodified canonical suit would assert that the suit has only the ranks scanned so
    # far — which is a claim about the deck, not a description of the directory. A card
    # with no files is a resolution failure, not a violation, so silence is the truthful
    # answer there. A suit is described only where the roster says it differs.
    roster = project.roster
    suits: dict[str, Any] = {}
    for suit, ranks in roster.suits.items():
        if roster.is_canonical_suit_sequence(suit):
            continue
        have = {m.card.rank for m in fronts if m.card and m.card.suit == suit}
        kept = [r for r in ranks if r in have]
        if kept:
            suits[suit] = {"ranks": kept}
    if suits:
        doc["suits"] = dict(sorted(suits.items()))

    excluded = [c for c in roster.excluded if c not in present_ids]
    if excluded:
        doc["excluded_cards"] = {"cards": list(excluded)}

    return doc


# -- image emission -------------------------------------------------------------------


def _emit_images(project: Project, masters: list[Master], target: Path, warnings: list[str]) -> int:
    heights = project.config.emit_heights
    written = 0
    for master in masters:
        img = cv2.imread(str(master.path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise EmitError(f"cannot read master {master.path}")
        mh, mw = img.shape[:2]
        if master.is_back:
            # §5.5: the top-level `card_backs/` holds backs of no declared kind or size.
            dest = target / CARD_BACKS_DIR / master.path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(master.path, dest)
            written += 1
        for height in heights:
            if height > mh:
                # §5.7.3: downscaling a raster is well-behaved and upscaling is not. A
                # target above the master's own height is skipped, never invented.
                warnings.append(
                    f"h{height}: master {master.relpath} is only {mh}px tall; skipping "
                    "rather than upscaling"
                )
                continue
            # §5.3: every image in `h<height>/` is exactly `<height>` px tall. Width
            # follows each master's own aspect, so widths differ by a pixel or two between
            # cards — nothing requires an `h` root to be uniform in width.
            width = max(1, round(height * mw / mh))
            # INTER_AREA for the downscale. Lanczos is right for the warp at native scale
            # and wrong for a 3x reduction, where it rings.
            small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            dest = target / f"h{height}" / master.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), small, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise EmitError(f"failed to write {dest}")
            written += 1
    return written


def _emit_names(project: Project, target: Path) -> None:
    src = project.names_dir
    if not src.is_dir():
        return
    files = sorted(p for p in src.glob("*.toml") if p.is_file())
    if not files:
        return
    (target / NAMES_DIR).mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copyfile(path, target / NAMES_DIR / path.name)


# -- the command ----------------------------------------------------------------------


def emit(project: Project, target: Path | None = None) -> EmitResult:
    if project.degraded:
        raise EmitError(
            f"cannot emit: {project.degraded_reason}. Everything the deck's metadata comes "
            "from lives in the state file."
        )
    dest = (target or project.deck_dir()).expanduser()
    check_target(dest)

    masters = project.masters()
    if not masters:
        raise EmitError(
            f"nothing to emit: no masters under {project.masters_dir}. §9.1 — a deck with "
            "no assets of any kind is not a deck."
        )

    warnings: list[str] = []
    doc = build_deck_doc(project, masters, warnings)
    text = tomlout.dumps(doc)

    dest.mkdir(parents=True, exist_ok=True)
    _wipe(dest)
    (dest / MARKER).write_text(MARKER_TEXT, encoding="utf-8")
    images = _emit_images(project, masters, dest, warnings)
    _emit_names(project, dest)
    (dest / "deck.toml").write_text(text, encoding="utf-8")

    fronts = {str(m.card) for m in masters if not m.is_back}
    backs = {m.design for m in masters if m.is_back}
    return EmitResult(
        target=dest, cards=len(fronts), backs=len(backs), images=images, warnings=warnings
    )
