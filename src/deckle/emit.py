"""Generate a deck directory from a project dir"""

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
from .project import CARD_BACKS_DIR, DECK_IN_FILE, NAMES_DIR, PROJECT_FILE, Master, Project

SCHEMA_VERSION = "2.0"

MARKER = ".deckle-emit"
MARKER_TEXT = (
    "# This tarot deck directory was created using deckle.\n"
    "# This file is used to mark the directory for overwriting. Remove it to prevent deckle\n"
    "# from overrwriting local changes.\n"
)

DECK_KEY_ORDER = (
    "schema_version",
    "name",
    "version",
    "identifier",
    "related",
    "signifies",
    "default_language",
    "aspect_ratio",
    "icon",
    "creator",
    "artist",
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
    "published_date",
    "links",
    "tags",
)

PATH_FIELDS = ("icon",)
DATE_FIELDS = ("created_date", "updated_date", "published_date")

# The top-level tables of a deck.toml, which are all deck.in.toml may hold.
DECK_TABLES = (
    "deck",
    "card_backs",
    "cards",
    "suits",
    "ranks",
    "minor_arcana",
    "excluded_cards",
    "strings",
    "app",
)


class EmitError(RuntimeError):
    """Cannot (or should not) emit a tarot deck from the project."""


@dataclass
class EmitResult:
    target: Path
    cards: int
    backs: int
    images: int
    warnings: list[str] = field(default_factory=list)


# -- safety ---------------------------------------------------------------------------


def safe_relpath(value: str, what: str) -> str:
    if not value or value.startswith("/") or PurePosixPath(value).is_absolute():
        raise EmitError(f"{what} {value!r} must be relative to the deck root")
    if "\\" in value:
        raise EmitError(f"{what} {value!r} must use / as its separator")
    if ".." in PurePosixPath(value).parts:
        raise EmitError(f"{what} {value!r} must not contain a `..` segment")
    return value


def check_target(target: Path) -> None:
    """Refuse a target that is not missing, empty, or marked as `emit`'s own."""
    if ".." in target.parts:
        raise EmitError(f"emit target {target} must not contain a .. segment")
    if target.is_symlink():
        raise EmitError(f"emit target {target} is a symlink")
    if not target.exists():
        return
    if not target.is_dir():
        raise EmitError(f"emit target {target} exists and is not a directory")
    entries = list(target.iterdir())
    if not entries:
        return
    if not (target / MARKER).is_file():
        raise EmitError(
            f"refusing to overwrite {target}: it is not empty and does not have a {MARKER} marker file."
        )


def _wipe(target: Path) -> None:
    for entry in target.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


# -- deck.toml construction -----------------------------------------------------------


def _as_date_string(value: Any, what: str) -> str:
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
            raise EmitError(f"link rel {rel!r} must be a well-formed custom name")
        parsed = urlparse(url or "")
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise EmitError(f"link url {url!r} must be absolute with an http(s) scheme")
        clean = {"rel": rel, "url": url}
        if entry.get("title"):
            clean["title"] = entry["title"]
        out.append(clean)
    return out


def _merge(computed: dict[str, Any], authored: dict[str, Any]) -> dict[str, Any]:
    """Authored over computed. Tables merge key by key; anything else, arrays included, is
    replaced whole."""
    out = dict(computed)
    for key, value in authored.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _with_safe_image(entry: Any, what: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise EmitError(f"{what} must be a table")
    out = dict(entry)
    if "image" in out:
        out["image"] = safe_relpath(str(out["image"]), f"{what}.image")
    return out


def authored_tables(project: Project) -> tuple[dict[str, Any], str]:
    """The authored metadata as deck.toml-shaped tables, and the file it came from.

    deck.in.toml, where the project has one, is the only source. Otherwise [deck] and
    [card_meta] are read from the project file.
    """
    overlay = project.authored()
    if overlay is None:
        return {"deck": dict(project.deck), "cards": dict(project.card_meta)}, PROJECT_FILE

    held = [f"[{key}]" for key in ("deck", "card_meta") if project.doc.get(key)]
    if held:
        raise EmitError(
            f"{PROJECT_FILE} has {' and '.join(held)} and {DECK_IN_FILE} exists; only one of them "
            f"may hold authored metadata. Move it into {DECK_IN_FILE} ([card_meta] becomes "
            f"[cards]) and delete it from {PROJECT_FILE}"
        )
    for key, value in overlay.items():
        if key not in DECK_TABLES:
            raise EmitError(
                f"{DECK_IN_FILE}: {key!r} is not a deck.toml table; expected one of "
                f"{', '.join(DECK_TABLES)}"
            )
        if not isinstance(value, dict):
            raise EmitError(f"{DECK_IN_FILE}: {key} must be a table")
    deck = overlay.get("deck", {})
    computed = {"schema_version": SCHEMA_VERSION, "aspect_ratio": project.config.aspect}
    for key, value in computed.items():
        # Restating the computed value is harmless, as the file is shaped like the output
        if key in deck and deck[key] != value:
            raise EmitError(
                f"{DECK_IN_FILE}: [deck].{key} = {deck[key]!r}, but emit computes {value!r}; "
                "remove it"
            )
    return overlay, DECK_IN_FILE


def build_deck_table(src: dict[str, Any], *, aspect: float, source: Path) -> dict[str, Any]:
    src = dict(src)
    for required in ("name", "version"):
        if not src.get(required):
            raise EmitError(f"[deck].{required} is required; set it in {source}")

    out: dict[str, Any] = {}
    src["schema_version"] = SCHEMA_VERSION
    src.setdefault("aspect_ratio", aspect)
    for key in DATE_FIELDS:
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

    for key in sorted(src):
        out[key] = src[key]

    return out


def _card_backs(
    project: Project, authored: dict[str, Any], backs: list[str], warnings: list[str]
) -> dict[str, Any]:
    table: dict[str, Any] = {}
    origin = "card_back_default"
    if project.config.card_back_default:
        table["default"] = project.config.card_back_default
    if "default" in authored:
        origin = "[card_backs].default"
    table = _merge(table, authored)

    default = table.get("default")
    if default is not None and default not in backs:
        warnings.append(
            f"{origin} = {default!r} has no master under masters/{CARD_BACKS_DIR}/; "
            "omitting [card_backs].default"
        )
        del table["default"]

    if "designs" in table:
        designs = table.pop("designs")
        if not isinstance(designs, dict):
            raise EmitError("[card_backs].designs must be a table")
        kept = {}
        for key, entry in sorted(designs.items()):
            what = f"[card_backs.designs.{key}]"
            if key not in backs:
                warnings.append(
                    f"{what} has no master under masters/{CARD_BACKS_DIR}/; not writing it"
                )
                continue
            kept[key] = _with_safe_image(entry, what)
        if kept:
            table["designs"] = kept
    return table


def build_deck_doc(project: Project, masters: list[Master], warnings: list[str]) -> dict[str, Any]:
    """Assemble deck.toml.

    The authored tables contribute values and the masters contribute existence: anything
    authored about a card that has no master is dropped with a warning.
    """
    authored, source = authored_tables(project)
    meta_table = "cards" if source == DECK_IN_FILE else "card_meta"

    fronts = [m for m in masters if not m.is_back]
    backs = sorted({m.design for m in masters if m.is_back and m.design})
    # A variant is present under its own reference and as its card.
    present_ids = {str(m.card) for m in fronts} | {m.ref for m in fronts}

    doc: dict[str, Any] = {
        "deck": build_deck_table(
            authored.get("deck", {}), aspect=project.config.aspect, source=project.root / source
        )
    }

    card_backs = _card_backs(project, authored.get("card_backs", {}), backs, warnings)
    if card_backs:
        doc["card_backs"] = card_backs

    cards = {}
    for ref, meta in sorted(authored.get("cards", {}).items()):
        if not meta:
            continue
        if ref not in present_ids:
            warnings.append(
                f"[{meta_table}.{ref!r}] has no master; not writing a [cards] row for it"
            )
            continue
        cards[ref] = _with_safe_image(meta, f"[{meta_table}.{ref!r}]")
    if cards:
        doc["cards"] = cards

    roster = project.roster
    suits: dict[str, Any] = {}
    for suit, ranks in roster.suits.items():
        if roster.is_canonical_suit_sequence(suit):
            continue
        have = {m.card.rank for m in fronts if m.card and m.card.suit == suit}
        kept = [r for r in ranks if r in have]
        if kept:
            suits[suit] = {"ranks": kept}
    suits = _merge(suits, authored.get("suits", {}))
    if suits:
        doc["suits"] = dict(sorted(suits.items()))

    for key in ("ranks", "minor_arcana"):
        if authored.get(key):
            doc[key] = dict(authored[key])

    excluded: dict[str, Any] = {"cards": [c for c in roster.excluded if c not in present_ids]}
    excluded = _merge(excluded, authored.get("excluded_cards", {}))
    if isinstance(excluded.get("cards"), list):
        for card in excluded["cards"]:
            if card in present_ids:
                warnings.append(f"[excluded_cards] names {card!r}, which has a master; dropping it")
        excluded["cards"] = [c for c in excluded["cards"] if c not in present_ids]
        if not excluded["cards"]:
            del excluded["cards"]
    if excluded:
        doc["excluded_cards"] = excluded

    for key in ("strings", "app"):
        if authored.get(key):
            doc[key] = dict(authored[key])

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
            # The top-level card_backs/
            dest = target / CARD_BACKS_DIR / master.path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(master.path, dest)
            written += 1
        for height in heights:
            if height > mh:
                warnings.append(
                    f"h{height}: master {master.relpath} is only {mh}px tall; skipping "
                    "rather than upscaling"
                )
                continue

            # Every image in h<height>/ should be exactly <height>px tall
            width = max(1, round(height * mw / mh))

            # INTER_AREA, because Lanczos rings on a large reduction.
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
        raise EmitError(f"cannot emit: {project.degraded_reason}.")
    dest = (target or project.deck_dir()).expanduser()
    check_target(dest)

    masters = project.masters()
    if not masters:
        raise EmitError(f"nothing to emit: no masters in {project.masters_dir}")

    warnings: list[str] = []
    doc = build_deck_doc(project, masters, warnings)
    try:
        text = tomlout.dumps(doc)
    except tomlout.TomlError as exc:
        raise EmitError(f"cannot write deck.toml: {exc}") from None

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
