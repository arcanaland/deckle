"""The project directory"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from . import tomlout
from .ids import (
    CANONICAL_MAJORS,
    CANONICAL_RANKS,
    CANONICAL_SUITS,
    CardId,
    IdError,
    card_id_from_relpath,
    check_custom_name,
    is_custom_name,
)
from .units import (
    DEFAULT_ASPECT,
    DEFAULT_CARD_MM,
    DEFAULT_DPI,
    DEFAULT_OPPOSITE_TOL_MM,
    DEFAULT_STRATEGY,
    EdgeStrategy,
)

PROJECT_FILE = "deckle.toml"
DECK_IN_FILE = "deck.in.toml"
MASTERS_DIR = "masters"
STAGING_DIR = "staging"
DEBUG_DIR = "debug"
NAMES_DIR = "names"
CARD_BACKS_DIR = "card_backs"

DEFAULT_HEIGHTS = (750, 1200, 2400)


class ProjectError(RuntimeError):
    """A project that cannot be used as one."""


@dataclass(frozen=True)
class Roster:
    """The cards a deck is meant to contain."""

    majors: tuple[str, ...]
    suits: dict[str, tuple[str, ...]]
    excluded: tuple[str, ...] = ()

    @classmethod
    def canonical(cls) -> Roster:
        return cls(
            majors=CANONICAL_MAJORS,
            suits={s: CANONICAL_RANKS for s in CANONICAL_SUITS},
        )

    @classmethod
    def from_doc(cls, table: dict[str, Any]) -> Roster:
        if not table:
            return cls.canonical()
        majors = tuple(str(m) for m in table.get("majors", CANONICAL_MAJORS))

        # Absent means the canonical four while an empty table means no suits.
        raw_suits = table.get("suits", {s: list(CANONICAL_RANKS) for s in CANONICAL_SUITS})
        suits = {k: tuple(str(r) for r in v) for k, v in raw_suits.items()}
        return cls(majors=majors, suits=suits, excluded=tuple(table.get("excluded", ())))

    def to_doc(self) -> dict[str, Any]:
        return {
            "majors": list(self.majors),
            "excluded": list(self.excluded),
            "suits": {k: list(v) for k, v in self.suits.items()},
        }

    def cards(self) -> list[CardId]:
        """Every card minus the exclusions."""
        excluded = set(self.excluded)
        out = [CardId("major", key=k) for k in self.majors]

        for suit, ranks in self.suits.items():
            out += [CardId("minor", suit=suit, rank=r) for r in ranks]

        return [c for c in out if str(c) not in excluded]

    def is_canonical_suit_sequence(self, suit: str) -> bool:
        return suit in CANONICAL_SUITS and self.suits.get(suit) == CANONICAL_RANKS


@dataclass(frozen=True)
class Config:
    """The [project] section"""

    card_width_mm: float = DEFAULT_CARD_MM[0]
    card_height_mm: float = DEFAULT_CARD_MM[1]
    aspect: float = DEFAULT_ASPECT
    dpi: float = DEFAULT_DPI
    edge_strategy: EdgeStrategy = DEFAULT_STRATEGY
    opposite_tol_mm: float = DEFAULT_OPPOSITE_TOL_MM
    emit_heights: tuple[int, ...] = DEFAULT_HEIGHTS
    deck_dir: str | None = None
    card_back_default: str | None = None

    @classmethod
    def from_doc(cls, table: dict[str, Any]) -> Config:
        raw = table.get("edge_strategy", DEFAULT_STRATEGY)
        try:
            strategy = EdgeStrategy(raw)
        except ValueError:
            raise ProjectError(
                f"{PROJECT_FILE}: edge_strategy = {raw!r} is not a known strategy; "
                f"expected one of {', '.join(repr(s.value) for s in EdgeStrategy)}"
            ) from None

        heights = table.get("emit_heights", DEFAULT_HEIGHTS)

        return cls(
            card_width_mm=float(table.get("card_width_mm", DEFAULT_CARD_MM[0])),
            card_height_mm=float(table.get("card_height_mm", DEFAULT_CARD_MM[1])),
            aspect=float(table.get("aspect", DEFAULT_ASPECT)),
            dpi=float(table.get("dpi", DEFAULT_DPI)),
            edge_strategy=strategy,
            opposite_tol_mm=float(table.get("opposite_tol_mm", DEFAULT_OPPOSITE_TOL_MM)),
            emit_heights=tuple(int(h) for h in heights),
            deck_dir=table.get("deck_dir"),
            card_back_default=table.get("card_back_default"),
        )

    def to_doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "card_width_mm": self.card_width_mm,
            "card_height_mm": self.card_height_mm,
            "aspect": self.aspect,
            "dpi": self.dpi,
            "edge_strategy": str(self.edge_strategy),
            "opposite_tol_mm": self.opposite_tol_mm,
            "emit_heights": list(self.emit_heights),
        }
        if self.deck_dir is not None:
            doc["deck_dir"] = self.deck_dir
        if self.card_back_default is not None:
            doc["card_back_default"] = self.card_back_default
        return doc


@dataclass(frozen=True)
class Master:
    """A single file in masters/"""

    ref: str  # card ID (plus :<variant>)
    card: CardId | None  # None for a card back
    variant: str | None
    design: str | None  # the design key, for a card back
    path: Path
    relpath: PurePosixPath

    @property
    def is_back(self) -> bool:
        return self.design is not None


def find_project(start: Path | None = None) -> Path | None:
    """Walk up from start to a directory holding deckle.toml or masters/"""
    here = (start or Path.cwd()).resolve()

    for candidate in (here, *here.parents):
        if (candidate / PROJECT_FILE).is_file() or (candidate / MASTERS_DIR).is_dir():
            return candidate

    return None


class Project:
    def __init__(self, root: Path, doc: dict[str, Any], *, degraded_reason: str | None = None):
        self.root = root
        self.doc = doc
        self.degraded_reason = degraded_reason

    # -- loading and saving ------------------------------------------------------

    @classmethod
    def load(cls, root: Path) -> Project:
        path = root / PROJECT_FILE
        try:
            with path.open("rb") as fh:
                doc = tomllib.load(fh)
        except FileNotFoundError:
            return cls(root, {}, degraded_reason=f"{PROJECT_FILE} is missing")
        except tomllib.TOMLDecodeError as exc:
            return cls(root, {}, degraded_reason=f"{PROJECT_FILE} is not valid TOML: {exc}")
        except OSError as exc:
            return cls(root, {}, degraded_reason=f"{PROJECT_FILE} cannot be read: {exc}")
        return cls(root, doc)

    @classmethod
    def discover(cls, explicit: Path | None = None, start: Path | None = None) -> Project:
        root = explicit.resolve() if explicit else find_project(start)
        if root is None:
            raise ProjectError(
                f"no {PROJECT_FILE} here or in any parent directory; run `deckle init` or "
                "pass --project"
            )
        if not root.is_dir():
            raise ProjectError(f"{root} is not a directory")
        return cls.load(root)

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None

    def save(self) -> None:
        if self.degraded:
            raise ProjectError(
                f"refusing to save over {PROJECT_FILE}: it was loaded in degraded mode "
                f"({self.degraded_reason})"
            )
        target = self.root / PROJECT_FILE
        text = tomlout.dumps(self.doc)
        tmp = target.with_name(f".{PROJECT_FILE}.{os.getpid()}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    # -- views onto the document -------------------------------------------------

    @property
    def config(self) -> Config:
        return Config.from_doc(self.doc.get("project", {}))

    @property
    def roster(self) -> Roster:
        return Roster.from_doc(self.doc.get("roster", {}))

    @property
    def deck(self) -> dict[str, Any]:
        """the [deck] section"""
        return self.doc.get("deck", {})

    @property
    def card_meta(self) -> dict[str, Any]:
        return self.doc.get("card_meta", {})

    @property
    def cards(self) -> dict[str, Any]:
        """[cards."<id>"]"""
        return self.doc.get("cards", {})

    def authored(self) -> dict[str, Any] | None:
        """deck.in.toml, or None where the project has none.

        It is written by hand and only ever read, so unlike deckle.toml it keeps its comments.
        """
        path = self.root / DECK_IN_FILE
        try:
            with path.open("rb") as fh:
                return tomllib.load(fh)
        except FileNotFoundError:
            return None
        except tomllib.TOMLDecodeError as exc:
            raise ProjectError(f"{DECK_IN_FILE} is not valid TOML: {exc}") from None
        except OSError as exc:
            raise ProjectError(f"{DECK_IN_FILE} cannot be read: {exc}") from None

    # -- paths -------------------------------------------------------------------

    @property
    def masters_dir(self) -> Path:
        return self.root / MASTERS_DIR

    @property
    def staging_dir(self) -> Path:
        return self.root / STAGING_DIR

    @property
    def debug_dir(self) -> Path:
        return self.root / DEBUG_DIR

    @property
    def names_dir(self) -> Path:
        return self.root / NAMES_DIR

    def deck_dir(self) -> Path:
        """The emit target."""
        raw = self.config.deck_dir
        if not raw:
            raise ProjectError(
                "no emit target: set deck_dir under [project] in deckle.toml, or pass --deck-dir"
            )
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (self.root / path)

    def master_path(self, card: CardId, variant: str | None = None) -> Path:
        return self.masters_dir / card.relpath(variant)

    def back_path(self, design: str) -> Path:
        return self.masters_dir / CARD_BACKS_DIR / f"{check_custom_name(design, 'design key')}.png"

    # -- masters/ ----------------------------------------------------------------

    def masters(self) -> list[Master]:
        """Every card image under masters/"""
        out: list[Master] = []
        base = self.masters_dir

        if not base.is_dir():
            return out

        for path in sorted(base.rglob("*.png")):
            rel = PurePosixPath(path.relative_to(base).as_posix())
            if rel.parts[:1] == (CARD_BACKS_DIR,) and len(rel.parts) == 2:
                design = rel.name[: -len(".png")]
                if not is_custom_name(design):
                    continue
                out.append(Master(design, None, None, design, path, rel))
                continue
            try:
                card, variant = card_id_from_relpath(rel)
            except IdError:
                continue
            ref = str(card) if variant is None else f"{card}:{variant}"
            out.append(Master(ref, card, variant, None, path, rel))
        return sorted(out, key=lambda m: (m.is_back, m.ref))

    def strays(self) -> list[Path]:
        """Files under masters/ that are not card images."""
        base = self.masters_dir
        if not base.is_dir():
            return []
        known = {m.path for m in self.masters()}
        return sorted(p for p in base.rglob("*") if p.is_file() and p not in known)

    # -- mutation ----------------------------------------------------------------

    def record_card(self, ref: str, provenance: dict[str, Any]) -> None:
        cards = self.doc.setdefault("cards", {})
        cards[ref] = provenance
        self.doc["cards"] = dict(sorted(cards.items()))

    def forget_card(self, ref: str) -> None:
        self.doc.get("cards", {}).pop(ref, None)


def default_doc(*, deck_dir: str | None = None) -> dict[str, Any]:
    return {
        "project": Config(deck_dir=deck_dir).to_doc(),
        "roster": Roster.canonical().to_doc(),
        "cards": {},
    }


def default_deck_in(name: str) -> str:
    return (
        "# What the deck is, shaped like the deck.toml that `deckle emit` writes. deckle reads\n"
        "# this file and never rewrites it. Entries for cards without a master are dropped.\n"
        "\n"
        "[deck]\n"
        f"name = {tomlout.format_string(name)}\n"
        'version = "0.1"\n'
    )


def init(root: Path, name: str, *, deck_dir: str | None = None) -> Project:
    """Create a project directory."""
    if (root / PROJECT_FILE).exists():
        raise ProjectError(f"{root / PROJECT_FILE} already exists")

    for sub in (MASTERS_DIR, STAGING_DIR, NAMES_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)

    project = Project(root, default_doc(deck_dir=deck_dir))
    project.save()

    deck_in = root / DECK_IN_FILE
    if not deck_in.exists():
        deck_in.write_text(default_deck_in(name), encoding="utf-8")

    return project
