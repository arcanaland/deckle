"""The deckle project directory: `deckle.toml`, `masters/`, `staging/`, `names/`.

[[ADR-003]]. Masters are the archive and any deck directory `emit` writes is derived
output that may be deleted and regenerated. This module owns the archive side of that:
what a project is, how its state file loads and saves, and how `masters/` is walked.

Two properties are load-bearing and are tested rather than asserted:

**The state file is written atomically.** A truncated `deckle.toml` after a crash would
look exactly like lost work, which against a failure mode measured in months of absence is
the worst way to fail. Every write goes to a temp file in the same directory and is
`os.replace`d onto the target.

**`masters/` is a second, independent index.** Losing `deckle.toml` costs the roster and
the provenance; it does not cost *identity*, because every master is named by its canonical
ID. `load()` therefore has a degraded mode rather than an error path, and `deckle status`
still answers "these are the cards I have" from the filesystem alone.

The whole parsed document is kept as the source of truth and the typed accessors below are
views onto it, so a key deckle does not understand survives a load/save round-trip instead
of being silently dropped.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__, tomlout
from .edges import DEFAULT_STRATEGY
from .ids import (
    CANONICAL_MAJORS,
    CANONICAL_RANKS,
    CANONICAL_SUITS,
    CardId,
    IdError,
    card_id_from_relpath,
    check_custom_name,
    is_custom_name,
    parse_card_id,
)
from .units import DEFAULT_DPI

PROJECT_FILE = "deckle.toml"
MASTERS_DIR = "masters"
STAGING_DIR = "staging"
NAMES_DIR = "names"
CARD_BACKS_DIR = "card_backs"

#: Written beside the staged masters by `rectify` and consumed by `assign`. It is inside
#: `staging/`, which is scratch space, so it is not a sidecar in the sense ADR-003 rejects
#: — nothing in it outlives the assignment that reads it.
STAGING_INDEX = "index.toml"

DEFAULT_HEIGHTS = (750, 1200, 2400)
#: Measured across eight cards on 2026-08-04, not the spec's 0.5789 default. RFC-001.
DEFAULT_ASPECT = 0.583
DEFAULT_CARD_MM = (70.0, 120.0)


class ProjectError(RuntimeError):
    """A project that cannot be used as one."""


def utc_now() -> str:
    """An ISO-8601 timestamp in UTC, to the second. Provenance, not a deck.toml date."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Roster:
    """What the deck is *meant* to contain — its denominator.

    2.0 gives no fixed one. `[suits.cups].ranks` may add `princess`, majors run to `99`,
    cards may be custom-keyed and `[excluded_cards]` removes slots on purpose, so walking
    `masters/` yields the numerator alone. This has to be written down.
    """

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
        raw_suits = table.get("suits") or {s: list(CANONICAL_RANKS) for s in CANONICAL_SUITS}
        suits = {k: tuple(str(r) for r in v) for k, v in raw_suits.items()}
        return cls(majors=majors, suits=suits, excluded=tuple(table.get("excluded", ())))

    def to_doc(self) -> dict[str, Any]:
        return {
            "majors": list(self.majors),
            "excluded": list(self.excluded),
            "suits": {k: list(v) for k, v in self.suits.items()},
        }

    def cards(self) -> list[CardId]:
        """Every card the roster names, in spec order (§4.3.2), minus the exclusions."""
        excluded = set(self.excluded)
        out = [CardId("major", key=k) for k in self.majors]
        for suit, ranks in self.suits.items():
            out += [CardId("minor", suit=suit, rank=r) for r in ranks]
        return [c for c in out if str(c) not in excluded]

    def is_canonical_suit_sequence(self, suit: str) -> bool:
        """Whether emitting `[suits.<suit>].ranks` would say anything a reader does not
        already know. §4.4: a `ranks` list on a canonical suit *replaces* its canonical
        sequence, so writing one where nothing has changed is a claim, not a description."""
        return suit in CANONICAL_SUITS and self.suits.get(suit) == CANONICAL_RANKS


@dataclass(frozen=True)
class Config:
    """`[project]`: what makes `deckle rectify <scan>` inside a project take no flags."""

    card_width_mm: float = DEFAULT_CARD_MM[0]
    card_height_mm: float = DEFAULT_CARD_MM[1]
    aspect: float = DEFAULT_ASPECT
    dpi: float = DEFAULT_DPI
    edge_strategy: str = DEFAULT_STRATEGY
    emit_heights: tuple[int, ...] = DEFAULT_HEIGHTS
    deck_dir: str | None = None
    card_back_default: str | None = None

    @classmethod
    def from_doc(cls, table: dict[str, Any]) -> Config:
        heights = table.get("emit_heights", DEFAULT_HEIGHTS)
        return cls(
            card_width_mm=float(table.get("card_width_mm", DEFAULT_CARD_MM[0])),
            card_height_mm=float(table.get("card_height_mm", DEFAULT_CARD_MM[1])),
            aspect=float(table.get("aspect", DEFAULT_ASPECT)),
            dpi=float(table.get("dpi", DEFAULT_DPI)),
            edge_strategy=str(table.get("edge_strategy", DEFAULT_STRATEGY)),
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
            "edge_strategy": self.edge_strategy,
            "emit_heights": list(self.emit_heights),
        }
        if self.deck_dir is not None:
            doc["deck_dir"] = self.deck_dir
        if self.card_back_default is not None:
            doc["card_back_default"] = self.card_back_default
        return doc


@dataclass(frozen=True)
class Master:
    """One file under `masters/`, as found by walking it."""

    ref: str  # canonical ID, plus `:<variant>` where the file carries one
    card: CardId | None  # None for a card back
    variant: str | None
    design: str | None  # the design key, for a card back
    path: Path
    relpath: PurePosixPath

    @property
    def is_back(self) -> bool:
        return self.design is not None


def find_project(start: Path | None = None) -> Path | None:
    """Walk up from `start` looking for a project root, the way `git` finds its own.

    A `masters/` directory counts as well as `deckle.toml`, so that a project whose state
    file has been lost is still *findable*. Recognising it only by the file would make the
    degraded path unreachable from inside the project it is meant to rescue.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / PROJECT_FILE).is_file() or (candidate / MASTERS_DIR).is_dir():
            return candidate
    return None


class Project:
    """A loaded project. `doc` is the source of truth; everything else is a view."""

    def __init__(self, root: Path, doc: dict[str, Any], *, degraded_reason: str | None = None):
        self.root = root
        self.doc = doc
        self.degraded_reason = degraded_reason

    # -- loading and saving ------------------------------------------------------

    @classmethod
    def load(cls, root: Path) -> Project:
        """Load a project, degrading rather than failing if the state file is unusable.

        This is [[ADR-003]]'s redundancy claim as code: a missing or corrupt `deckle.toml`
        costs the roster and the provenance, never identity. `masters/` still names every
        card, so `status` still works and says what it cannot tell you.
        """
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
        if not isinstance(doc, dict):  # pragma: no cover - tomllib cannot produce this
            return cls(root, {}, degraded_reason=f"{PROJECT_FILE} is not a table")
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
        # A missing or unreadable state file is *not* an error here. It degrades, so that
        # `status` can still answer from `masters/` alone — [[ADR-003]]'s redundancy claim
        # is worth nothing if the command that relies on it refuses to start.
        return cls.load(root)

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None

    def save(self) -> None:
        """Write `deckle.toml` atomically: temp file in the same directory, then rename.

        Same directory matters — `os.replace` is only atomic within a filesystem, and the
        project may well live on the NFS share beside the scans.
        """
        if self.degraded:
            raise ProjectError(
                f"refusing to save over {PROJECT_FILE}: it was loaded in degraded mode "
                f"({self.degraded_reason}), so saving would destroy whatever is there"
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
        """`[deck]`: metadata rendered into the emitted `deck.toml`. Human authorship
        lives here, not in the deck directory, because `emit` overwrites that."""
        return self.doc.get("deck", {})

    @property
    def card_meta(self) -> dict[str, Any]:
        """`[card_meta."<id>"]`: display strings destined for the emitted `[cards]` table.

        Deliberately *not* `[cards]`, which this file uses for provenance. The two are
        different things pointed at the same IDs, and conflating them would put a scan
        filename into a deck a user reads.
        """
        return self.doc.get("card_meta", {})

    @property
    def cards(self) -> dict[str, Any]:
        """`[cards."<id>"]`: per-master provenance, one table per assigned card."""
        return self.doc.get("cards", {})

    # -- paths -------------------------------------------------------------------

    @property
    def masters_dir(self) -> Path:
        return self.root / MASTERS_DIR

    @property
    def staging_dir(self) -> Path:
        return self.root / STAGING_DIR

    @property
    def names_dir(self) -> Path:
        return self.root / NAMES_DIR

    def deck_dir(self) -> Path:
        """The emit target, resolved against the project root and `~` expanded."""
        raw = self.config.deck_dir
        if not raw:
            raise ProjectError(
                "no emit target: set deck_dir under [project] in deckle.toml, or pass " "--deck-dir"
            )
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (self.root / path)

    def master_path(self, card: CardId, variant: str | None = None) -> Path:
        return self.masters_dir / card.relpath(variant)

    def back_path(self, design: str) -> Path:
        return self.masters_dir / CARD_BACKS_DIR / f"{check_custom_name(design, 'design key')}.png"

    # -- the second index --------------------------------------------------------

    def masters(self) -> list[Master]:
        """Walk `masters/`, returning one entry per card asset found, sorted by ref.

        Files the mapping does not recognise are skipped rather than raising: this walk is
        the fallback path that has to work when nothing else does.
        """
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
        """Files under `masters/` that the canonical-ID mapping does not recognise."""
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


def default_doc(name: str, *, deck_dir: str | None = None) -> dict[str, Any]:
    """The document `deckle init` writes.

    The roster starts at the canonical seventy-eight. That is a starting point and not an
    assumption: a deck with extra cards, renamed suits or deliberate exclusions is expected
    to edit it, and until someone does, `status`'s denominator is simply wrong rather than
    authoritative.
    """
    return {
        "project": Config(deck_dir=deck_dir).to_doc(),
        "roster": Roster.canonical().to_doc(),
        "deck": {
            "name": name,
            "version": "0.1",
        },
        "card_meta": {},
        "cards": {},
    }


def init(root: Path, name: str, *, deck_dir: str | None = None) -> Project:
    """Create a project directory. Refuses to overwrite an existing `deckle.toml`."""
    if (root / PROJECT_FILE).exists():
        raise ProjectError(f"{root / PROJECT_FILE} already exists")
    for sub in (MASTERS_DIR, STAGING_DIR, NAMES_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    project = Project(root, default_doc(name, deck_dir=deck_dir))
    project.save()
    return project


# -- staging -------------------------------------------------------------------------


def read_staging_index(project: Project) -> dict[str, Any]:
    path = project.staging_dir / STAGING_INDEX
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def write_staging_index(project: Project, index: dict[str, Any]) -> None:
    project.staging_dir.mkdir(parents=True, exist_ok=True)
    path = project.staging_dir / STAGING_INDEX
    tmp = path.with_name(f".{STAGING_INDEX}.{os.getpid()}.tmp")
    try:
        tmp.write_text(tomlout.dumps(dict(sorted(index.items()))), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def provenance_for(
    filename: str,
    index: dict[str, Any],
    *,
    rotate_180: bool,
) -> dict[str, Any]:
    """Build a provenance row for a staged file, from whatever the index recorded.

    Provenance is only as honest as `assign` ([[ADR-003]]): recording the source scan and
    slot makes a misidentification *traceable*, not impossible. Where the index is absent —
    a file dropped into `staging/` by hand — the scan and slot are still recovered from
    `rectify`'s `<scan-stem>_r<row>c<col>.png` naming, and the measurements are simply
    omitted rather than guessed.
    """
    row = dict(index.get(filename, {}))
    stem = filename[: -len(".png")] if filename.endswith(".png") else filename
    scan, _, slot = stem.rpartition("_")
    prov: dict[str, Any] = {}
    prov["source_scan"] = row.get("source_scan") or (f"{scan}.jpg" if scan else filename)
    if row.get("slot") or slot:
        prov["slot"] = row.get("slot") or slot
    for key in ("width_mm", "height_mm", "aspect", "skew_deg", "edge_strategy", "dpi"):
        if key in row:
            prov[key] = row[key]
    prov["rotate_180"] = rotate_180
    prov["deckle_version"] = __version__
    prov["assigned_at"] = utc_now()
    return prov


# -- status --------------------------------------------------------------------------


@dataclass(frozen=True)
class Status:
    """What `deckle status` found. `total` is None when the roster is unavailable."""

    root: Path
    degraded_reason: str | None
    present: list[Master]
    total: int | None
    missing: list[str]
    unexpected: list[str]
    backs: list[str]
    strays: list[Path]


def status(project: Project) -> Status:
    masters = project.masters()
    backs = [m.ref for m in masters if m.is_back]
    fronts = [m for m in masters if not m.is_back]
    # A variant is another artwork for a card the deck already has, not another card, so
    # it must not count towards the numerator twice (§3.1.2).
    present_ids = {str(m.card) for m in fronts}

    if project.degraded:
        return Status(
            root=project.root,
            degraded_reason=project.degraded_reason,
            present=fronts,
            total=None,
            missing=[],
            unexpected=[],
            backs=backs,
            strays=project.strays(),
        )

    wanted = [str(c) for c in project.roster.cards()]
    wanted_set = set(wanted)
    return Status(
        root=project.root,
        degraded_reason=None,
        present=fronts,
        total=len(wanted),
        missing=[c for c in wanted if c not in present_ids],
        unexpected=sorted(present_ids - wanted_set),
        backs=backs,
        strays=project.strays(),
    )


# -- assign --------------------------------------------------------------------------


def resolve_ref(ref: str) -> tuple[CardId, str | None]:
    """Split a card reference (§3.1.2) into its card and variant key."""
    cid, sep, variant = ref.partition(":")
    card = parse_card_id(cid)
    if sep and not variant:
        raise IdError(f"{ref!r}: empty variant key")
    return card, variant or None
