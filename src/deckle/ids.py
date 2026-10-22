"""Canonical card IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

MAJOR = "major_arcana"
MINOR = "minor_arcana"

CANONICAL_SUITS = ("wands", "cups", "swords", "pentacles")
CANONICAL_RANKS = (
    "ace",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "page",
    "knight",
    "queen",
    "king",
)

RESERVED_NAMES = frozenset({MAJOR, MINOR, *CANONICAL_SUITS, *CANONICAL_RANKS})

_CUSTOM_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")
_TWO_DIGIT = re.compile(r"[0-9]{2}\Z")

CANONICAL_MAJORS = tuple(f"{n:02d}" for n in range(22))


class IdError(ValueError):
    """A string offered as a card ID or custom name that is not one."""


def is_custom_name(name: str) -> bool:
    return bool(_CUSTOM_NAME.fullmatch(name)) and name not in RESERVED_NAMES


def check_custom_name(name: str, what: str) -> str:
    if not _CUSTOM_NAME.fullmatch(name):
        raise IdError(
            f"{what} {name!r} is not a custom name: only lowercase letters, digits and "
            "underscores are allowed, and it may not start with a digit"
        )
    if name in RESERVED_NAMES:
        raise IdError(f"{what} {name!r} is a reserved canonical key")
    return name


def check_major_key(key: str) -> str:
    """Two digits or a custom name."""
    if _TWO_DIGIT.fullmatch(key):
        return key

    if key.isdigit():
        raise IdError(
            f"major arcana key {key!r} must be written with both digits, e.g. {int(key):02d}"
        )

    return check_custom_name(key, "major arcana key")


@dataclass(frozen=True, order=True)
class CardId:
    """`key` is set for a major arcanum; `suit` and `rank` for a minor one."""

    kind: str  # "major" or "minor"
    key: str | None = None
    suit: str | None = None
    rank: str | None = None

    def __str__(self) -> str:
        if self.kind == "major":
            return f"{MAJOR}.{self.key}"
        return f"{MINOR}.{self.suit}.{self.rank}"

    @property
    def base(self) -> str:
        """Filename stem of this card's images, before any variant."""
        return self.key if self.kind == "major" else self.rank  # type: ignore[return-value]

    @property
    def subpath(self) -> PurePosixPath:
        if self.kind == "major":
            return PurePosixPath(MAJOR)
        return PurePosixPath(MINOR, self.suit)  # type: ignore[arg-type]

    def relpath(self, variant: str | None = None, ext: str = "png") -> PurePosixPath:
        """Image path relative to an image root, with the variant infixed before ext"""
        stem = (
            self.base
            if variant is None
            else f"{self.base}.{check_custom_name(variant, 'variant key')}"
        )
        return self.subpath / f"{stem}.{ext}"

    @property
    def is_extended_major(self) -> bool:
        """A numbered major beyond the canonical twenty-two."""
        return (
            self.kind == "major"
            and bool(_TWO_DIGIT.fullmatch(self.key or ""))
            and self.key not in CANONICAL_MAJORS
        )


def parse_card_id(cid: str) -> CardId:
    parts = cid.split(".")

    if parts[0] == MAJOR:
        if len(parts) != 2:
            raise IdError(f"{cid!r}: a major arcana ID is {MAJOR}.<key>")
        return CardId("major", key=check_major_key(parts[1]))

    if parts[0] == MINOR:
        if len(parts) != 3:
            raise IdError(f"{cid!r}: a minor arcana ID is {MINOR}.<suit>.<rank>")
        _, suit, rank = parts
        if suit not in CANONICAL_SUITS:
            check_custom_name(suit, "suit key")
        if rank not in CANONICAL_RANKS:
            check_custom_name(rank, "rank key")
        return CardId("minor", suit=suit, rank=rank)

    raise IdError(f"{cid!r}: a canonical ID begins with {MAJOR}. or {MINOR}.")


def resolve_ref(ref: str) -> tuple[CardId, str | None]:
    """Split `<id>[:<variant>]` into its card and variant key."""
    cid, sep, variant = ref.partition(":")
    card = parse_card_id(cid)

    if sep and not variant:
        raise IdError(f"{ref!r}: empty variant key")

    return card, variant or None


def card_id_from_relpath(rel: PurePosixPath) -> tuple[CardId, str | None]:
    """Invert `CardId.relpath`. Raises `IdError` for a path that is not a card image."""
    parts = rel.parts
    stem = rel.name.rsplit(".", 1)[0] if "." in rel.name else rel.name
    base, _, variant = stem.partition(".")

    if variant:
        check_custom_name(variant, "variant key")

    if len(parts) == 2 and parts[0] == MAJOR:
        return CardId("major", key=check_major_key(base)), variant or None

    if len(parts) == 3 and parts[0] == MINOR:
        return parse_card_id(f"{MINOR}.{parts[1]}.{base}"), variant or None

    raise IdError(f"{rel}: not a card asset path")


def canonical_78() -> list[CardId]:
    cards = [CardId("major", key=k) for k in CANONICAL_MAJORS]

    for suit in CANONICAL_SUITS:
        cards += [CardId("minor", suit=suit, rank=r) for r in CANONICAL_RANKS]

    return cards
