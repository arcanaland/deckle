"""Canonical card IDs, and the mapping from an ID to a master's path on disk.

`DECK.md` 2.0 §3.5 gives the grammar in ABNF; this module is that grammar and §3.2's
reserved-name rules, and nothing else. It is deliberately the only place in deckle that
knows how a card is spelled, because two places would drift.

The mapping to `masters/<...>.png` is injective, which is what lets `masters/` serve as
the second, independent index [[ADR-003]] relies on: §3.2 forbids a custom name from
shadowing a canonical suit or rank, §3.5 forbids a `.` inside a variant key, and every key
is lowercase so §2.3's case-insensitive stem comparison never fires.

This is *not* a `deck.toml` validator. Validating a deck against §9.4 is libarcana's job;
deckle never grows a second implementation of that rule table. What lives here is the
narrower question `assign` has to answer before it moves a file: is this string a card ID,
and if so which file is it.
"""

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

#: §3.2. A custom name may not be any of these, with §4.4's one exception: a canonical
#: suit is legal as a `[suits]` *table key*, because that is how a deck modifies it.
RESERVED_NAMES = frozenset({MAJOR, MINOR, *CANONICAL_SUITS, *CANONICAL_RANKS})

#: §3.5 `custom-name = name-start *name-char`. Lowercase, digits and underscore; no
#: leading digit.
_CUSTOM_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")
#: §3.5 `canonical-major = 2DIGIT`, and §3.5's normative note that both digits are written.
_TWO_DIGIT = re.compile(r"[0-9]{2}\Z")

#: The canonical twenty-two. Extended majors run to `99` but only these have a meaning
#: shared between decks (§3.1.1).
CANONICAL_MAJORS = tuple(f"{n:02d}" for n in range(22))


class IdError(ValueError):
    """A string that was offered as a canonical ID or custom name and is not one."""


def is_custom_name(name: str) -> bool:
    """§3.2: well-formed under the grammar and not a reserved canonical key."""
    return bool(_CUSTOM_NAME.fullmatch(name)) and name not in RESERVED_NAMES


def check_custom_name(name: str, what: str) -> str:
    """`is_custom_name`, but raising with a reason. Used for variant and design keys."""
    if not _CUSTOM_NAME.fullmatch(name):
        raise IdError(
            f"{what} {name!r} is not a custom name: §3.5 allows lowercase letters, digits "
            "and underscores, and forbids a leading digit"
        )
    if name in RESERVED_NAMES:
        raise IdError(f"{what} {name!r} is a reserved canonical key (§3.2)")
    return name


def check_major_key(key: str) -> str:
    """A major arcana key: two digits, or a custom name that is *not* two digits.

    The second clause is §3.2's extra rule for majors, and it is what keeps a custom key
    from colliding with a numbered slot. It never fires on a two-digit string, because
    such a string is read as `canonical-major` first — `major_arcana.23` is the deck's
    twenty-fourth major arcanum, not a malformed custom key.
    """
    if _TWO_DIGIT.fullmatch(key):
        return key
    if key.isdigit():
        raise IdError(
            f"major arcana key {key!r} must be written with both digits (§3.5), e.g. "
            f"{int(key):02d}"
        )
    return check_custom_name(key, "major arcana key")


@dataclass(frozen=True, order=True)
class CardId:
    """A parsed canonical ID. `suit` and `rank` are set for a minor arcanum, `key` for a
    major one; exactly one of the two shapes is populated."""

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
        """The filename base this card's assets use (§5.7.2)."""
        return self.key if self.kind == "major" else self.rank  # type: ignore[return-value]

    @property
    def subpath(self) -> PurePosixPath:
        """The directory a deck files this card under, within an image root (§5.7.1)."""
        if self.kind == "major":
            return PurePosixPath(MAJOR)
        return PurePosixPath(MINOR, self.suit)  # type: ignore[arg-type]

    def relpath(self, variant: str | None = None, ext: str = "png") -> PurePosixPath:
        """Where this card's image sits inside an image root, or inside `masters/`.

        A variant key is infixed between base and extension (§4.7), which is why the
        mapping stays injective: §3.5 forbids a `.` inside a variant key, so the stem
        splits back apart unambiguously.
        """
        stem = (
            self.base
            if variant is None
            else f"{self.base}.{check_custom_name(variant, 'variant key')}"
        )
        return self.subpath / f"{stem}.{ext}"

    @property
    def is_extended_major(self) -> bool:
        """A numbered major beyond the canonical twenty-two (§1.3)."""
        return (
            self.kind == "major"
            and bool(_TWO_DIGIT.fullmatch(self.key or ""))
            and self.key not in CANONICAL_MAJORS
        )


def parse_card_id(cid: str) -> CardId:
    """Parse a canonical ID, or raise `IdError` saying which rule it broke."""
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
    raise IdError(f"{cid!r}: a canonical ID begins with {MAJOR}. or {MINOR}. (§3.1)")


def card_id_from_relpath(rel: PurePosixPath) -> tuple[CardId, str | None]:
    """Invert `CardId.relpath`, returning the card and its variant key.

    Raises `IdError` for a path that is not a card asset, which is how the `masters/` walk
    tells a stray file from a card.
    """
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
    """The canonical seventy-eight, in spec order (§4.3.2): majors, then suits by rank.

    This is a *starting point* for a new project's roster and nothing more. 2.0 gives no
    fixed denominator — ranks can be added, majors run to `99`, cards can be custom-keyed
    and `[excluded_cards]` removes slots on purpose — so 78 must never be a constant in
    code that counts a deck. See [[ADR-003]] §roster.
    """
    cards = [CardId("major", key=k) for k in CANONICAL_MAJORS]
    for suit in CANONICAL_SUITS:
        cards += [CardId("minor", suit=suit, rank=r) for r in CANONICAL_RANKS]
    return cards
