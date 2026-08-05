"""The canonical-ID grammar of `DECK.md` 2.0 §3.5 and the reserved names of §3.2.

These are cheap tests on a small module, and they are worth having because `assign` is the
one stage that writes a name a human will later trust. A malformed ID that reaches
`masters/` breaks the filesystem-as-index property everything else rests on.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from deckle.ids import (
    CardId,
    IdError,
    canonical_78,
    card_id_from_relpath,
    check_custom_name,
    check_major_key,
    is_custom_name,
    parse_card_id,
)


@pytest.mark.parametrize(
    "cid",
    [
        "major_arcana.00",
        "major_arcana.21",
        "major_arcana.23",  # an extended major: legal, just not one Appendix C names
        "major_arcana.99",
        "major_arcana.happy_squirrel",
        "major_arcana.the_morning",
        "minor_arcana.cups.king",
        "minor_arcana.wands.ace",
        "minor_arcana.stars.ace",  # custom suit, canonical rank
        "minor_arcana.cups.princess",  # canonical suit, custom rank
    ],
)
def test_accepts(cid):
    assert str(parse_card_id(cid)) == cid


@pytest.mark.parametrize(
    "cid",
    [
        "major_arcana.6",  # §3.5: both digits are written
        "major_arcana.006",
        "minor_arcana.cups.King",  # §3.5 is lowercase throughout
        "minor_arcana.Cups.king",
        "major_arcana.2women",  # a custom name may not start with a digit
        "major_arcana.two-women",  # nor contain a hyphen
        "minor_arcana.cups",  # a minor ID has three parts
        "minor_arcana.cups.king.extra",
        "major_arcana",
        "trumps.00",
        "minor_arcana.cups.wands",  # §3.2: a rank key may not shadow a canonical suit
        "minor_arcana.king.ace",  # nor a suit key a canonical rank
        "",
    ],
)
def test_rejects(cid):
    with pytest.raises(IdError):
        parse_card_id(cid)


def test_two_digit_string_is_not_a_custom_name():
    """§3.2: a custom major arcana key MUST NOT be a two-digit string.

    The rule can never fire through `parse_card_id`, because two digits are read as
    `canonical-major` first — `major_arcana.23` is a card, not a malformed custom key. It
    bites where a *custom name* is checked as one, which is what this asserts.
    """
    assert check_major_key("23") == "23"  # as a major key: the deck's 24th major arcanum
    assert not is_custom_name("23")  # as a custom name: rejected
    with pytest.raises(IdError):
        check_custom_name("23", "variant key")


def test_reserved_names_rejected_as_custom():
    for name in ("cups", "king", "major_arcana", "minor_arcana"):
        assert not is_custom_name(name)


def test_relpath_mapping():
    assert parse_card_id("major_arcana.03").relpath() == PurePosixPath("major_arcana/03.png")
    assert parse_card_id("minor_arcana.cups.king").relpath() == PurePosixPath(
        "minor_arcana/cups/king.png"
    )
    assert parse_card_id("major_arcana.06").relpath("two_women") == PurePosixPath(
        "major_arcana/06.two_women.png"
    )


def test_relpath_round_trips():
    """The mapping is injective, which is what makes `masters/` a faithful second index."""
    for card in canonical_78():
        back, variant = card_id_from_relpath(card.relpath())
        assert back == card and variant is None
    card = parse_card_id("major_arcana.06")
    back, variant = card_id_from_relpath(card.relpath("two_women"))
    assert back == card and variant == "two_women"


def test_relpath_rejects_non_card_paths():
    for bad in ("card_backs/classic.png", "major_arcana/cups/king.png", "loose.png"):
        with pytest.raises(IdError):
            card_id_from_relpath(PurePosixPath(bad))


def test_canonical_78_is_78_and_unique():
    cards = canonical_78()
    assert len(cards) == 78
    assert len({str(c) for c in cards}) == 78
    assert cards[0] == CardId("major", key="00")
    assert str(cards[-1]) == "minor_arcana.pentacles.king"
