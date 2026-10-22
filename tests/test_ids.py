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
    resolve_ref,
)


@pytest.mark.parametrize(
    "cid",
    [
        "major_arcana.00",
        "major_arcana.21",
        "major_arcana.23",  # extended major
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
        "major_arcana.6",  # both digits are written
        "major_arcana.006",
        "minor_arcana.cups.King",  # lowercase only
        "minor_arcana.Cups.king",
        "major_arcana.2women",  # no leading digit
        "major_arcana.two-women",  # no hyphen
        "minor_arcana.cups",
        "minor_arcana.cups.king.extra",
        "major_arcana",
        "trumps.00",
        "minor_arcana.cups.wands",  # a rank key may not shadow a canonical suit
        "minor_arcana.king.ace",  # nor a suit key a canonical rank
        "",
    ],
)
def test_rejects(cid):
    with pytest.raises(IdError):
        parse_card_id(cid)


def test_two_digit_string_is_a_major_key_but_not_a_custom_name():
    assert check_major_key("23") == "23"
    assert not is_custom_name("23")
    with pytest.raises(IdError):
        check_custom_name("23", "variant key")


def test_reserved_names_rejected_as_custom():
    for name in ("cups", "king", "major_arcana", "minor_arcana"):
        assert not is_custom_name(name)


def test_resolve_ref():
    assert resolve_ref("major_arcana.06") == (parse_card_id("major_arcana.06"), None)
    assert resolve_ref("major_arcana.06:two_women") == (
        parse_card_id("major_arcana.06"),
        "two_women",
    )


@pytest.mark.parametrize("ref", ["major_arcana.06:", "major_arcana.6:two_women", ":two_women"])
def test_resolve_ref_rejects(ref):
    with pytest.raises(IdError):
        resolve_ref(ref)


def test_relpath_mapping():
    assert parse_card_id("major_arcana.03").relpath() == PurePosixPath("major_arcana/03.png")
    assert parse_card_id("minor_arcana.cups.king").relpath() == PurePosixPath(
        "minor_arcana/cups/king.png"
    )
    assert parse_card_id("major_arcana.06").relpath("two_women") == PurePosixPath(
        "major_arcana/06.two_women.png"
    )


def test_relpath_round_trips():
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
