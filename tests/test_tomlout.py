from __future__ import annotations

import tomllib

from deckle.tomlout import dumps


def test_round_trips_and_preserves_order():
    doc = {
        "project": {"dpi": 600.0, "emit_heights": [750, 1200, 2400], "aspect": 0.5},
        "deck": {"name": 'a "quoted" name', "tags": ["a", "b"], "flag": True, "n": 3},
        "cards": {"minor_arcana.cups.king": {"slot": "r0c1"}},
    }
    text = dumps(doc)
    assert tomllib.loads(text) == doc
    assert list(tomllib.loads(text)["project"]) == ["dpi", "emit_heights", "aspect"]
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_quotes_dotted_keys():
    text = dumps({"cards": {"major_arcana.03": {"slot": "r0c0"}}})
    assert '[cards."major_arcana.03"]' in text
    assert tomllib.loads(text)["cards"]["major_arcana.03"]["slot"] == "r0c0"


def test_writes_inline_tables_in_arrays():
    doc = {"deck": {"links": [{"rel": "homepage", "url": "https://example.com/"}]}}
    assert tomllib.loads(dumps(doc)) == doc


def test_keeps_bools_out_of_ints():
    assert dumps({"a": True, "b": 1}) == "a = true\nb = 1\n"
