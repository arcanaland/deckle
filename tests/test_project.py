"""The project directory: round-tripping, atomic saves, and the degraded status path.

Two of these are the ADR's own claims rather than ordinary unit tests:

- `test_crash_mid_write_leaves_previous_state_intact` — a truncated `deckle.toml` after a
  crash would look exactly like lost work.
- `test_status_survives_a_deleted_state_file` — [[ADR-003]]'s redundancy claim. Identity
  lives in the filenames under `masters/`, so losing the state file costs the roster and
  the provenance and nothing else.
"""

from __future__ import annotations

import tomllib

import pytest

from deckle.assign import AssignError, assign
from deckle.ids import IdError
from deckle.project import (
    PROJECT_FILE,
    Project,
    ProjectError,
    Roster,
    find_project,
    init,
    status,
    write_staging_index,
)
from deckle.tomlout import dumps


def make_project(tmp_path, **kw):
    return init(tmp_path / "proj", "Test Deck", **kw)


def touch_master(project, relpath: str) -> None:
    path = project.masters_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a png")


# -- the writer ----------------------------------------------------------------------


def test_tomlout_round_trips_and_preserves_order():
    doc = {
        "project": {"dpi": 600.0, "emit_heights": [750, 1200, 2400], "aspect": 0.583},
        "deck": {"name": 'a "quoted" name', "tags": ["a", "b"], "flag": True, "n": 3},
        "cards": {"minor_arcana.cups.king": {"slot": "r0c1"}},
    }
    text = dumps(doc)
    assert tomllib.loads(text) == doc
    assert list(tomllib.loads(text)["project"]) == ["dpi", "emit_heights", "aspect"]
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_tomlout_quotes_dotted_keys():
    text = dumps({"cards": {"major_arcana.03": {"slot": "r0c0"}}})
    assert '[cards."major_arcana.03"]' in text
    assert tomllib.loads(text)["cards"]["major_arcana.03"]["slot"] == "r0c0"


def test_tomlout_writes_inline_tables_in_arrays():
    doc = {"deck": {"links": [{"rel": "homepage", "url": "https://example.com/"}]}}
    assert tomllib.loads(dumps(doc)) == doc


def test_tomlout_keeps_bools_out_of_ints():
    assert dumps({"a": True, "b": 1}) == "a = true\nb = 1\n"


# -- load / save ---------------------------------------------------------------------


def test_init_creates_the_layout(tmp_path):
    project = make_project(tmp_path)
    for sub in ("masters", "staging", "names"):
        assert (project.root / sub).is_dir()
    assert (project.root / PROJECT_FILE).is_file()
    assert project.roster == Roster.canonical()
    assert len(project.roster.cards()) == 78


def test_init_refuses_to_clobber(tmp_path):
    make_project(tmp_path)
    with pytest.raises(ProjectError):
        init(tmp_path / "proj", "Test Deck")


def test_unknown_keys_survive_a_round_trip(tmp_path):
    """A key deckle does not model is authorship, not noise. Losing it silently on save
    would be the state file quietly editing the user's work."""
    project = make_project(tmp_path)
    project.doc["deck"]["some_future_key"] = "keep me"
    project.doc["totally_unknown"] = {"a": 1}
    project.save()

    reloaded = Project.load(project.root)
    assert reloaded.doc["deck"]["some_future_key"] == "keep me"
    assert reloaded.doc["totally_unknown"] == {"a": 1}
    assert list(reloaded.doc) == list(project.doc)


def test_crash_mid_write_leaves_previous_state_intact(tmp_path, monkeypatch):
    """Simulate a crash between writing the temp file and renaming it."""
    project = make_project(tmp_path)
    project.doc["deck"]["name"] = "Original"
    project.save()
    before = (project.root / PROJECT_FILE).read_bytes()

    def boom(src, dst):
        raise OSError("crash before rename")

    monkeypatch.setattr("deckle.project.os.replace", boom)
    project.doc["deck"]["name"] = "Half-written"
    with pytest.raises(OSError):
        project.save()

    assert (project.root / PROJECT_FILE).read_bytes() == before
    assert Project.load(project.root).deck["name"] == "Original"
    # And no temp file is left lying around to be mistaken for state.
    assert not list(project.root.glob(f".{PROJECT_FILE}.*"))


def test_find_project_walks_up(tmp_path):
    project = make_project(tmp_path)
    deep = project.root / "masters" / "minor_arcana" / "cups"
    deep.mkdir(parents=True, exist_ok=True)
    assert find_project(deep) == project.root
    assert find_project(tmp_path) is None


# -- masters as the second index -----------------------------------------------------


def test_masters_walk_finds_cards_variants_and_backs(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    touch_master(project, "major_arcana/06.two_women.png")
    touch_master(project, "minor_arcana/cups/king.png")
    touch_master(project, "card_backs/classic.png")

    refs = [m.ref for m in project.masters()]
    assert refs == [
        "major_arcana.03",
        "major_arcana.06:two_women",
        "minor_arcana.cups.king",
        "classic",
    ]
    assert [m.ref for m in project.masters() if m.is_back] == ["classic"]


def test_masters_walk_skips_unrecognised_files(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    touch_master(project, "major_arcana/King.png")
    touch_master(project, "somewhere_else/thing.png")

    assert [m.ref for m in project.masters()] == ["major_arcana.03"]
    assert [p.name for p in project.strays()] == ["King.png", "thing.png"]


def test_variants_do_not_double_count(tmp_path):
    """A variant is other artwork for a card the deck already has, not another card."""
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/06.png")
    touch_master(project, "major_arcana/06.two_women.png")
    assert len(status(project).present) == 2  # two files
    assert status(project).total == 78
    assert "major_arcana.06" not in status(project).missing


# -- status ---------------------------------------------------------------------------


def test_empty_project_reports_zero_of_78_and_exits_zero(tmp_path):
    from deckle.cli import main

    project = make_project(tmp_path)
    st = status(project)
    assert (len(st.present), st.total) == (0, 78)
    assert len(st.missing) == 78
    assert main(["status", "--project", str(project.root)]) == 0


def test_status_exits_zero_when_incomplete(tmp_path):
    from deckle.cli import main

    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    assert main(["status", "--project", str(project.root)]) == 0


def test_status_survives_a_deleted_state_file(tmp_path, capsys):
    """[[ADR-003]]'s redundancy claim, as a test. This is a required behaviour."""
    from deckle.cli import main

    project = make_project(tmp_path)
    for rel in ("major_arcana/03.png", "minor_arcana/cups/king.png", "card_backs/classic.png"):
        touch_master(project, rel)
    (project.root / PROJECT_FILE).unlink()

    reloaded = Project.load(project.root)
    assert reloaded.degraded
    st = status(reloaded)
    assert st.total is None  # no roster, so no denominator may be invented
    assert [m.ref for m in st.present] == ["major_arcana.03", "minor_arcana.cups.king"]
    assert st.backs == ["classic"]

    assert main(["status", "--project", str(project.root)]) == 0
    out = capsys.readouterr().out
    assert "major_arcana.03" in out and "minor_arcana.cups.king" in out
    assert "roster and provenance unavailable" in out


def test_status_survives_a_corrupt_state_file(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    (project.root / PROJECT_FILE).write_text("this is not [ valid toml", encoding="utf-8")

    reloaded = Project.load(project.root)
    assert reloaded.degraded and "not valid TOML" in reloaded.degraded_reason
    assert [m.ref for m in status(reloaded).present] == ["major_arcana.03"]


def test_degraded_project_refuses_to_save(tmp_path):
    """Saving a degraded load would write a fresh default over whatever is there."""
    project = make_project(tmp_path)
    (project.root / PROJECT_FILE).write_text("broken [", encoding="utf-8")
    reloaded = Project.load(project.root)
    with pytest.raises(ProjectError):
        reloaded.save()
    assert (project.root / PROJECT_FILE).read_text() == "broken ["


def test_status_reports_masters_the_roster_does_not_mention(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/23.png")
    st = status(project)
    assert st.unexpected == ["major_arcana.23"]


def test_roster_excludes_are_removed_from_the_denominator(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"]["excluded"] = [
        "minor_arcana.pentacles.page",
        "minor_arcana.pentacles.knight",
    ]
    project.save()
    assert status(Project.load(project.root)).total == 76


# -- assign ---------------------------------------------------------------------------


def stage(project, name: str) -> object:
    path = project.staging_dir / name
    path.write_bytes(b"pixels")
    return path


def test_assign_moves_and_records_provenance(tmp_path):
    project = make_project(tmp_path)
    src = stage(project, "20260804193927_001_r0c0.png")
    write_staging_index(
        project,
        {
            src.name: {
                "source_scan": "20260804193927_001.jpg",
                "slot": "r0c0",
                "width_mm": 70.207,
                "height_mm": 120.26,
                "skew_deg": -0.31,
                "edge_strategy": "brightest",
            }
        },
    )

    assign(project, src, "minor_arcana.pentacles.king")

    assert not src.exists()
    assert (project.masters_dir / "minor_arcana/pentacles/king.png").is_file()

    row = Project.load(project.root).cards["minor_arcana.pentacles.king"]
    assert row["source_scan"] == "20260804193927_001.jpg"
    assert row["slot"] == "r0c0"
    assert row["width_mm"] == 70.207
    assert row["skew_deg"] == -0.31
    assert row["rotate_180"] is False
    assert row["master"] == "masters/minor_arcana/pentacles/king.png"
    assert row["deckle_version"] and row["assigned_at"].endswith("Z")


def test_assign_recovers_scan_and_slot_without_an_index(tmp_path):
    """A file dropped into staging/ by hand still names its scan and slot. The
    measurements are omitted rather than guessed."""
    project = make_project(tmp_path)
    src = stage(project, "20260801204347_001_r0c1.png")
    assign(project, src, "minor_arcana.cups.king")
    row = project.cards["minor_arcana.cups.king"]
    assert row["source_scan"] == "20260801204347_001.jpg"
    assert row["slot"] == "r0c1"
    assert "width_mm" not in row


def test_reassigning_leaves_no_master_behind(tmp_path):
    project = make_project(tmp_path)
    assign(project, stage(project, "s_r0c0.png"), "major_arcana.03")
    assert (project.masters_dir / "major_arcana/03.png").is_file()

    # Same card, re-read as a different one.
    assign(project, stage(project, "s_r0c0.png"), "major_arcana.03")  # same slot again
    src = project.masters_dir / "major_arcana/03.png"
    assert src.is_file()

    # And now to a genuinely different ID: the old file must not survive.
    staged = project.staging_dir / "s2_r0c0.png"
    staged.write_bytes(b"pixels")
    assign(project, staged, "major_arcana.09")
    reloaded = Project.load(project.root)
    assert reloaded.cards["major_arcana.09"]["master"] == "masters/major_arcana/09.png"
    assert [m.ref for m in reloaded.masters()] == ["major_arcana.03", "major_arcana.09"]


def test_assign_rejects_malformed_ids(tmp_path):
    project = make_project(tmp_path)
    for bad in ("major_arcana.6", "minor_arcana.cups.King", "major_arcana.2women"):
        src = stage(project, "s_r0c0.png")
        with pytest.raises(IdError):
            assign(project, src, bad)
        assert src.exists(), "a rejected assignment must not move the file"


def test_assign_accepts_extended_and_custom_ids(tmp_path):
    project = make_project(tmp_path)
    for i, cid in enumerate(
        ["major_arcana.23", "major_arcana.happy_squirrel", "minor_arcana.stars.ace"]
    ):
        assign(project, stage(project, f"s_r0c{i}.png"), cid)
    assert {m.ref for m in project.masters()} == {
        "major_arcana.23",
        "major_arcana.happy_squirrel",
        "minor_arcana.stars.ace",
    }


def test_assign_card_back(tmp_path):
    project = make_project(tmp_path)
    assign(project, stage(project, "s_r0c0.png"), None, card_back="classic")
    assert (project.masters_dir / "card_backs/classic.png").is_file()
    assert project.cards["card_backs.classic"]["slot"] == "r0c0"


def test_assign_needs_exactly_one_target(tmp_path):
    project = make_project(tmp_path)
    src = stage(project, "s_r0c0.png")
    with pytest.raises(AssignError):
        assign(project, src, None)
    with pytest.raises(AssignError):
        assign(project, src, "major_arcana.03", card_back="classic")


def test_assign_variant(tmp_path):
    project = make_project(tmp_path)
    assign(project, stage(project, "s_r0c0.png"), "major_arcana.06:two_women")
    assert (project.masters_dir / "major_arcana/06.two_women.png").is_file()


def test_assign_consumes_the_staging_index_row(tmp_path):
    from deckle.project import read_staging_index

    project = make_project(tmp_path)
    src = stage(project, "s_r0c0.png")
    write_staging_index(project, {src.name: {"source_scan": "s.jpg", "slot": "r0c0"}})
    assign(project, src, "major_arcana.03")
    assert read_staging_index(project) == {}
