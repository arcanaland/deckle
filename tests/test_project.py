from __future__ import annotations

import tomllib

import pytest

from deckle.detect import CardSpec
from deckle.project import (
    DECK_IN_FILE,
    PROJECT_FILE,
    Config,
    Project,
    ProjectError,
    Roster,
    find_project,
    init,
)
from deckle.units import EdgeStrategy


def make_project(tmp_path, **kw):
    return init(tmp_path / "proj", "Test Deck", **kw)


def touch_master(project, relpath: str) -> None:
    path = project.masters_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a png")


def write_doc(project, doc) -> Project:
    from deckle.tomlout import dumps

    (project.root / PROJECT_FILE).write_text(dumps(doc), encoding="utf-8")
    return Project.load(project.root)


# -- load / save ---------------------------------------------------------------------


def test_init_creates_the_layout(tmp_path):
    project = make_project(tmp_path)
    for sub in ("masters", "staging", "names"):
        assert (project.root / sub).is_dir()
    assert (project.root / PROJECT_FILE).is_file()
    assert project.roster == Roster.canonical()
    assert len(project.roster.cards()) == 78


def test_init_writes_the_authored_file_and_no_deck_table(tmp_path):
    project = init(tmp_path / "proj", 'A "Quoted" Deck')
    assert "deck" not in project.doc and "card_meta" not in project.doc
    assert project.authored() == {"deck": {"name": 'A "Quoted" Deck', "version": "0.1"}}
    assert (project.root / DECK_IN_FILE).read_text().startswith("#")


def test_init_keeps_an_existing_authored_file(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / DECK_IN_FILE).write_text('[deck]\nname = "Mine"\nversion = "2"\n', encoding="utf-8")
    assert init(root, "Other").authored()["deck"]["name"] == "Mine"


def test_an_unparseable_authored_file_is_an_error(tmp_path):
    project = make_project(tmp_path)
    (project.root / DECK_IN_FILE).write_text("[deck\n", encoding="utf-8")
    with pytest.raises(ProjectError, match=DECK_IN_FILE):
        project.authored()


def test_init_refuses_to_clobber(tmp_path):
    make_project(tmp_path)
    with pytest.raises(ProjectError):
        init(tmp_path / "proj", "Test Deck")


def test_unknown_keys_survive_a_round_trip(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["some_future_key"] = "keep me"
    project.doc["totally_unknown"] = {"a": 1}
    project.save()

    reloaded = Project.load(project.root)
    assert reloaded.doc["project"]["some_future_key"] == "keep me"
    assert reloaded.doc["totally_unknown"] == {"a": 1}
    assert list(reloaded.doc) == list(project.doc)


def test_crash_mid_write_leaves_previous_state_intact(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    project.doc["project"]["deck_dir"] = "original"
    project.save()
    before = (project.root / PROJECT_FILE).read_bytes()

    def boom(src, dst):
        raise OSError("crash before rename")

    monkeypatch.setattr("deckle.project.os.replace", boom)
    project.doc["project"]["deck_dir"] = "half-written"
    with pytest.raises(OSError):
        project.save()

    assert (project.root / PROJECT_FILE).read_bytes() == before
    assert Project.load(project.root).config.deck_dir == "original"
    assert not list(project.root.glob(f".{PROJECT_FILE}.*"))


def test_find_project_walks_up(tmp_path):
    project = make_project(tmp_path)
    deep = project.root / "masters" / "minor_arcana" / "cups"
    deep.mkdir(parents=True, exist_ok=True)
    assert find_project(deep) == project.root
    assert find_project(tmp_path) is None


def test_find_project_without_a_state_file(tmp_path):
    project = make_project(tmp_path)
    (project.root / PROJECT_FILE).unlink()
    assert find_project(project.root / "masters") == project.root


# -- degraded mode -------------------------------------------------------------------


def test_a_deleted_state_file_degrades_but_keeps_the_masters(tmp_path):
    project = make_project(tmp_path)
    for rel in ("major_arcana/03.png", "minor_arcana/cups/king.png", "card_backs/classic.png"):
        touch_master(project, rel)
    (project.root / PROJECT_FILE).unlink()

    reloaded = Project.discover(project.root)
    assert reloaded.degraded and "missing" in reloaded.degraded_reason
    assert [m.ref for m in reloaded.masters()] == [
        "major_arcana.03",
        "minor_arcana.cups.king",
        "classic",
    ]


def test_a_corrupt_state_file_degrades(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    (project.root / PROJECT_FILE).write_text("this is not [ valid toml", encoding="utf-8")

    reloaded = Project.load(project.root)
    assert reloaded.degraded and "not valid TOML" in reloaded.degraded_reason
    assert [m.ref for m in reloaded.masters()] == ["major_arcana.03"]


def test_degraded_project_refuses_to_save(tmp_path):
    project = make_project(tmp_path)
    (project.root / PROJECT_FILE).write_text("broken [", encoding="utf-8")
    reloaded = Project.load(project.root)
    with pytest.raises(ProjectError):
        reloaded.save()
    assert (project.root / PROJECT_FILE).read_text() == "broken ["


def test_discover_without_a_project(tmp_path):
    with pytest.raises(ProjectError, match="deckle init"):
        Project.discover(start=tmp_path)


# -- masters/ ------------------------------------------------------------------------


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


# -- roster --------------------------------------------------------------------------


def test_roster_excludes_are_removed(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"]["excluded"] = [
        "minor_arcana.pentacles.page",
        "minor_arcana.pentacles.knight",
    ]
    project.save()
    assert len(Project.load(project.root).roster.cards()) == 76


def test_absent_suits_are_the_canonical_four():
    roster = Roster.from_doc({"majors": ["00", "01"]})
    assert len(roster.cards()) == 2 + 56


def test_empty_suits_are_no_suits(tmp_path):
    project = make_project(tmp_path)
    doc = project.doc
    doc["roster"] = {"majors": ["22", "23"], "excluded": [], "suits": {}}
    reloaded = write_doc(project, doc)
    assert [str(c) for c in reloaded.roster.cards()] == ["major_arcana.22", "major_arcana.23"]


def test_declared_suits_are_as_declared():
    roster = Roster.from_doc({"majors": [], "suits": {"stars": ["ace", "princess"]}})
    assert [str(c) for c in roster.cards()] == [
        "minor_arcana.stars.ace",
        "minor_arcana.stars.princess",
    ]


def test_empty_suits_survive_a_save(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"] = Roster(majors=("22",), suits={}).to_doc()
    project.save()
    text = (project.root / PROJECT_FILE).read_text()
    assert "[roster.suits]" in text

    reloaded = Project.load(project.root)
    assert reloaded.roster.suits == {}
    reloaded.save()
    assert (project.root / PROJECT_FILE).read_text() == text


# -- config --------------------------------------------------------------------------


def test_config_defaults_match_the_detector(tmp_path):
    config = make_project(tmp_path).config
    spec = CardSpec()
    assert (config.card_width_mm, config.card_height_mm) == spec.card_mm
    assert config.aspect == spec.aspect
    assert config.opposite_tol_mm == spec.opposite_tol_mm


def test_config_round_trips():
    config = Config.from_doc(
        {"opposite_tol_mm": 0.40, "edge_strategy": "outermost", "deck_dir": "~/decks/x"}
    )
    assert config.opposite_tol_mm == 0.40
    assert config.edge_strategy is EdgeStrategy.OUTERMOST
    assert Config.from_doc(config.to_doc()) == config


def test_edge_strategy_is_written_as_a_plain_string(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["edge_strategy"] = "outermost"
    project.save()
    text = (project.root / PROJECT_FILE).read_text()
    assert 'edge_strategy = "outermost"' in text
    assert tomllib.loads(text)["project"]["edge_strategy"] == "outermost"


def test_unknown_edge_strategy_is_refused(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["edge_strategy"] = "outer_most"
    with pytest.raises(ProjectError, match="outer_most"):
        _ = project.config


def test_deck_dir_resolves_against_the_project(tmp_path):
    project = make_project(tmp_path, deck_dir="out/deck")
    assert project.deck_dir() == project.root / "out" / "deck"
    project.doc["project"].pop("deck_dir")
    with pytest.raises(ProjectError, match="deck_dir"):
        project.deck_dir()
