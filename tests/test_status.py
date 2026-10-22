from __future__ import annotations

from deckle.project import PROJECT_FILE, Project, init
from deckle.status import status
from deckle.units import EdgeStrategy


def make_project(tmp_path, **kw):
    return init(tmp_path / "proj", "Test Deck", **kw)


def touch_master(project, relpath: str) -> None:
    path = project.masters_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really a png")


def test_empty_project_is_zero_of_78(tmp_path):
    st = status(make_project(tmp_path))
    assert (len(st.present), st.total) == (0, 78)
    assert len(st.missing) == 78
    assert st.edge_strategy == EdgeStrategy.BRIGHTEST
    assert st.config_problem is None


def test_variants_do_not_double_count(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/06.png")
    touch_master(project, "major_arcana/06.two_women.png")
    st = status(project)
    assert len(st.present) == 2
    assert st.total == 78
    assert "major_arcana.06" not in st.missing


def test_a_deleted_state_file_reports_no_total(tmp_path):
    project = make_project(tmp_path)
    for rel in ("major_arcana/03.png", "minor_arcana/cups/king.png", "card_backs/classic.png"):
        touch_master(project, rel)
    (project.root / PROJECT_FILE).unlink()

    st = status(Project.load(project.root))
    assert "missing" in st.degraded_reason
    assert st.total is None
    assert [m.ref for m in st.present] == ["major_arcana.03", "minor_arcana.cups.king"]
    assert st.backs == ["classic"]
    assert st.missing == [] and st.unexpected == []


def test_a_corrupt_state_file_still_lists_the_masters(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    (project.root / PROJECT_FILE).write_text("this is not [ valid toml", encoding="utf-8")

    st = status(Project.load(project.root))
    assert "not valid TOML" in st.degraded_reason
    assert [m.ref for m in st.present] == ["major_arcana.03"]


def test_an_unknown_edge_strategy_is_reported_not_raised(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    project.doc["project"]["edge_strategy"] = "sharpest"
    project.save()

    st = status(Project.load(project.root))
    assert st.edge_strategy is None
    assert "'sharpest' is not a known strategy" in st.config_problem
    assert (len(st.present), st.total) == (1, 78)


def test_masters_the_roster_does_not_mention_are_unexpected(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/23.png")
    assert status(project).unexpected == ["major_arcana.23"]


def test_roster_excludes_are_removed_from_the_total(tmp_path):
    project = make_project(tmp_path)
    project.doc["roster"]["excluded"] = [
        "minor_arcana.pentacles.page",
        "minor_arcana.pentacles.knight",
    ]
    project.save()
    st = status(Project.load(project.root))
    assert st.total == 76
    assert "minor_arcana.pentacles.page" not in st.missing


def test_strays_are_reported(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/03.png")
    touch_master(project, "minor_arcana/cups/King.png")
    assert [p.name for p in status(project).strays] == ["King.png"]
