from __future__ import annotations

import errno
import os

import pytest

from deckle.labels import COMMENT_XATTR
from deckle.project import PROJECT_FILE, Project, init
from deckle.render import status_report
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


def test_a_variant_is_listed_under_its_card_not_counted(tmp_path):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/06.png")
    touch_master(project, "major_arcana/06.two_women.png")
    st = status(project)
    assert [m.ref for m in st.present] == ["major_arcana.06"]
    assert st.variants == {"major_arcana.06": ["major_arcana.06:two_women"]}
    assert st.no_default == []
    assert st.total == 78
    assert "major_arcana.06" not in st.missing


def test_a_variant_alone_does_not_make_a_card_present(tmp_path, capsys):
    project = make_project(tmp_path)
    touch_master(project, "major_arcana/02.crowned.png")
    touch_master(project, "major_arcana/03.png")
    st = status(project)
    assert [m.ref for m in st.present] == ["major_arcana.03"]
    assert "major_arcana.02" in st.missing and len(st.missing) == 77
    assert st.no_default == ["major_arcana.02"]

    status_report(st)
    out = capsys.readouterr().out
    assert "  1 of 78 roster cards present\n" in out
    assert "    major_arcana.02  (has variants, but no default master)\n" in out
    assert (
        "    major_arcana.02  variants: major_arcana.02:crowned  — missing its default master\n"
        in out
    )


def test_present_counts_distinct_cards_with_a_bare_master(tmp_path, capsys):
    project = make_project(tmp_path)
    for rel in ("major_arcana/06.png", "major_arcana/06.a.png", "major_arcana/06.b.png"):
        touch_master(project, rel)
    status_report(status(project))
    out = capsys.readouterr().out
    assert "  1 of 78 roster cards present\n" in out
    assert "    major_arcana.06  variants: major_arcana.06:a, major_arcana.06:b\n" in out


def label(path, text: str) -> None:
    try:
        os.setxattr(path, COMMENT_XATTR, text.encode())
    except OSError as exc:
        if exc.errno == errno.ENOTSUP:
            pytest.skip(f"{path.parent} does not support user xattrs")
        raise


def test_the_staging_line_counts_the_plan(tmp_path, capsys):
    project = make_project(tmp_path)
    touch_master(project, "minor_arcana/wands/seven.png")
    for name, text in [("a_r0c0.png", "m2"), ("a_r0c1.png", "w7"), ("a_r1c0.png", "M3")]:
        (project.staging_dir / name).write_bytes(b"pixels")
        label(project.staging_dir / name, text)
    (project.staging_dir / "a_r1c1.png").write_bytes(b"pixels")

    st = status(project)
    # the replace counts as a problem, as in a dry run without --force
    assert (st.staging.files, st.staging.labelled, st.staging.unlabelled) == (4, 3, 1)
    assert len(st.staging.problems) == 2
    status_report(st)
    out = capsys.readouterr().out
    assert (
        "  staging: 4 files, 3 labelled, 1 unlabelled, 2 problem(s)  (deckle assign for detail)\n"
        in out
    )


def test_no_staged_pngs_means_no_staging_line(tmp_path, capsys):
    project = make_project(tmp_path)
    (project.staging_dir / "index.toml").write_text("")
    st = status(project)
    assert st.staging is None
    status_report(st)
    assert "staging:" not in capsys.readouterr().out


def test_a_degraded_project_skips_the_staging_line(tmp_path):
    project = make_project(tmp_path)
    (project.staging_dir / "a_r0c0.png").write_bytes(b"pixels")
    (project.root / PROJECT_FILE).unlink()
    assert status(Project.load(project.root)).staging is None


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
