from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from deckle.project import PROJECT_FILE, Project, init
from deckle.staging import (
    STAGING_INDEX,
    provenance_for,
    read_staging_index,
    record_rectified,
    write_staging_index,
)
from deckle.units import EdgeStrategy


def make_project(tmp_path):
    return init(tmp_path / "proj", "Test Deck")


def fake_card(row: int, col: int, **measured):
    """The attributes record_rectified reads from a detected card."""
    m = dict(width_mm=70.12345, height_mm=120.54321, aspect=0.5817234, skew_deg=-0.312345)
    m.update(measured)
    return SimpleNamespace(window=SimpleNamespace(row=row, col=col), **m)


def test_index_round_trips_sorted(tmp_path):
    project = make_project(tmp_path)
    write_staging_index(project, {"b.png": {"slot": "r0c1"}, "a.png": {"slot": "r0c0"}})
    assert list(read_staging_index(project)) == ["a.png", "b.png"]


def test_a_missing_or_corrupt_index_reads_empty(tmp_path):
    project = make_project(tmp_path)
    assert read_staging_index(project) == {}
    (project.staging_dir / STAGING_INDEX).write_text("not [ toml", encoding="utf-8")
    assert read_staging_index(project) == {}


def test_record_rectified_indexes_staged_masters(tmp_path):
    project = make_project(tmp_path)
    written = [project.staging_dir / "scan_r0c0.png", project.staging_dir / "scan_r1c1.png"]
    cards = [fake_card(0, 0), fake_card(1, 1, skew_deg=0.5)]

    record_rectified(
        project, Path("in/scan.jpg"), cards, written, strategy=EdgeStrategy.OUTERMOST, dpi=600.0
    )

    index = read_staging_index(project)
    assert index["scan_r0c0.png"] == {
        "source_scan": "scan.jpg",
        "slot": "r0c0",
        "width_mm": 70.1235,
        "height_mm": 120.5432,
        "aspect": 0.58172,
        "skew_deg": -0.3123,
        "edge_strategy": "outermost",
        "dpi": 600.0,
    }
    assert index["scan_r1c1.png"]["slot"] == "r1c1"
    assert index["scan_r1c1.png"]["skew_deg"] == 0.5


def test_record_rectified_keeps_earlier_rows(tmp_path):
    project = make_project(tmp_path)
    write_staging_index(project, {"old_r0c0.png": {"source_scan": "old.jpg"}})
    record_rectified(
        project,
        Path("new.jpg"),
        [fake_card(0, 0)],
        [project.staging_dir / "new_r0c0.png"],
        strategy="brightest",
        dpi=600.0,
    )
    assert list(read_staging_index(project)) == ["new_r0c0.png", "old_r0c0.png"]


def test_record_rectified_ignores_masters_written_elsewhere(tmp_path):
    project = make_project(tmp_path)
    record_rectified(
        project,
        Path("scan.jpg"),
        [fake_card(0, 0)],
        [tmp_path / "elsewhere" / "scan_r0c0.png"],
        strategy="brightest",
        dpi=600.0,
    )
    assert not (project.staging_dir / STAGING_INDEX).exists()


def test_record_rectified_does_nothing_in_a_degraded_project(tmp_path):
    project = make_project(tmp_path)
    (project.root / PROJECT_FILE).unlink()
    degraded = Project.load(project.root)
    record_rectified(
        degraded,
        Path("scan.jpg"),
        [fake_card(0, 0)],
        [degraded.staging_dir / "scan_r0c0.png"],
        strategy="brightest",
        dpi=600.0,
    )
    assert not (project.staging_dir / STAGING_INDEX).exists()


def test_provenance_carries_the_index_row():
    index = {
        "scan_r0c0.png": {
            "source_scan": "scan.jpg",
            "slot": "r0c0",
            "width_mm": 70.2,
            "edge_strategy": "brightest",
            "unrelated": 1,
        }
    }
    prov = provenance_for("scan_r0c0.png", index, rotate_180=True)
    assert prov["source_scan"] == "scan.jpg"
    assert prov["slot"] == "r0c0"
    assert prov["width_mm"] == 70.2
    assert prov["edge_strategy"] == "brightest"
    assert "unrelated" not in prov
    assert prov["rotate_180"] is True
    assert prov["deckle_version"]
    assert prov["assigned_at"].endswith("Z")


def test_provenance_without_an_index_reads_the_filename():
    prov = provenance_for("scan_001_r0c1.png", {}, rotate_180=False)
    assert prov["source_scan"] == "scan_001.jpg"
    assert prov["slot"] == "r0c1"
    assert "width_mm" not in prov
