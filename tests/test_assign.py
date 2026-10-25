from __future__ import annotations

import cv2
import numpy as np
import pytest

from deckle.assign import AssignError, assign, unassign
from deckle.ids import IdError
from deckle.labels import read_label
from deckle.project import PROJECT_FILE, Project, init
from deckle.staging import read_staging_index, write_staging_index


def make_project(tmp_path):
    return init(tmp_path / "proj", "Test Deck")


def stage(project, name: str):
    path = project.staging_dir / name
    path.write_bytes(b"pixels")
    return path


def stage_png(project, name: str) -> tuple[object, np.ndarray]:
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    path = project.staging_dir / name
    assert cv2.imwrite(str(path), img)
    return path, img


def test_assign_moves_and_records_provenance(tmp_path):
    project = make_project(tmp_path)
    src = stage(project, "scan_r0c0.png")
    write_staging_index(
        project,
        {
            src.name: {
                "source_scan": "scan.jpg",
                "slot": "r0c0",
                "width_mm": 70.2,
                "height_mm": 120.3,
                "skew_deg": -0.31,
                "edge_strategy": "brightest",
            }
        },
    )

    result = assign(project, src, "minor_arcana.pentacles.king")

    assert not src.exists()
    assert result.dest == project.masters_dir / "minor_arcana/pentacles/king.png"
    assert result.dest.is_file()

    row = Project.load(project.root).cards["minor_arcana.pentacles.king"]
    assert row["source_scan"] == "scan.jpg"
    assert row["slot"] == "r0c0"
    assert row["width_mm"] == 70.2
    assert row["skew_deg"] == -0.31
    assert row["rotate_180"] is False
    assert row["master"] == "masters/minor_arcana/pentacles/king.png"
    assert row["deckle_version"] and row["assigned_at"].endswith("Z")


def test_assign_without_an_index_reads_scan_and_slot_from_the_name(tmp_path):
    project = make_project(tmp_path)
    assign(project, stage(project, "scan_001_r0c1.png"), "minor_arcana.cups.king")
    row = project.cards["minor_arcana.cups.king"]
    assert row["source_scan"] == "scan_001.jpg"
    assert row["slot"] == "r0c1"
    assert "width_mm" not in row


def test_assign_consumes_the_staging_index_row(tmp_path):
    project = make_project(tmp_path)
    src = stage(project, "s_r0c0.png")
    write_staging_index(project, {src.name: {"source_scan": "s.jpg", "slot": "r0c0"}})
    assign(project, src, "major_arcana.03")
    assert read_staging_index(project) == {}


def test_reassigning_an_id_replaces_its_master(tmp_path):
    project = make_project(tmp_path)
    assign(project, stage(project, "a_r0c0.png"), "major_arcana.03")
    result = assign(project, stage(project, "b_r0c0.png"), "major_arcana.03")

    assert result.replaced is None
    assert [m.ref for m in project.masters()] == ["major_arcana.03"]
    assert Project.load(project.root).cards["major_arcana.03"]["source_scan"] == "b.jpg"


def test_reassigning_removes_a_previous_master_at_another_path(tmp_path):
    project = make_project(tmp_path)
    old = project.masters_dir / "major_arcana/old.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"pixels")
    project.record_card("major_arcana.03", {"master": "masters/major_arcana/old.png"})

    result = assign(project, stage(project, "s_r0c0.png"), "major_arcana.03")

    assert result.replaced == old
    assert not old.exists()
    assert result.dest.is_file()


def test_assign_rotates_180(tmp_path):
    project = make_project(tmp_path)
    src, img = stage_png(project, "s_r0c0.png")
    result = assign(project, src, "major_arcana.03", rotate_180=True)

    assert not src.exists()
    out = cv2.imread(str(result.dest), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(out, img[::-1, ::-1])
    assert project.cards["major_arcana.03"]["rotate_180"] is True


@pytest.mark.parametrize("rotate_180", [True, False])
def test_assigning_a_master_onto_itself_is_refused(tmp_path, rotate_180):
    project = make_project(tmp_path)
    src, img = stage_png(project, "s_r0c0.png")
    master = assign(project, src, "major_arcana.12").dest
    before = (project.root / PROJECT_FILE).read_bytes()

    with pytest.raises(AssignError, match="already the master"):
        assign(project, master, "major_arcana.12", rotate_180=rotate_180)

    assert np.array_equal(cv2.imread(str(master), cv2.IMREAD_UNCHANGED), img)
    assert (project.root / PROJECT_FILE).read_bytes() == before


def test_rejected_ids_leave_the_file_in_staging(tmp_path):
    project = make_project(tmp_path)
    for bad in ("major_arcana.6", "minor_arcana.cups.King", "major_arcana.2women"):
        src = stage(project, "s_r0c0.png")
        with pytest.raises(IdError):
            assign(project, src, bad)
        assert src.exists()


def test_assign_accepts_extended_and_custom_ids(tmp_path):
    project = make_project(tmp_path)
    ids = ["major_arcana.23", "major_arcana.happy_squirrel", "minor_arcana.stars.ace"]
    for i, cid in enumerate(ids):
        assign(project, stage(project, f"s_r0c{i}.png"), cid)
    assert {m.ref for m in project.masters()} == set(ids)


def test_assign_variant(tmp_path):
    project = make_project(tmp_path)
    assign(project, stage(project, "s_r0c0.png"), "major_arcana.06:two_women")
    assert (project.masters_dir / "major_arcana/06.two_women.png").is_file()


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


def test_a_degraded_project_refuses_to_assign(tmp_path):
    project = make_project(tmp_path)
    src = stage(project, "s_r0c0.png")
    (project.root / PROJECT_FILE).write_text("broken [", encoding="utf-8")
    with pytest.raises(AssignError, match="not valid TOML"):
        assign(Project.load(project.root), src, "major_arcana.03")
    assert src.exists()


# --- unassign ---------------------------------------------------------------------------


def label_of(path) -> str | None:
    try:
        return read_label(path)
    except OSError:
        return None


def assigned(project, name: str, ref: str, **kw):
    """Stage an indexed png and assign it; returns the staged pixels."""
    path, img = stage_png(project, name)
    scan, _, slot = name[: -len(".png")].rpartition("_")
    index = read_staging_index(project)
    index[name] = {"source_scan": f"{scan}.jpg", "slot": slot, "width_mm": 70.1, "dpi": 600.0}
    write_staging_index(project, index)
    assign(project, path, ref, **kw)
    return img


def test_unassign_restores_the_staged_file(tmp_path):
    project = make_project(tmp_path)
    img = assigned(project, "scan_r1c0.png", "minor_arcana.wands.seven", rotate_180=True)

    (u,) = unassign(project, ["w7"])

    staged = project.staging_dir / "scan_r1c0.png"
    assert u.restaged == staged and u.ref == "minor_arcana.wands.seven"
    assert not u.master.exists()
    # back in the orientation rectify wrote, carrying the label that reassigns it as it was
    assert np.array_equal(cv2.imread(str(staged), cv2.IMREAD_UNCHANGED), img)
    assert u.label == "w7~"
    if u.labelled:
        assert label_of(staged) == "w7~"
    assert read_staging_index(project)["scan_r1c0.png"] == {
        "source_scan": "scan.jpg",
        "slot": "r1c0",
        "width_mm": 70.1,
        "dpi": 600.0,
    }
    assert "minor_arcana.wands.seven" not in Project.load(project.root).cards


def test_unassign_then_assign_is_the_same_assignment(tmp_path):
    project = make_project(tmp_path)
    assigned(project, "scan_r0c1.png", "major_arcana.02:alt", rotate_180=True)
    before = dict(Project.load(project.root).cards["major_arcana.02:alt"])

    (u,) = unassign(project, ["major_arcana.02:alt"])
    assert u.label == "m2:alt~"
    assign(project, u.restaged, "major_arcana.02:alt", rotate_180=True)

    after = Project.load(project.root).cards["major_arcana.02:alt"]
    for key in ("source_scan", "slot", "width_mm", "dpi", "rotate_180", "master"):
        assert after[key] == before[key]


def test_unassign_a_card_back(tmp_path):
    project = make_project(tmp_path)
    project.doc["project"]["card_back_default"] = "plain"
    assigned(project, "scan_r0c0.png", None, card_back="plain")
    (u,) = unassign(project, ["back"])
    assert (u.ref, u.label) == ("card_backs.plain", "back:plain")
    assert not project.back_path("plain").exists()


def test_unassign_checks_every_ref_before_moving_any(tmp_path):
    project = make_project(tmp_path)
    assigned(project, "scan_r0c0.png", "minor_arcana.wands.seven")
    master = project.masters_dir / "minor_arcana/wands/seven.png"

    with pytest.raises(AssignError) as exc:
        unassign(project, ["w7", "w8", "M2", "w7~"])
    msg = str(exc.value)
    assert msg.startswith("nothing was unassigned")
    assert "minor_arcana.wands.eight has no record" in msg
    assert "expected m<number>" in msg
    assert "has no meaning here" in msg
    assert master.is_file()
    assert "minor_arcana.wands.seven" in Project.load(project.root).cards


def test_unassign_refuses_to_overwrite_a_staged_file(tmp_path):
    project = make_project(tmp_path)
    assigned(project, "scan_r0c0.png", "minor_arcana.wands.seven")
    stage(project, "scan_r0c0.png")
    with pytest.raises(AssignError, match="already in staging"):
        unassign(project, ["w7"])
    assert (project.masters_dir / "minor_arcana/wands/seven.png").is_file()


def test_unassign_refuses_a_record_whose_master_is_gone(tmp_path):
    project = make_project(tmp_path)
    assigned(project, "scan_r0c0.png", "minor_arcana.wands.seven")
    (project.masters_dir / "minor_arcana/wands/seven.png").unlink()
    with pytest.raises(AssignError, match="is missing"):
        unassign(project, ["w7"])


def test_thin_provenance_restages_under_the_card_name(tmp_path):
    project = make_project(tmp_path)
    src = stage(project, "hand-staged.png")
    assign(project, src, "major_arcana.02:alt")
    (u,) = unassign(project, ["m2:alt"])
    assert u.restaged.name == "major_arcana.02.alt.png"
    assert read_staging_index(project) == {}


def test_unassign_refuses_a_degraded_project(tmp_path):
    project = make_project(tmp_path)
    (project.root / PROJECT_FILE).unlink()
    with pytest.raises(AssignError, match="cannot unassign"):
        unassign(Project.load(project.root), ["w7"])
