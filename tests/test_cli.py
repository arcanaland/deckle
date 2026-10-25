from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import cv2
import pytest
from test_detect import CARD_MM, DPI, TWO, Placed, jig_scene

from deckle import build_id
from deckle.cli import _parser, _settings, main
from deckle.detect import CardSpec
from deckle.labels import COMMENT_XATTR
from deckle.project import PROJECT_FILE, Project, init
from deckle.staging import read_staging_index
from deckle.units import DEFAULT_DPI, DEFAULT_STRATEGY, EdgeStrategy

FLAGS = ["--dpi", str(DPI), "--card", f"{CARD_MM[0]}x{CARD_MM[1]}"]
FLAGS += ["--aspect", str(CARD_MM[0] / CARD_MM[1])]


@pytest.fixture(autouse=True)
def outside_any_project(tmp_path, monkeypatch) -> None:
    """Project discovery walks up from the working directory, so pin it."""
    monkeypatch.chdir(tmp_path)


def make_project(root: Path, **config) -> Project:
    """A project whose [project] describes the synthetic scene, so no flags are needed."""
    project = init(root, "Test Deck", deck_dir=str(root.parent / "deck"))
    project.doc["project"].update(
        {
            "card_width_mm": CARD_MM[0],
            "card_height_mm": CARD_MM[1],
            "aspect": CARD_MM[0] / CARD_MM[1],
            "dpi": DPI,
            **config,
        }
    )
    project.save()
    return project


@pytest.fixture(scope="module")
def scan(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("scans") / "two.png"
    cv2.imwrite(str(path), jig_scene(TWO))
    return path


@pytest.fixture(scope="module")
def empty_window(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("scans") / "one-empty.png"
    cv2.imwrite(str(path), jig_scene([[Placed(), None]]))
    return path


def test_detect_json_reports_every_card(scan, capsys) -> None:
    assert main(["detect", str(scan), *FLAGS, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["scan"] == str(scan)
    assert out["dpi"] == DPI
    assert [(c["row"], c["col"]) for c in out["cards"]] == [(0, 0), (0, 1)]
    for c in out["cards"]:
        assert c["width_mm"] == pytest.approx(CARD_MM[0], abs=0.02)
        assert c["height_mm"] == pytest.approx(CARD_MM[1], abs=0.02)


def test_detect_report_names_each_slot(scan, capsys) -> None:
    assert main(["detect", str(scan), *FLAGS]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{scan}: 2 card(s)")
    assert "r0c0" in out and "r0c1" in out


def test_rectify_writes_one_master_per_slot(scan, tmp_path, capsys) -> None:
    assert main(["rectify", str(scan), *FLAGS, "-o", str(tmp_path)]) == 0
    written = capsys.readouterr().out.split()
    assert written == [str(tmp_path / f"{scan.stem}_r0c{c}.png") for c in (0, 1)]
    for p in written:
        h, w = cv2.imread(p).shape[:2]
        assert h > w


def test_a_refusal_exits_nonzero_and_says_why(empty_window, tmp_path, capsys) -> None:
    out_dir = tmp_path / "masters"
    assert main(["rectify", str(empty_window), *FLAGS, "-o", str(out_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    head, *rest = captured.err.splitlines()
    assert head == f"deckle: {empty_window}: 4 problems in 1 of 2 windows"
    assert [ln.split(":")[0] for ln in rest] == ["  window 1 (r0c1)"] * 4
    assert not out_dir.exists()


def test_detect_lists_every_failing_window(tmp_path, capsys) -> None:
    path = tmp_path / "two-broken.png"
    scene = [[None, Placed()], [Placed(), Placed(bottom_extra_mm=0.4)]]
    cv2.imwrite(str(path), jig_scene(scene))
    assert main(["detect", str(path), *FLAGS, "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    head, *rest = captured.err.splitlines()
    assert head == f"deckle: {path}: {len(rest)} problems in 2 of 4 windows"
    assert sorted({ln.split(":")[0] for ln in rest}) == ["  window 0 (r0c0)", "  window 3 (r1c1)"]
    assert "  window 0 (r0c0): left: 0 of " in captured.err
    assert "not a rectangle" in captured.err


def test_a_wrong_window_count_is_refused(scan, capsys) -> None:
    assert main(["detect", str(scan), *FLAGS, "--num-windows", "4"]) == 1
    assert "expected 4 windows, found 2" in capsys.readouterr().err


def test_an_unreadable_scan_is_refused(tmp_path) -> None:
    bogus = tmp_path / "not-an-image.png"
    bogus.write_text("nope")
    with pytest.raises(SystemExit, match="cannot read image"):
        main(["detect", str(bogus)])


@pytest.mark.parametrize(
    "argv",
    [
        ["detect", "x.png", "--edge-strategy", "widest"],
        ["detect", "x.png", "--card", "70by120"],
    ],
    ids=["unknown-strategy", "malformed-card"],
)
def test_bad_arguments_are_rejected_before_reading_the_scan(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "cannot read image" not in capsys.readouterr().err


def test_outside_a_project_the_defaults_are_the_detectors_own() -> None:
    args = _parser().parse_args(["detect", "x.png"])
    assert _settings(args, None) == (CardSpec(), DEFAULT_DPI, DEFAULT_STRATEGY)


def test_rectify_outside_a_project_needs_out(scan, capsys) -> None:
    assert main(["rectify", str(scan), *FLAGS]) == 1
    assert "-o/--out is required outside a project" in capsys.readouterr().err


def test_version_is_the_build_id(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"deckle {build_id()}"


def test_detect_inside_a_project_reads_its_settings(scan, tmp_path, monkeypatch, capsys) -> None:
    # the default card size cannot find the scene's cards, so passing means [project] was read
    assert main(["detect", str(scan)]) == 1
    capsys.readouterr()

    project = make_project(tmp_path / "proj")
    monkeypatch.chdir(project.masters_dir)
    assert main(["detect", str(scan), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["cards"]) == 2


def test_a_flag_beats_the_project(tmp_path) -> None:
    project = make_project(tmp_path / "proj", edge_strategy="outermost", opposite_tol_mm=0.4)
    args = _parser().parse_args(["detect", "x.png", "--dpi", "600", "--card", "70x120"])
    spec, dpi, strategy = _settings(args, project)
    assert (spec.card_mm, dpi) == ((70.0, 120.0), 600.0)
    assert (spec.aspect, spec.opposite_tol_mm) == (CARD_MM[0] / CARD_MM[1], 0.4)
    assert strategy is EdgeStrategy.OUTERMOST


def test_a_degraded_project_falls_back_to_the_defaults(tmp_path) -> None:
    project = make_project(tmp_path / "proj")
    (project.root / PROJECT_FILE).write_text("not = [toml")
    args = _parser().parse_args(["detect", "x.png", "--project", str(project.root)])
    assert _settings(args, Project.load(project.root))[1] == DEFAULT_DPI


def test_rectify_inside_a_project_stages_and_indexes(scan, tmp_path, capsys) -> None:
    project = make_project(tmp_path / "proj")
    assert main(["rectify", str(scan), "--project", str(project.root)]) == 0
    written = [Path(p) for p in capsys.readouterr().out.split()]
    assert [p.parent for p in written] == [project.staging_dir] * 2

    index = read_staging_index(project)
    assert sorted(index) == [p.name for p in written]
    row = index[written[1].name]
    assert (row["source_scan"], row["slot"], row["dpi"]) == (scan.name, "r0c1", DPI)
    assert row["edge_strategy"] == str(DEFAULT_STRATEGY)


def test_rectify_elsewhere_from_a_project_does_not_index(scan, tmp_path, capsys) -> None:
    project = make_project(tmp_path / "proj")
    out = tmp_path / "elsewhere"
    assert main(["rectify", str(scan), "--project", str(project.root), "-o", str(out)]) == 0
    assert len(list(out.glob("*.png"))) == 2
    assert read_staging_index(project) == {}


def test_a_bad_strategy_is_reported_by_status_and_refused_by_detect(scan, tmp_path, capsys):
    project = make_project(tmp_path / "proj", edge_strategy="widest")
    assert main(["status", "--project", str(project.root)]) == 0
    assert "detect and rectify will refuse" in capsys.readouterr().out
    assert main(["detect", str(scan), "--project", str(project.root)]) == 1
    assert "edge_strategy = 'widest'" in capsys.readouterr().err


def test_init_then_init_again_is_refused(tmp_path, capsys) -> None:
    assert main(["init", "proj"]) == 0
    assert capsys.readouterr().out.startswith("initialised proj\n")
    assert Project.load(tmp_path / "proj").authored()["deck"]["name"] == "proj"
    assert main(["init", "proj"]) == 1
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", [["status"], ["emit"], ["assign", "x.png", "major_arcana.00"]])
def test_project_commands_outside_a_project_are_refused(cmd, capsys) -> None:
    assert main(cmd) == 1
    assert capsys.readouterr().err.startswith(f"deckle: no {PROJECT_FILE} here")


def test_scan_to_deck(scan, tmp_path, monkeypatch, capsys) -> None:
    project = make_project(tmp_path / "proj")
    monkeypatch.chdir(project.root)

    assert main(["rectify", str(scan)]) == 0
    first, second = capsys.readouterr().out.split()

    assert main(["assign", first, "major_arcana.00"]) == 0
    assert capsys.readouterr().out == "major_arcana.00 -> masters/major_arcana/00.png\n"
    assert main(["assign", second, "--card-back", "plain", "--rotate-180"]) == 0
    assert "(rotated 180)" in capsys.readouterr().out
    row = Project.load(project.root).cards["major_arcana.00"]
    assert (row["source_scan"], row["slot"]) == (scan.name, "r0c0")

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "1 of 78 roster cards present" in out
    assert "card backs (1): plain" in out

    assert main(["emit"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(f"{tmp_path / 'deck'}: 1 card(s), 1 back(s)")
    assert (tmp_path / "deck" / "deck.toml").is_file()


def test_an_assign_refusal_exits_nonzero(tmp_path, capsys) -> None:
    project = make_project(tmp_path / "proj")
    staged = project.staging_dir / "s_r0c0.png"
    staged.write_bytes(b"pixels")
    argv = ["assign", str(staged), "not_a_card", "--project", str(project.root)]
    assert main(argv) == 1
    assert capsys.readouterr().err.startswith("deckle: ")
    assert staged.is_file()


# --- assign from labels -----------------------------------------------------------------


def label(path: Path | str, text: str) -> None:
    try:
        os.setxattr(path, COMMENT_XATTR, text.encode())
    except OSError as exc:
        if exc.errno == errno.ENOTSUP:
            pytest.skip(f"{Path(path).parent} does not support user xattrs")
        raise


@pytest.fixture
def staged(scan, tmp_path, monkeypatch, capsys) -> tuple[Project, list[Path]]:
    """Two rectified, indexed files in a project's staging/, cwd at the project root."""
    project = make_project(tmp_path / "proj")
    monkeypatch.chdir(project.root)
    assert main(["rectify", str(scan)]) == 0
    files = [Path(p) for p in capsys.readouterr().out.split()]
    return project, files


def test_a_dry_run_prints_the_plan_and_moves_nothing(staged, capsys) -> None:
    project, (first, second) = staged
    label(first, "m0")
    label(second, "w7~")
    assert main(["assign"]) == 0
    out = capsys.readouterr().out
    assert out == (
        f"{first.name} -> major_arcana.00\n"
        f"{second.name} -> minor_arcana.wands.seven (rotated 180)\n"
        "staging: 2 files, 2 labelled, 0 unlabelled, 0 problem(s)\n"
    )
    assert first.is_file() and second.is_file()
    assert Project.load(project.root).cards == {}


def test_apply_with_a_problem_moves_nothing(staged, capsys) -> None:
    project, (first, second) = staged
    label(first, "m0")
    label(second, "m0~")
    assert main(["assign", "--apply"]) == 1
    captured = capsys.readouterr()
    assert "problem: major_arcana.00 is claimed by 2 files" in captured.out
    assert "nothing was assigned" in captured.err
    assert first.is_file() and second.is_file()
    assert not project.masters_dir.joinpath("major_arcana").exists()


def test_a_dry_run_with_problems_exits_1(staged, capsys) -> None:
    _, (first, _) = staged
    label(first, "M0")
    assert main(["assign"]) == 1
    assert "staging: 2 files, 1 labelled, 1 unlabelled, 1 problem(s)" in capsys.readouterr().out


def test_apply_assigns_with_full_provenance(staged, scan, capsys) -> None:
    project, (first, second) = staged
    index = read_staging_index(project)
    label(first, "m0")
    label(second, "major_arcana.00:alt~")
    assert main(["assign", "--apply"]) == 0
    out = capsys.readouterr().out
    assert "assigned 2 file(s)" in out
    assert not first.exists() and not second.exists()
    assert read_staging_index(project) == {}

    cards = Project.load(project.root).cards
    assert set(cards) == {"major_arcana.00", "major_arcana.00:alt"}
    bare, alt = cards["major_arcana.00"], cards["major_arcana.00:alt"]
    assert (bare["slot"], alt["slot"]) == ("r0c0", "r0c1")
    assert (bare["rotate_180"], alt["rotate_180"]) == (False, True)
    for row, name in ((bare, first.name), (alt, second.name)):
        assert row["source_scan"] == scan.name
        for key in ("width_mm", "height_mm", "aspect", "skew_deg", "edge_strategy", "dpi"):
            assert row[key] == index[name][key]
    assert alt["master"] == "masters/major_arcana/00.alt.png"
    assert (project.root / alt["master"]).is_file()


def test_apply_force_replaces(staged, capsys) -> None:
    project, (first, second) = staged
    assert main(["assign", str(first), "major_arcana.00"]) == 0
    label(second, "m0")
    capsys.readouterr()

    assert main(["assign", "--apply"]) == 1
    assert "already has a master" in capsys.readouterr().out
    assert second.is_file()

    assert main(["assign", "--apply", "--force"]) == 0
    assert "(replaces masters/major_arcana/00.png)" in capsys.readouterr().out
    assert not second.exists()
    assert Project.load(project.root).cards["major_arcana.00"]["slot"] == "r0c1"


def test_label_mode_refuses_a_degraded_project(staged, capsys) -> None:
    project, (first, _) = staged
    label(first, "m0")
    (project.root / PROJECT_FILE).unlink()
    assert main(["assign", "--apply"]) == 1
    assert capsys.readouterr().err.startswith("deckle: cannot assign: ")
    assert first.is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["assign", "--variant", "alt"],
        ["assign", "--card-back", "plain"],
        ["assign", "--rotate-180"],
        ["assign", "--apply", "--rotate-180"],
        ["assign", "x.png", "major_arcana.00", "--apply"],
        ["assign", "x.png", "major_arcana.00", "--force"],
        ["assign", "x.png", "--card-back", "plain", "--apply"],
    ],
)
def test_flags_from_the_other_mode_are_usage_errors(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "usage: deckle assign" in capsys.readouterr().err


# --- debug bundles ----------------------------------------------------------------------


def test_a_failure_in_a_project_writes_a_bundle_and_ends_with_its_path(
    empty_window, tmp_path, capsys
) -> None:
    project = make_project(tmp_path / "proj")
    assert main(["detect", str(empty_window), "--project", str(project.root)]) == 1
    err = capsys.readouterr().err.splitlines()
    overview = project.debug_dir / empty_window.stem / "overview.jpg"
    assert err[0] == f"deckle: {empty_window}: 4 problems in 1 of 2 windows"
    assert err[-1] == f"deckle: debug overview: {overview}"
    assert cv2.imread(str(overview)) is not None
    assert (overview.parent / "trace.json").is_file()


def test_a_success_in_a_project_writes_nothing(scan, tmp_path, capsys) -> None:
    project = make_project(tmp_path / "proj")
    assert main(["detect", str(scan), "--project", str(project.root)]) == 0
    assert capsys.readouterr().err == ""
    assert not project.debug_dir.exists()


def test_a_success_does_not_touch_an_earlier_bundle(scan, tmp_path, capsys) -> None:
    project = make_project(tmp_path / "proj")
    old = project.debug_dir / scan.stem / "overview.jpg"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"from a failed run")
    assert main(["detect", str(scan), "--project", str(project.root)]) == 0
    assert old.read_bytes() == b"from a failed run"


@pytest.mark.parametrize("extra", [[], ["--json"]], ids=["text", "json"])
def test_debug_on_a_success_leaves_stdout_alone(scan, tmp_path, capsys, extra) -> None:
    project = make_project(tmp_path / "proj")
    argv = ["detect", str(scan), "--project", str(project.root), *extra]
    assert main(argv) == 0
    plain = capsys.readouterr()

    assert main([*argv, "--debug"]) == 0
    debugged = capsys.readouterr()
    overview = project.debug_dir / scan.stem / "overview.jpg"
    assert debugged.out == plain.out
    assert debugged.err == f"deckle: debug overview: {overview}\n"
    assert overview.is_file()


def test_outside_a_project_nothing_is_written_without_the_flag(empty_window, tmp_path) -> None:
    before = sorted(tmp_path.rglob("*"))
    assert main(["detect", str(empty_window), *FLAGS]) == 1
    assert sorted(tmp_path.rglob("*")) == before


def test_debug_dir_works_outside_a_project_and_for_rectify(empty_window, tmp_path, capsys):
    out = tmp_path / "dbg"
    argv = ["rectify", str(empty_window), *FLAGS, "-o", str(tmp_path / "m"), "--debug", str(out)]
    assert main(argv) == 1
    assert (
        capsys.readouterr()
        .err.splitlines()[-1]
        .endswith(str(out / empty_window.stem / "overview.jpg"))
    )


def test_a_bare_debug_outside_a_project_needs_a_directory(scan, capsys) -> None:
    assert main(["detect", str(scan), *FLAGS, "--debug"]) == 1
    assert "--debug needs a directory outside a project" in capsys.readouterr().err


def test_a_broken_bundle_never_changes_the_verdict(
    scan, empty_window, tmp_path, monkeypatch, capsys
):
    def boom(*_a, **_k):
        raise RuntimeError("renderer bug")

    monkeypatch.setattr("deckle.cli.write_bundle", boom)
    project = make_project(tmp_path / "proj")
    at = ["--project", str(project.root)]

    assert main(["detect", str(empty_window), *at]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[0].startswith(f"deckle: {empty_window}: 4 problems")
    assert err[-1] == "deckle: could not write the debug bundle: renderer bug"

    assert main(["detect", str(scan), *at, "--debug", "--json"]) == 0
    captured = capsys.readouterr()
    assert len(json.loads(captured.out)["cards"]) == 2
    assert "renderer bug" in captured.err
