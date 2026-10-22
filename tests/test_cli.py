"""`deckle detect` and `deckle rectify` end to end, on a synthetic scan written to disk."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from test_detect import CARD_MM, DPI, TWO, Placed, jig_scene

from deckle import build_id
from deckle.cli import _parser, main
from deckle.detect import CardSpec
from deckle.units import DEFAULT_DPI, DEFAULT_STRATEGY

FLAGS = ["--dpi", str(DPI), "--card", f"{CARD_MM[0]}x{CARD_MM[1]}"]
FLAGS += ["--aspect", str(CARD_MM[0] / CARD_MM[1])]


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
    assert captured.err.startswith(f"deckle: {empty_window}: window 1:")
    assert not out_dir.exists()


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
        ["rectify", "x.png"],
    ],
    ids=["unknown-strategy", "malformed-card", "rectify-without-out"],
)
def test_bad_arguments_are_rejected_before_reading_the_scan(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "cannot read image" not in capsys.readouterr().err


def test_defaults_are_the_detectors_own() -> None:
    args = _parser().parse_args(["detect", "x.png"])
    spec = CardSpec()
    assert args.card == spec.card_mm
    assert args.aspect == spec.aspect
    assert args.opposite_tol == spec.opposite_tol_mm
    assert args.dpi == DEFAULT_DPI
    assert args.edge_strategy is DEFAULT_STRATEGY


def test_version_is_the_build_id(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"deckle {build_id()}"
