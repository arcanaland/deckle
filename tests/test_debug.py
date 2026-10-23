from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from test_detect import DPI, PLATEN_BGR, SPEC, TWO, Placed, jig_scene

from deckle import build_id
from deckle.debug import OVERVIEW, OVERVIEW_PX, TRACE, render_overview, trace_dict, write_bundle
from deckle.detect import detect_report
from deckle.edges import Reason

SCENES = {
    "jig": np.full((600, 600, 3), PLATEN_BGR, np.uint8),
    "edge": jig_scene([[Placed(), None]]),
    "gate": jig_scene([[Placed(bottom_extra_mm=0.4)]]),
    "pass": jig_scene(TWO),
}


def report_of(name: str):
    return detect_report(SCENES[name], SPEC, dpi=DPI)


# --- trace.json -------------------------------------------------------------------------


def test_the_trace_round_trips_and_its_counts_are_the_reports() -> None:
    report = report_of("edge")
    out = json.loads(json.dumps(trace_dict(report, Path("scan.png"))))
    assert out["deckle_version"] == build_id()
    assert out["problems"] == report.problems
    assert out["jig_error"] is None
    assert out["settings"]["card"]["width_mm"] == SPEC.width_mm

    for w, row in zip(report.windows, out["windows"], strict=True):
        assert row["ok"] == (not w.problems)
        for e, a in w.edges.items():
            edge = row["edges"][e]
            assert edge["counts"] == {str(r): n for r, n in a.counts().items()}
            assert sum(edge["counts"].values()) == edge["n_scanlines"] == a.n_scanlines
            assert edge["fitted"] == (a.fit is not None)

    empty = out["windows"][1]
    assert empty["card"] is None
    assert empty["edges"]["left"]["counts"][str(Reason.WEAK_STEP)] > 0
    assert empty["edges"]["left"]["median_gap_mm"] is None


def test_the_trace_records_a_jig_failure() -> None:
    out = trace_dict(report_of("jig"), Path("scan.png"))
    assert "no frame-colored pixels" in out["jig_error"]
    assert out["windows"] == []


def test_a_card_that_fails_a_gate_still_has_its_measurements() -> None:
    (window,) = trace_dict(report_of("gate"), Path("scan.png"))["windows"]
    assert not window["ok"]
    assert all(e["fitted"] for e in window["edges"].values())
    assert window["card"]["opposite_width_delta_mm"] > SPEC.opposite_tol_mm


# --- overview.jpg -----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENES))
def test_the_overview_renders(name) -> None:
    img = render_overview(report_of(name), SCENES[name])
    assert img.dtype == np.uint8 and img.ndim == 3
    assert max(img.shape[:2]) <= OVERVIEW_PX


def test_a_large_scan_is_downsampled_to_fit() -> None:
    big = cv2.resize(SCENES["pass"], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    report = detect_report(big, SPEC, dpi=4 * DPI)
    assert not report.problems
    assert max(render_overview(report, big).shape[:2]) <= OVERVIEW_PX


# --- the bundle -------------------------------------------------------------------------


def test_the_bundle_replaces_an_earlier_one(tmp_path) -> None:
    stale = tmp_path / "scan" / "strip.png"
    stale.parent.mkdir()
    stale.write_bytes(b"old")

    overview = write_bundle(report_of("edge"), SCENES["edge"], Path("in/scan.png"), tmp_path)
    assert overview == tmp_path / "scan" / OVERVIEW
    assert sorted(p.name for p in overview.parent.iterdir()) == [OVERVIEW, TRACE]
    assert max(cv2.imread(str(overview)).shape[:2]) <= OVERVIEW_PX
    trace = json.loads((overview.parent / TRACE).read_text())
    assert trace["scan"] == str(Path("in/scan.png").resolve())
    assert [p.name for p in tmp_path.iterdir()] == ["scan"]
