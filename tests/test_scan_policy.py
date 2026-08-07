from __future__ import annotations

import tomllib
from pathlib import Path

from conftest import scan_library

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_an_explicit_library_is_used_when_it_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("DECKLE_SCANS", str(tmp_path))
    assert scan_library() == tmp_path


def test_an_explicit_library_that_does_not_exist_is_no_library(monkeypatch, tmp_path):
    monkeypatch.setenv("DECKLE_SCANS", str(tmp_path / "nope"))
    assert scan_library() is None


def test_empty_means_absent_rather_than_default(monkeypatch):
    monkeypatch.setenv("DECKLE_SCANS", "  ")
    assert scan_library() is None


def test_the_scans_marker_is_registered():
    config = tomllib.loads(PYPROJECT.read_text())
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.split(":")[0] == "scans" for m in markers), markers
