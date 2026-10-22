"""Detection against real scans, driven by a manifest kept with the scan library.

The manifest says what deckle must make of each scan, so this file names none. It is
`deckle-scans.toml` at the library root, or wherever DECKLE_SCAN_MANIFEST points:

    [[scan]]
    path = "some/deck/scan_001.png"   # relative to the library
    card_mm = [70.0, 120.0]
    aspect = 0.583
    opposite_tol_mm = 0.15            # optional, CardSpec's default otherwise
    strategy = "brightest"
    dpi = 600.0                       # optional
    num_windows = 4                   # optional window count
    refuses = "regex"                 # either this, matched against the error...

    [[scan.card]]                     # ...or one table per slot, in slot order
    slot = "r0c0"
    width_mm = 69.81
    height_mm = 119.65
    tol_mm = 0.02                     # optional, DEFAULT_TOL_MM otherwise
    calipers_mm = [69.8, 119.6]       # optional, a physical measurement
    calipers_tol_mm = 0.1             # optional, DEFAULT_CALIPERS_TOL_MM otherwise
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import cv2
import pytest
from conftest import scan_library

from deckle.detect import CardSpec, DetectionError, detect

pytestmark = pytest.mark.scans

MANIFEST_NAME = "deckle-scans.toml"
# Under a pixel at 600dpi: loose enough for a subpixel refinement, tight enough that a
# detector landing on a different feature cannot pass.
DEFAULT_TOL_MM = 0.02
DEFAULT_CALIPERS_TOL_MM = 0.1


def manifest_path(library: Path) -> Path:
    raw = os.environ.get("DECKLE_SCAN_MANIFEST", "").strip()
    return Path(raw) if raw else library / MANIFEST_NAME


def _entries() -> list:
    library = scan_library()
    if library is None:
        return []  # conftest refuses the run before any of these execute
    path = manifest_path(library)
    if not path.is_file():
        return [pytest.param(None, id="no-manifest")]
    scans = tomllib.loads(path.read_text()).get("scan", [])
    if not scans:
        return [pytest.param(None, id="empty-manifest")]
    return [pytest.param(s, id=s["path"]) for s in scans]


def _spec(entry: dict) -> CardSpec:
    w, h = entry["card_mm"]
    kw = {"opposite_tol_mm": entry["opposite_tol_mm"]} if "opposite_tol_mm" in entry else {}
    return CardSpec(width_mm=w, height_mm=h, aspect=entry["aspect"], **kw)


@pytest.mark.parametrize("entry", _entries())
def test_scan(scans: Path, entry: dict | None) -> None:
    assert entry is not None, (
        f"no scans listed in {manifest_path(scans)} — a scan run that checks nothing must "
        f"not pass"
    )
    path = scans / entry["path"]
    assert path.is_file(), f"{path} is listed in the manifest but missing"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"{path} did not decode"

    def run():
        return detect(
            img,
            _spec(entry),
            dpi=entry.get("dpi", 600.0),
            num_windows=entry.get("num_windows"),
            strategy=entry["strategy"],
        )

    if "refuses" in entry:
        with pytest.raises(DetectionError) as exc:
            run()
        assert re.search(entry["refuses"], str(exc.value)), str(exc.value)
        return

    cards = run()
    want = entry.get("card", [])
    assert [f"r{c.window.row}c{c.window.col}" for c in cards] == [w["slot"] for w in want]
    for c, w in zip(cards, want, strict=True):
        tol = w.get("tol_mm", DEFAULT_TOL_MM)
        assert c.width_mm == pytest.approx(w["width_mm"], abs=tol), w["slot"]
        assert c.height_mm == pytest.approx(w["height_mm"], abs=tol), w["slot"]
        if "calipers_mm" in w:
            ctol = w.get("calipers_tol_mm", DEFAULT_CALIPERS_TOL_MM)
            cw, ch = w["calipers_mm"]
            assert c.width_mm == pytest.approx(cw, abs=ctol), f"{w['slot']} vs calipers"
            assert c.height_mm == pytest.approx(ch, abs=ctol), f"{w['slot']} vs calipers"
