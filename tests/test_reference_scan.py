"""Regression tests against the real reference scan.

These run against the scan *path* rather than a committed crop. A crop cannot stand in
here: detection starts by finding the green frame and the holes in it, so anything small
enough to be worth committing has already thrown away the thing under test. The reference
scan is `20260801204347_001.jpg` at the repo root — one printed jig half, two
white-bordered cards, the case that bare-lid segmentation failed on.

Ground truth is calipers, 2026-08-01: *The Empress* 70.36 x 120.32mm and *King of Honey
Pots* 70.33 x 120.14mm.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from deckle.detect import CardSpec, DetectionError, detect, edge_bands
from deckle.jig import find_windows
from deckle.rectify import master_name, rectify_card
from deckle.units import px_to_mm

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "20260801204347_001.jpg"
SCANS = Path("/mnt/truenas/home/media/tarot/working/scans")
# The jigless negative control. NFS, read-only source material.
CONTROL = SCANS / "pre-jig" / "20260801101236_001.jpg"
# Both halves printed and butted, four cards, a different deck. Not part of TASK-002's
# acceptance — it was scanned while this task was being written — but it is the only
# evidence that the assembled jig and its seam work at all.
ASSEMBLED = SCANS / "20260801222210_001.jpg"

CALIPERS = {0: (70.36, 120.32), 1: (70.33, 120.14)}

needs_reference = pytest.mark.skipif(
    not REFERENCE.exists(), reason=f"reference scan not present at {REFERENCE}"
)
needs_control = pytest.mark.skipif(
    not CONTROL.exists(), reason=f"negative control scan not present at {CONTROL}"
)
needs_assembled = pytest.mark.skipif(
    not ASSEMBLED.exists(), reason=f"assembled-jig scan not present at {ASSEMBLED}"
)


@pytest.fixture(scope="module")
def scan():
    return cv2.imread(str(REFERENCE), cv2.IMREAD_COLOR)


@pytest.fixture(scope="module")
def cards(scan):
    return detect(scan, CardSpec(), expect=2)


# --- Step 2: the jig's own geometry, before any card is touched ----------------------


@needs_reference
def test_finds_exactly_two_windows(scan):
    """One printed half. Nothing in the pipeline may assume the assembled jig's four."""
    windows = find_windows(scan)
    assert len(windows) == 2
    assert [(w.row, w.col) for w in windows] == [(0, 0), (0, 1)]


@needs_reference
def test_window_openings_are_near_nominal(scan):
    """~0.6% under the 73 x 123mm design, which is PLA shrink, not detection error.

    Note this deliberately does *not* assert the two openings agree to 0.05mm as
    TASK-002 anticipated: measured per-window they are 122.07 and 122.33mm. See the
    accompanying note in the PR — window 0's top wall is genuinely ragged over a tenth of
    its length, and the earlier figure came from a measurement contaminated by specks
    elsewhere in the frame.
    """
    for w in find_windows(scan):
        assert 72.0 <= w.opening_w_mm <= 73.5
        assert 121.5 <= w.opening_h_mm <= 123.0


# --- Step 3: every edge must be able to justify itself -------------------------------


@needs_reference
def test_every_edge_is_justified_not_merely_fitted(cards):
    """Gap in range, step above the floor, and nearly every scanline agreeing.

    This is the check that the known-bad fit of RFC-001 would have failed: it had a 6.8um
    residual on an edge that was physically absent.
    """
    spec = CardSpec()
    assert len(cards) == 2
    for c in cards:
        bands = edge_bands(c.window, spec)
        for name, e in c.edges.items():
            assert spec.gap_min_mm <= e.median_gap_mm <= bands[name], name
            assert e.median_step >= 12.0, name
            assert e.yield_frac >= 0.95, name


@needs_reference
def test_cards_are_rectangles(cards):
    for c in cards:
        assert abs(c.top_w_mm - c.bottom_w_mm) < 0.150
        assert abs(c.left_h_mm - c.right_h_mm) < 0.150


@needs_reference
def test_skew_is_small_but_real(cards):
    """The window walls constrain rotation to a tenth of hand placement's 0.5-1.2deg.

    Small, but deskew still cannot be skipped: 0.25deg over 120mm is half a millimetre.
    """
    for c in cards:
        assert 0.02 < abs(c.skew_deg) < 0.60


# --- Step 4: the numbers, against calipers -------------------------------------------


@needs_reference
def test_width_reproduces_across_the_two_cards(cards):
    """The control that must not regress. Bare-lid segmentation scattered heights across
    4.8mm; the jig holds width to tens of microns."""
    widths = [c.width_mm for c in cards]
    assert abs(widths[0] - widths[1]) < 0.100


@needs_reference
def test_dimensions_are_close_to_calipers(cards):
    """Where the detector actually lands: within 90um on width, 25um on card 0's height.

    The tolerance here records measured reality rather than the 60um target, which is NOT
    met on three of four dimensions. Deliberately kept tight enough to catch a regression.
    """
    for c in cards:
        cw, ch = CALIPERS[c.window.index]
        assert abs(c.width_mm - cw) < 0.100
        assert abs(c.height_mm - ch) < 0.200


@needs_reference
@pytest.mark.xfail(
    strict=True,
    reason="TASK-002's 60um gate: met on 1 of 4 dimensions. "
    "Widths run 83-89um under calipers on both cards, "
    "and card 1 images 175um taller than its caliper "
    "height. Two independent subpixel estimators agree "
    "to 45um, so this is a property of the scan or of "
    "the ground truth, not of the fit. Needs a caliper "
    "re-measure to resolve; see the PR notes.",
)
def test_task_002_sixty_micron_gate(cards):
    for c in cards:
        cw, ch = CALIPERS[c.window.index]
        assert abs(c.width_mm - cw) <= 0.060
        assert abs(c.height_mm - ch) <= 0.060


# --- The detector must not depend on where it was aimed ------------------------------


@needs_reference
def test_detection_is_independent_of_the_search_anchor(scan, cards):
    """Displacing the anchor outward must not move the answer.

    This is the whole point of taking the innermost sustained step rather than the
    strongest gradient. Anchoring to design coordinates is what produced RFC-001's
    -376um error, and it is the one failure this project has already paid for.
    """
    base = {c.window.index: (c.width_mm, c.height_mm) for c in cards}
    for offset in (-0.25, -0.50):
        moved = detect(scan, CardSpec(), anchor_offset_mm=offset)
        for c in moved:
            w, h = base[c.window.index]
            # 0.5um, i.e. a hundredth of a pixel: the only residual is the profile's
            # integer start offset, and it must stay that way.
            assert c.width_mm == pytest.approx(w, abs=5e-4)
            assert c.height_mm == pytest.approx(h, abs=5e-4)


@needs_reference
@pytest.mark.parametrize("offset", [0.5, 1.0, -2.0])
def test_a_badly_aimed_detector_fails_loudly(scan, offset):
    """Aimed off the card, it must refuse — never return a plausible-looking crop."""
    with pytest.raises(DetectionError):
        detect(scan, CardSpec(), anchor_offset_mm=offset)


@needs_reference
def test_wrong_window_count_is_a_hard_failure(scan):
    with pytest.raises(DetectionError):
        detect(scan, CardSpec(), expect=4)


# --- Step 5: the negative control ----------------------------------------------------


@needs_control
def test_jigless_scan_is_rejected():
    """Four real cards, hand-placed, no jig. A detector that finds cards here is broken."""
    img = cv2.imread(str(CONTROL), cv2.IMREAD_COLOR)
    with pytest.raises(DetectionError, match="no jig found"):
        detect(img, CardSpec())


# --- Step 6: masters -----------------------------------------------------------------


@needs_reference
def test_masters_are_portrait_and_match_the_measured_size(scan, cards, tmp_path):
    for c in cards:
        img = rectify_card(scan, c)
        h, w = img.shape[:2]
        assert h > w, "masters must be portrait"
        assert px_to_mm(w) == pytest.approx(c.width_mm, abs=px_to_mm(1.0))
        assert px_to_mm(h) == pytest.approx(c.height_mm, abs=px_to_mm(1.0))
        out = tmp_path / master_name(REFERENCE, c)
        assert cv2.imwrite(str(out), img)
        assert cv2.imread(str(out)).shape == img.shape


@needs_reference
def test_master_names_carry_slot_not_identity(cards):
    names = [master_name(REFERENCE, c) for c in cards]
    assert names == [
        "20260801204347_001_r0c0.png",
        "20260801204347_001_r0c1.png",
    ]


@needs_reference
def test_master_edges_are_card_not_jig(scan, cards):
    """A one-pixel border sampled inside each master must be card, never green wall.

    The corners are the sharp intersections of the fitted lines, so the rounded corner
    radius is excluded from the check — reproducing those is the alpha mask's job later.
    """
    for c in cards:
        img = rectify_card(scan, c)
        h, w = img.shape[:2]
        inset = int(0.08 * w)
        strips = [
            img[2, inset : w - inset],
            img[h - 3, inset : w - inset],
            img[inset : h - inset, 2],
            img[inset : h - inset, w - 3],
        ]
        for s in strips:
            hsv = cv2.cvtColor(s.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            green = np.mean(
                (hsv[:, 0] >= 35) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 60) & (hsv[:, 2] >= 40)
            )
            assert green < 0.02


# --- The assembled jig: both halves, four windows, the butt seam ---------------------


@needs_assembled
def test_assembled_jig_yields_four_windows():
    """Four windows in a 2x2 grid, spanning the seam between two printed halves.

    The openings agree far better here than on the single-half reference scan (92um
    across all four, against 260um between the reference's two), which is the control
    TASK-002 step 2 was really asking for.
    """
    img = cv2.imread(str(ASSEMBLED), cv2.IMREAD_COLOR)
    windows = find_windows(img)
    assert len(windows) == 4
    assert [(w.row, w.col) for w in windows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    heights = [w.opening_h_mm for w in windows]
    assert max(heights) - min(heights) < 0.150


@needs_assembled
def test_assembled_jig_cards_measure_seventy_by_one_twenty():
    """Raw geometry only — three of these four cards do not pass the rectangle gates.

    Deliberately calls detect_card rather than detect: the point is that the *sizes* are
    right to a few tens of microns even where the shape checks reject the card, which
    localises the open problem to one edge rather than to the measurement as a whole.
    """
    from deckle.detect import detect_card

    img = cv2.imread(str(ASSEMBLED), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    spec = CardSpec()
    for w in find_windows(img):
        c = detect_card(gray, w, spec)
        assert c.width_mm == pytest.approx(70.0, abs=0.10)
        # "the tower" (window 2) has an unresolved bottom edge; see the PR notes.
        if w.index != 2:
            assert c.height_mm == pytest.approx(120.0, abs=0.10)


@needs_assembled
def test_assembled_jig_is_rejected_while_one_edge_is_unresolved():
    """Documents current behaviour: it fails loudly rather than emitting a bad crop.

    If a later change makes this pass, that is good news and this test should be replaced
    by real assertions on all four cards — but it must not start passing silently.
    """
    img = cv2.imread(str(ASSEMBLED), cv2.IMREAD_COLOR)
    with pytest.raises(DetectionError):
        detect(img, CardSpec(), expect=4)
