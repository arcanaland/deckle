"""Debug info for a failed scan"""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import build_id
from .detect import DetectionReport, WindowResult
from .edges import INWARD, EdgeAttempt, Reason
from .jig import frame_mask
from .units import mm_to_px

OVERVIEW = "overview.jpg"
TRACE = "trace.json"

# longest side of the overview
OVERVIEW_PX = 1600
JPEG_QUALITY = 90

# brightness of the scan under the annotations
DIM = 0.45

# about this many ticks along each edge
TICKS_PER_EDGE = 100

REASON_BGR = {
    Reason.HIT: (255, 255, 0),  # cyan
    Reason.OFF_IMAGE: (150, 150, 150),  # grey
    Reason.NO_ROOM: (60, 110, 170),  # brown
    Reason.NO_VOID: (255, 0, 255),  # magenta
    Reason.WEAK_STEP: (0, 140, 255),  # orange
    Reason.NO_STEP: (0, 0, 255),  # red
    Reason.VOID_AFTER: (0, 255, 255),  # yellow
    Reason.NOT_FLAT: (255, 120, 170),  # violet
}

FRAME_BGR = (200, 110, 30)
WINDOW_BGR = (255, 170, 70)
BAND_BGR = (210, 210, 210)
CARD_BGR = (255, 255, 255)
OK_BGR = REASON_BGR[Reason.HIT]
FAIL_BGR = REASON_BGR[Reason.NO_STEP]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
LINE_PX = 18
SWATCH_PX = 12
PAD_PX = 8

LEGEND_W = 760


def _round(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def _edge_dict(a: EdgeAttempt) -> dict:
    return {
        "n_scanlines": a.n_scanlines,
        "counts": {str(r): n for r, n in a.counts().items()},
        "yield": _round(a.yield_frac),
        "median_gap_mm": None if a.fit is None else _round(a.fit.median_gap_mm),
        "band_mm": _round(a.band_mm),
        "fitted": a.fit is not None,
    }


_CARD_KEYS = (
    "width_mm",
    "height_mm",
    "aspect",
    "skew_deg",
    "opposite_width_delta_mm",
    "opposite_height_delta_mm",
    "corners_px",
)


def _window_dict(w: WindowResult) -> dict:
    card = None if w.card is None else w.card.to_dict()
    return {
        "index": w.window.index,
        "row": w.window.row,
        "col": w.window.col,
        "opening_mm": [_round(w.window.opening_w_mm), _round(w.window.opening_h_mm)],
        "ok": not w.problems,
        "problems": list(w.problems),
        "card": None if card is None else {k: card[k] for k in _CARD_KEYS},
        "edges": {e: _edge_dict(a) for e, a in w.edges.items()},
    }


def trace_dict(report: DetectionReport, scan: Path) -> dict:
    """The report as plain JSON."""
    return {
        "deckle_version": build_id(),
        "scan": str(scan),
        "settings": {
            "card": asdict(report.spec),
            "dpi": report.dpi,
            "num_windows": report.num_windows,
            "strategy": str(report.strategy),
            "anchor_offset_mm": report.anchor_offset_mm,
            "frame_color": asdict(report.color),
        },
        "jig_error": report.jig_error,
        "problems": list(report.problems),
        "windows": [_window_dict(w) for w in report.windows],
    }


# --- drawing ------------------------------------------------------------------------------

_SHIFT = 4  # fractional bits for subpixel vertices


def _fixed(pts: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(pts) * (1 << _SHIFT)).astype(np.int32).reshape(-1, 1, 2)


def _pt(q: np.ndarray) -> tuple[int, int]:
    return int(q[0, 0]), int(q[0, 1])


def _text(img: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255)) -> None:
    """Text with a dark outline."""
    cv2.putText(img, text, org, FONT, FONT_SCALE, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, FONT_SCALE, color, 1, cv2.LINE_AA)


def _text_w(text: str) -> int:
    return cv2.getTextSize(text, FONT, FONT_SCALE, 1)[0][0]


def _at_depth(a: EdgeAttempt, idx: np.ndarray, depth_px: np.ndarray) -> np.ndarray:
    """Image points depth_px inward from each scanline's anchor, as (N, 2) x, y."""
    dx, dy = INWARD[a.edge]
    along = a.trace.scanline[idx].astype(np.float64)
    across = a.trace.anchor[idx].astype(np.float64) + depth_px * (dy if dy else dx)
    if a.edge in ("top", "bottom"):
        return np.column_stack([along, across])
    return np.column_stack([across, along])


def _band_px(a: EdgeAttempt) -> tuple[float, float]:
    """The searched depths, outer and inner, relative to the anchor."""
    return -mm_to_px(a.lead_mm, a.dpi), mm_to_px(a.band_mm, a.dpi)


def _draw_edge(img: np.ndarray, a: EdgeAttempt, s: float) -> None:
    n = len(a.trace)
    if n == 0:
        return
    outer, inner = _band_px(a)

    ends = np.array([0, n - 1])
    band = np.concatenate([_at_depth(a, ends, outer), _at_depth(a, ends[::-1], inner)])
    cv2.polylines(img, [_fixed(band * s)], True, BAND_BGR, 1, cv2.LINE_AA, _SHIFT)

    idx = np.arange(0, n, max(1, n // TICKS_PER_EDGE))
    p0, p1 = _at_depth(a, idx, outer) * s, _at_depth(a, idx, inner) * s
    for i, q0, q1 in zip(idx, _fixed(p0), _fixed(p1), strict=True):
        color = REASON_BGR[Reason(a.trace.reason[i])]
        cv2.line(img, _pt(q0), _pt(q1), color, 1, cv2.LINE_AA, _SHIFT)

    hits = idx[a.trace.hit[idx]]
    placed = _at_depth(a, hits, a.trace.dist_px[hits].astype(np.float64)) * s
    for q in _fixed(placed):
        cv2.circle(img, _pt(q), 1 << _SHIFT, CARD_BGR, -1, cv2.LINE_AA, _SHIFT)


def _draw_window(img: np.ndarray, w: WindowResult, s: float) -> None:
    corners = np.asarray(w.window.corners)
    cv2.polylines(img, [_fixed(corners * s)], True, WINDOW_BGR, 1, cv2.LINE_AA, _SHIFT)
    for a in w.edges.values():
        _draw_edge(img, a, s)
    if w.card is not None:
        quad = np.asarray(w.card.corners)
        cv2.polylines(img, [_fixed(quad * s)], True, CARD_BGR, 1, cv2.LINE_AA, _SHIFT)

    # inside the window, clear of the top and left bands
    depth = {e: _band_px(a)[1] - _band_px(a)[0] for e, a in w.edges.items()}
    x = int(corners[:, 0].min() * s + depth.get("left", 0.0) * s) + PAD_PX
    y = int(corners[:, 1].min() * s + depth.get("top", 0.0) * s) + PAD_PX + LINE_PX
    y_of = {e: f"{e} {100 * a.yield_frac:.0f}%" for e, a in w.edges.items()}
    ok = not w.problems
    rows = [
        (f"{w.window.label} {'OK' if ok else 'FAIL'}", OK_BGR if ok else FAIL_BGR),
        (f"opening {w.window.opening_w_mm:.1f} x {w.window.opening_h_mm:.1f}mm", CARD_BGR),
        ("  ".join(y_of[e] for e in ("top", "bottom") if e in y_of), CARD_BGR),
        ("  ".join(y_of[e] for e in ("left", "right") if e in y_of), CARD_BGR),
    ]
    for i, (text, color) in enumerate(rows):
        _text(img, text, (x, y + i * LINE_PX), color)


# a legend row is a list of (swatch color or None, text)
Row = list[tuple[tuple[int, int, int] | None, str]]


def _flow(items: Row) -> list[Row]:
    """Wrap swatch items into rows no wider than the legend."""
    rows: list[Row] = [[]]
    x = 0
    for color, text in items:
        w = SWATCH_PX + 4 + _text_w(text) + 2 * PAD_PX
        if rows[-1] and x + w > LEGEND_W - 2 * PAD_PX:
            rows.append([])
            x = 0
        rows[-1].append((color, text))
        x += w
    return rows


def _legend(report: DetectionReport, s: float, step_note: str) -> list[Row]:
    if report.jig_error is not None:
        verdict = ["FAIL at the jig stage, so no window was searched"]
        verdict += report.jig_error.splitlines()
    elif report.problems:
        verdict = ["FAIL: " + report.format().splitlines()[0]]
    else:
        verdict = [f"OK: {len(report.cards)} card(s)"]
    wrap = max(20, (LEGEND_W - 2 * PAD_PX) // _text_w("n"))
    rows: list[Row] = [[(None, ln)] for v in verdict for ln in textwrap.wrap(v, wrap) or [""]]

    rows += _flow([(REASON_BGR[r], str(r)) for r in Reason])
    rows += _flow(
        [
            (FRAME_BGR, "frame"),
            (WINDOW_BGR, "window"),
            (BAND_BGR, "search band"),
            (CARD_BGR, "card outline, placed edge"),
        ]
    )
    rows += [
        [(None, f"ticks: {step_note}, colored by reason")],
        [(None, f"scan dimmed to {DIM:.0%}, scaled {s:.3f}x ({1 / s:.2f} scan px per px)")],
        [(None, f"strategy {report.strategy}, {report.dpi:g}dpi")],
    ]
    return rows


def _step_note(report: DetectionReport) -> str:
    steps = sorted(
        {max(1, len(a.trace) // TICKS_PER_EDGE) for w in report.windows for a in w.edges.values()}
    )
    if not steps:
        return "none"
    if len(steps) == 1:
        return f"1 in {steps[0]} scanlines"
    return f"1 in {steps[0]}-{steps[-1]} scanlines"


def _legend_h(n_rows: int) -> int:
    return n_rows * LINE_PX + 2 * PAD_PX


def _draw_legend(img: np.ndarray, rows: list[Row], top: int) -> None:
    for i, row in enumerate(rows):
        y = top + PAD_PX + (i + 1) * LINE_PX - 5
        x = PAD_PX
        for color, text in row:
            if color is not None:
                cv2.rectangle(img, (x, y - SWATCH_PX + 1), (x + SWATCH_PX, y), color, -1)
                x += SWATCH_PX + 4
            _text(img, text, (x, y))
            x += _text_w(text) + 2 * PAD_PX


def render_overview(report: DetectionReport, bgr: np.ndarray) -> np.ndarray:
    """The whole scan, downsampled and dimmed"""
    h, w = bgr.shape[:2]

    step_note = _step_note(report)
    legend_h = _legend_h(len(_legend(report, 1.0, step_note)))
    s = min(1.0, OVERVIEW_PX / w, (OVERVIEW_PX - legend_h) / h)
    size = (max(1, round(w * s)), max(1, round(h * s)))

    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA).astype(np.float32) * DIM
    frame = cv2.resize(frame_mask(bgr, report.color), size, interpolation=cv2.INTER_AREA) > 127
    small[frame] = 0.5 * small[frame] + 0.5 * np.array(FRAME_BGR, np.float32)

    canvas = np.zeros((size[1] + legend_h, max(size[0], LEGEND_W), 3), np.uint8)
    canvas[: size[1], : size[0]] = small.astype(np.uint8)
    view = canvas[: size[1], : size[0]]
    for wr in report.windows:
        _draw_window(view, wr, s)
    _draw_legend(canvas, _legend(report, s, step_note), size[1])
    return canvas


def write_bundle(report: DetectionReport, bgr: np.ndarray, scan: Path, root: Path) -> Path:
    """Write debug bundle to <root>/<scan>."""
    trace = json.dumps(trace_dict(report, scan.resolve()), indent=2, ensure_ascii=False) + "\n"
    ok, jpeg = cv2.imencode(
        ".jpg", render_overview(report, bgr), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("could not encode the overview")

    dest = root / scan.stem
    tmp = root / f".{scan.stem}.{os.getpid()}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        (tmp / TRACE).write_text(trace, encoding="utf-8")
        (tmp / OVERVIEW).write_bytes(jpeg.tobytes())
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest / OVERVIEW
