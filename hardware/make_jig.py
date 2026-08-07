#!/usr/bin/env python3


"""Generate the deckle scanning jig as an STL.

The jig is a flat frame that sits on the scanner glass and surrounds four
cards in a 2x2 grid, giving the detector a chroma-key background to find card
edges against. It prints as two identical halves.

Every dimension below is in millimeters.
"""

from __future__ import annotations

import argparse
import struct

# --- Parameters ------------------------------------------------------------
# Card size
# Re-measure and pass --card for a deck that differs.
CARD_W, CARD_H = 70.0, 120.0

CLEARANCE = 1.5

THICKNESS = 3.0
MARGIN_X = 15.0     # Green border left and right of the outer windows.
MARGIN_Y = 6.0      # Green border at the outer (non-seam) edge.
WEB_X = 24.0        # Green between the two columns.
SEAM_WEB = 7.0      # Green from window to the butt seam. Two halves -> 14mm.

NOTCH_DEPTH = 8.0   # Finger access to lift card
NOTCH_WIDTH = 16.0


def build(card_w: float, card_h: float) -> tuple[list[tuple], float, float]:
    """Return (boxes, half_w, half_h) for ONE half of the jig."""
    win_w = card_w + 2 * CLEARANCE
    win_h = card_h + 2 * CLEARANCE

    half_w = 2 * MARGIN_X + 2 * win_w + WEB_X
    half_h = MARGIN_Y + win_h + SEAM_WEB

    # Windows span y in [MARGIN_Y, MARGIN_Y + win_h]; the seam is at y = half_h.
    y0, y1 = MARGIN_Y, MARGIN_Y + win_h
    left_x0, left_x1 = MARGIN_X, MARGIN_X + win_w
    right_x0 = left_x1 + WEB_X
    right_x1 = right_x0 + win_w

    # Finger notches face each other across the center web in the middle of the
    # windows' long edges. What remains between them is a bridge of
    # WEB_X - 2*NOTCH_DEPTH, so keep that positive.
    ny0 = y0 + (win_h - NOTCH_WIDTH) / 2
    ny1 = ny0 + NOTCH_WIDTH
    bridge = WEB_X - 2 * NOTCH_DEPTH
    if bridge <= 2.0:
        raise SystemExit(
            f"center web bridge would be {bridge:.1f}mm; widen WEB_X or "
            f"shrink NOTCH_DEPTH")

    boxes = [
        (0.0, 0.0, half_w, y0),                     # outer margin bar
        (0.0, y1, half_w, half_h),                  # seam bar
        (0.0, y0, left_x0, y1),                     # left margin
        (right_x1, y0, half_w, y1),                 # right margin
        # Center web, split into three so the two notches are left as voids.
        (left_x1, y0, right_x0, ny0),
        (left_x1, ny1, right_x0, y1),
        (left_x1 + NOTCH_DEPTH, ny0, right_x0 - NOTCH_DEPTH, ny1),
    ]
    return boxes, half_w, half_h


def box_triangles(x0, y0, x1, y1, z0=0.0, z1=THICKNESS):
    """12 triangles for an axis-aligned box, outward-facing."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]

    # I'mma keep it a buck, only Claude knows what these values mean
    faces = [
        (0, 2, 1), (0, 3, 2),   # bottom
        (4, 5, 6), (4, 6, 7),   # top
        (0, 1, 5), (0, 5, 4),   # front
        (1, 2, 6), (1, 6, 5),   # right
        (2, 3, 7), (2, 7, 6),   # back
        (3, 0, 4), (3, 4, 7),   # left
    ]
    for a, b, c in faces:
        yield v[a], v[b], v[c]


def normal(a, b, c):
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    w = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (u[1] * w[2] - u[2] * w[1],
         u[2] * w[0] - u[0] * w[2],
         u[0] * w[1] - u[1] * w[0])
    m = sum(t * t for t in n) ** 0.5
    return tuple(t / m for t in n) if m else (0.0, 0.0, 0.0)


def write_stl(path: str, boxes) -> int:
    tris = [t for b in boxes for t in box_triangles(*b)]
    with open(path, "wb") as f:
        f.write(b"deckle scanning jig - one half, print twice".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            f.write(struct.pack("<3f", *normal(a, b, c)))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    return len(tris)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default=f"{CARD_W}x{CARD_H}",
                    help="card size WxH in mm (default %(default)s)")
    ap.add_argument("-o", "--out", default="deckle-jig-half.stl")
    args = ap.parse_args()

    card_w, card_h = (float(v) for v in args.card.lower().split("x"))
    boxes, half_w, half_h = build(card_w, card_h)
    n = write_stl(args.out, boxes)

    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
    print(f"{args.out}: {n} triangles")
    print(f"card         {card_w} x {card_h} mm")
    print(f"window       {card_w + 2*CLEARANCE} x {card_h + 2*CLEARANCE} mm "
          f"({CLEARANCE}mm clearance per side)")
    print(f"half         {half_w} x {half_h} x {THICKNESS} mm")
    print(f"assembled    {half_w} x {2*half_h} mm  (two halves, one rotated 180)")
    print(f"material     ~{area * THICKNESS / 1000:.1f} cm3 per half "
          f"(~{area * THICKNESS / 1000 * 1.24:.0f} g PLA)")


if __name__ == "__main__":
    main()
