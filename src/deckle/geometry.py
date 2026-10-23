"""Line fitting and linear algebra stuff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Line:
    """An infinite line as a point + direction (unit vector)."""

    point: np.ndarray  # (2,) a point on the line
    direction: np.ndarray  # (2,) unit vector (scaled)

    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """How far 2d points are from our line."""
        return (pts - self.point) @ self.normal()

    def angle_deg(self) -> float:
        """Direction angle in degrees, wrapped to (-90, 90]."""
        a = np.degrees(np.arctan2(self.direction[1], self.direction[0]))
        while a <= -90.0:
            a += 180.0
        while a > 90.0:
            a -= 180.0
        return float(a)


@dataclass(frozen=True)
class LineFit:
    line: Line
    inliers: np.ndarray  # (M,2) points that survived trimming
    residual_sd_px: float
    n_input: int
    inlier_mask: np.ndarray  # (n_input,) bool, which input points are the inliers

    @property
    def n_inliers(self) -> int:
        return len(self.inliers)


def fit_line_tls(pts: np.ndarray) -> Line:
    """Total-least-squares line through some 2d points."""
    if len(pts) < 2:
        raise ValueError("need at least 2 points to fit a line")

    centroid = pts.mean(axis=0)

    # find direction of fit
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)

    return Line(point=centroid, direction=vt[0] / np.linalg.norm(vt[0]))


def fit_line_trimmed(pts: np.ndarray, sigma: float = 2.5, iterations: int = 5) -> LineFit:
    """Fit the line but throw out outliers"""
    n_input = len(pts)
    keep = pts
    kept = np.ones(n_input, dtype=bool)
    line = fit_line_tls(keep)
    for _ in range(iterations):
        d = line.distance(keep)
        sd = float(np.std(d))

        # we gucci
        if sd <= 0.0:
            break

        # figure out which outliers to throw out
        mask = np.abs(d) <= sigma * sd

        # don't throw away too much now
        if mask.sum() < max(8, 0.25 * n_input) or mask.all():
            break

        keep = keep[mask]
        kept[np.flatnonzero(kept)[~mask]] = False
        line = fit_line_tls(keep)

    return LineFit(
        line=line,
        inliers=keep,
        residual_sd_px=float(np.std(line.distance(keep))),
        n_input=n_input,
        inlier_mask=kept,
    )


def intersect(a: Line, b: Line) -> np.ndarray:
    """Intersection point of two lines."""
    m = np.column_stack((a.direction, -b.direction))
    det = float(np.linalg.det(m))

    if abs(det) < 1e-9:
        raise ValueError("lines are parallel")

    t = np.linalg.solve(m, b.point - a.point)

    return a.point + t[0] * a.direction


@dataclass(frozen=True, slots=True, eq=False)
class Quad:
    """Four corners in image coordinates, clockwise from the top-left."""

    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray

    @property
    def top(self) -> float:
        """Length of the top side, in px."""
        return float(np.linalg.norm(self.top_right - self.top_left))

    @property
    def bottom(self) -> float:
        return float(np.linalg.norm(self.bottom_right - self.bottom_left))

    @property
    def left(self) -> float:
        return float(np.linalg.norm(self.bottom_left - self.top_left))

    @property
    def right(self) -> float:
        return float(np.linalg.norm(self.bottom_right - self.top_right))

    @property
    def width(self) -> float:
        """Mean of the top and bottom sides, in px."""
        return 0.5 * (self.top + self.bottom)

    @property
    def height(self) -> float:
        """Mean of the left and right sides, in px."""
        return 0.5 * (self.left + self.right)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """A (4, 2) array, clockwise from the top-left."""
        corners = (self.top_left, self.top_right, self.bottom_right, self.bottom_left)
        return np.array(corners, dtype=dtype)


def corners_from_lines(lines: Mapping[str, Line]) -> Quad:
    """Find the 4 corners of 4 intersecting lines (like the edges of a tarot card)"""
    return Quad(
        top_left=intersect(lines["top"], lines["left"]),
        top_right=intersect(lines["top"], lines["right"]),
        bottom_right=intersect(lines["bottom"], lines["right"]),
        bottom_left=intersect(lines["bottom"], lines["left"]),
    )
