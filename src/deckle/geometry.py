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
        line = fit_line_tls(keep)

    return LineFit(
        line=line,
        inliers=keep,
        residual_sd_px=float(np.std(line.distance(keep))),
        n_input=n_input,
    )


def intersect(a: Line, b: Line) -> np.ndarray:
    """Intersection point of two lines."""
    m = np.column_stack((a.direction, -b.direction))
    det = float(np.linalg.det(m))

    if abs(det) < 1e-9:
        raise ValueError("lines are parallel")

    t = np.linalg.solve(m, b.point - a.point)

    return a.point + t[0] * a.direction


def corners_from_lines(lines: Mapping[str, Line]) -> np.ndarray:
    """Find the 4 corners of 4 intersecting lines (like the edges of a tarot card)"""
    return np.array(
        [
            intersect(lines["top"], lines["left"]),
            intersect(lines["top"], lines["right"]),
            intersect(lines["bottom"], lines["right"]),
            intersect(lines["bottom"], lines["left"]),
        ]
    )
