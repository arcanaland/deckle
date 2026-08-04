"""Line fitting and intersection in image coordinates.

The card's corners are never observed directly — they are rounded, and RFC-001 defines
the corner as the intersection of the four fitted edge lines (the card's *sharp* corner).
So the primitives here are: fit a line to many noisy points, and intersect two lines.

Fits are total-least-squares (perpendicular distance), not y-on-x: a card edge can be
near-vertical, and an ordinary least-squares fit blows up there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Line:
    """An infinite line as a point on it plus a unit direction."""

    point: np.ndarray  # (2,) a point on the line
    direction: np.ndarray  # (2,) unit vector

    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed perpendicular distance from each of pts (N,2) to the line."""
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
    """Total-least-squares line through pts (N,2)."""
    if len(pts) < 2:
        raise ValueError("need at least 2 points to fit a line")
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    return Line(point=centroid, direction=vt[0] / np.linalg.norm(vt[0]))


def fit_line_trimmed(pts: np.ndarray, sigma: float = 2.5, iterations: int = 5) -> LineFit:
    """Iteratively-trimmed TLS fit: fit, drop points beyond `sigma` sd, refit.

    This is the RANSAC-trim of RFC-001 in its cheap deterministic form. With hundreds of
    scanline points per edge and outliers that are individually rare, iterative trimming
    converges to the same answer as sampling consensus without the randomness — which
    matters because a detector that gives different answers on reruns cannot be regression
    tested.
    """
    n_input = len(pts)
    keep = pts
    line = fit_line_tls(keep)
    for _ in range(iterations):
        d = line.distance(keep)
        sd = float(np.std(d))
        if sd <= 0.0:
            break
        mask = np.abs(d) <= sigma * sd
        # Never trim away so much that the fit stops being supported.
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
    """Intersection point of two lines. Raises if they are near-parallel."""
    m = np.column_stack((a.direction, -b.direction))
    det = float(np.linalg.det(m))
    if abs(det) < 1e-9:
        raise ValueError("lines are parallel; no intersection")
    t = np.linalg.solve(m, b.point - a.point)
    return a.point + t[0] * a.direction
