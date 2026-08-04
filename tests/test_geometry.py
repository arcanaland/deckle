"""Unit tests for the line primitives. These need no scan and always run."""

from __future__ import annotations

import numpy as np
import pytest

from deckle.geometry import fit_line_tls, fit_line_trimmed, intersect


def test_fits_a_vertical_line():
    """A y-on-x fit blows up here; a card edge really can be vertical."""
    pts = np.column_stack((np.full(50, 7.0), np.linspace(0, 100, 50)))
    line = fit_line_tls(pts)
    assert abs(abs(line.direction[1]) - 1.0) < 1e-9
    assert np.allclose(line.distance(pts), 0.0, atol=1e-9)


def test_recovers_a_known_slope():
    x = np.linspace(0, 1000, 500)
    pts = np.column_stack((x, 3.0 + 0.01 * x))
    assert fit_line_tls(pts).angle_deg() == pytest.approx(np.degrees(np.arctan(0.01)))


def test_trimming_rejects_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1000, 600)
    y = 50.0 + rng.normal(0, 0.3, x.size)
    y[::37] += 40.0  # a notch-like cluster well off the line
    fit = fit_line_trimmed(np.column_stack((x, y)))
    assert fit.n_inliers < fit.n_input
    assert fit.residual_sd_px < 1.0
    assert abs(fit.line.point[1] - 50.0) < 0.5


def test_intersection():
    a = fit_line_tls(np.array([[0.0, 0.0], [10.0, 0.0]]))
    b = fit_line_tls(np.array([[4.0, -5.0], [4.0, 5.0]]))
    assert np.allclose(intersect(a, b), [4.0, 0.0])


def test_parallel_lines_have_no_intersection():
    a = fit_line_tls(np.array([[0.0, 0.0], [10.0, 0.0]]))
    b = fit_line_tls(np.array([[0.0, 5.0], [10.0, 5.0]]))
    with pytest.raises(ValueError):
        intersect(a, b)
