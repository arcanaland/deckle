"""Unit tests for the line primitives. These need no scan and always run."""

from __future__ import annotations

import numpy as np
import pytest

from deckle.geometry import corners_from_lines, fit_line_tls, fit_line_trimmed, intersect


def _rect_lines(x0, y0, x1, y1, deg=0.0):
    """The four edge lines of a rectangle, optionally rotated about its centre."""
    t = np.radians(deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    corners = (corners - centre) @ rot.T + centre
    tl, tr, br, bl = corners
    return corners, {
        "top": fit_line_tls(np.array([tl, tr])),
        "right": fit_line_tls(np.array([tr, br])),
        "bottom": fit_line_tls(np.array([br, bl])),
        "left": fit_line_tls(np.array([bl, tl])),
    }


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


def test_corners_come_back_wound_tl_tr_br_bl():
    """The winding is a contract: `rectify` maps this quad onto the output in this order."""
    expected, lines = _rect_lines(10.0, 20.0, 110.0, 220.0)
    assert np.allclose(corners_from_lines(lines), expected)


def test_corners_survive_the_skew_hand_placement_produces():
    """RFC-001 measured up to 1.2deg of skew from placing cards by hand."""
    expected, lines = _rect_lines(10.0, 20.0, 110.0, 220.0, deg=1.2)
    assert np.allclose(corners_from_lines(lines), expected)


def test_corners_are_the_intersection_not_an_observed_point():
    """Lines fitted from stubs that stop well short of the corner still give the corner.

    This is the whole reason the primitive exists -- a card's real corner is rounded and
    there is nothing there to sample, so it is only ever recovered by intersecting.
    """
    _, lines = _rect_lines(0.0, 0.0, 100.0, 200.0)
    stubs = {
        "top": fit_line_tls(np.array([[40.0, 0.0], [60.0, 0.0]])),
        "right": fit_line_tls(np.array([[100.0, 80.0], [100.0, 120.0]])),
        "bottom": fit_line_tls(np.array([[40.0, 200.0], [60.0, 200.0]])),
        "left": fit_line_tls(np.array([[0.0, 80.0], [0.0, 120.0]])),
    }
    assert np.allclose(corners_from_lines(stubs), corners_from_lines(lines))


def test_a_degenerate_quad_raises_rather_than_returning_nonsense():
    """A side edge fitted parallel to a top edge has no corner, and saying so beats
    handing back an arbitrary point that later reads as a real measurement."""
    _, lines = _rect_lines(0.0, 0.0, 100.0, 200.0)
    lines = {**lines, "left": lines["top"]}
    with pytest.raises(ValueError):
        corners_from_lines(lines)


def test_a_missing_edge_is_not_silently_tolerated():
    _, lines = _rect_lines(0.0, 0.0, 100.0, 200.0)
    with pytest.raises(KeyError):
        corners_from_lines({k: v for k, v in lines.items() if k != "bottom"})
