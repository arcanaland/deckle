"""Unit tests for the line primitives"""

from __future__ import annotations

import numpy as np
import pytest

from deckle.geometry import Quad, corners_from_lines, fit_line_tls, fit_line_trimmed, intersect


def _rect_lines(x0, y0, x1, y1, deg=0.0):
    """The four edge lines of a rectangle"""
    t = np.radians(deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    q = Quad(*((corners - center) @ rot.T + center))
    return q, {
        "top": fit_line_tls(np.array([q.top_left, q.top_right])),
        "right": fit_line_tls(np.array([q.top_right, q.bottom_right])),
        "bottom": fit_line_tls(np.array([q.bottom_right, q.bottom_left])),
        "left": fit_line_tls(np.array([q.bottom_left, q.top_left])),
    }


def test_fits_a_vertical_line():
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
    expected, lines = _rect_lines(10.0, 20.0, 110.0, 220.0)
    assert np.allclose(corners_from_lines(lines), expected)


def test_corners_survive_the_skew_hand_placement_produces():
    expected, lines = _rect_lines(10.0, 20.0, 110.0, 220.0, deg=1.2)
    assert np.allclose(corners_from_lines(lines), expected)


def test_corners_are_the_intersection_not_an_observed_point():
    """Lines fitted from stubs that stop short of the corner"""
    _, lines = _rect_lines(0.0, 0.0, 100.0, 200.0)
    stubs = {
        "top": fit_line_tls(np.array([[40.0, 0.0], [60.0, 0.0]])),
        "right": fit_line_tls(np.array([[100.0, 80.0], [100.0, 120.0]])),
        "bottom": fit_line_tls(np.array([[40.0, 200.0], [60.0, 200.0]])),
        "left": fit_line_tls(np.array([[0.0, 80.0], [0.0, 120.0]])),
    }
    assert np.allclose(corners_from_lines(stubs), corners_from_lines(lines))


def test_a_degenerate_quad_raises():
    _, lines = _rect_lines(0.0, 0.0, 100.0, 200.0)
    lines = {**lines, "left": lines["top"]}
    with pytest.raises(ValueError):
        corners_from_lines(lines)


def test_a_missing_edge_is_not_tolerated():
    _, lines = _rect_lines(0.0, 0.0, 100.0, 200.0)
    with pytest.raises(KeyError):
        corners_from_lines({k: v for k, v in lines.items() if k != "bottom"})


def test_a_quad_is_an_array_clockwise_from_the_top_left():
    q = Quad(*np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 9.0], [0.0, 9.0]]))
    a = np.asarray(q, dtype=np.float32)
    assert a.shape == (4, 2) and a.dtype == np.float32
    assert np.array_equal(a, [[0, 0], [4, 0], [4, 9], [0, 9]])


def test_a_quad_measures_its_sides():
    q = Quad(*np.array([[0.0, 0.0], [4.0, 0.0], [5.0, 9.0], [0.0, 10.0]]))
    assert (q.top, q.bottom) == (4.0, pytest.approx(np.hypot(5.0, 1.0)))
    assert (q.left, q.right) == (10.0, pytest.approx(np.hypot(1.0, 9.0)))
    assert q.width == pytest.approx(0.5 * (q.top + q.bottom))
    assert q.height == pytest.approx(0.5 * (q.left + q.right))


def test_the_inlier_mask_composes_every_trim():
    """Far outliers go in the first trim; near ones only once the far ones stop inflating
    the spread. The mask must mark both."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1000, 600)
    y = 50.0 + rng.normal(0, 0.3, x.size)
    far, near = np.arange(0, x.size, 37), np.arange(11, x.size, 41)
    y[far] += 40.0
    y[near] += 3.0
    pts = np.column_stack((x, y))
    fit = fit_line_trimmed(pts)
    assert fit.inlier_mask.shape == (fit.n_input,)
    assert np.array_equal(pts[fit.inlier_mask], fit.inliers)
    assert not fit.inlier_mask[far].any()
    assert not fit.inlier_mask[near].any()
    # the near outliers survive a single trim, so this mask really is a composition
    once = fit_line_trimmed(pts, iterations=1)
    assert once.inlier_mask[near].all()
