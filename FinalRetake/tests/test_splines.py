import numpy as np
from navigation.splines import CatmullRom2D


def test_spline_hits_endpoints():
    pts = np.array([[0, 0], [1, 2], [3, 3], [6, 2], [8, 0]], dtype=float)
    sp = CatmullRom2D(pts)
    p0 = sp.eval(0.0)
    pN = sp.eval(sp.n - 1.0)
    assert np.linalg.norm(p0 - pts[0]) < 1e-9
    assert np.linalg.norm(pN - pts[-1]) < 1e-9


def test_arclength_table_monotone_and_positive_length():
    pts = np.array([[0, 0], [1, 2], [3, 3], [6, 2], [8, 0]], dtype=float)
    sp = CatmullRom2D(pts)
    L = sp.build_arclength_table(M=1500)
    assert L > 0.0
    assert np.all(np.diff(sp._s_tab) >= -1e-12)


def test_equal_arclength_samples_are_roughly_equal_spaced():
    pts = np.array([[0, 0], [2, 4], [6, 5], [10, 2], [14, 0]], dtype=float)
    sp = CatmullRom2D(pts)
    sp.build_arclength_table(M=3000)
    eq = sp.sample_by_arclength(60)
    d = np.linalg.norm(np.diff(eq, axis=0), axis=1)
    # allow some approximation error, but should not vary wildly
    cv = float(np.std(d) / (np.mean(d) + 1e-12))
    assert cv < 0.15

def test_catmull_rom_is_not_straight_for_noncollinear_points():
    # this configuration produces curvature on segment between P1 and P2
    pts = np.array([[0, 0], [1, 0], [2, 1], [3, 0]], dtype=float)
    sp = CatmullRom2D(pts, alpha=0.5)

    p1 = sp.eval(1.2)
    p2 = sp.eval(1.5)
    p3 = sp.eval(1.8)

    # triangle area > 0 means not collinear => not a straight line
    area2 = abs(np.cross(p2 - p1, p3 - p1))  # double area
    assert area2 > 2e-3
