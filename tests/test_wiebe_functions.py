"""Double Wiebe mass-fraction-burned model."""
import numpy as np
import pytest

from src.models.wiebe_functions import double_wiebe, single_wiebe
from src.utils.thermodynamics import compute_CA50

THETA = np.round(np.arange(-180.0, 180.0 + 1e-9, 0.1), 1)


def test_single_wiebe_zero_before_start_of_combustion():
    xb = single_wiebe(THETA, -10.0, 40.0, 5.0, 2.0)
    assert np.all(xb[THETA < -10.0] == 0.0)


@pytest.mark.parametrize("a,m", [(5.0, 2.0), (6.9, 1.0), (3.0, 3.5)])
def test_single_wiebe_closed_form_at_end_of_duration(a, m):
    th0, dth = -10.0, 40.0
    xb = single_wiebe(np.array([th0 + dth]), th0, dth, a, m)
    assert xb[0] == pytest.approx(1.0 - np.exp(-a), rel=1e-12)


def test_single_wiebe_monotone_and_bounded():
    xb = single_wiebe(THETA, -10.0, 40.0, 5.0, 2.0)
    assert np.all(np.diff(xb) >= -1e-15)
    assert xb.min() >= 0.0 and xb.max() <= 1.0


def test_double_wiebe_is_weighted_sum():
    th0, dth, a, m1, m2, lw, k = -8.0, 35.0, 5.0, 1.5, 1.0, 0.6, 2.0
    xb = double_wiebe(THETA, th0, dth, a, m1, m2, lw, k)
    ref = lw * single_wiebe(THETA, th0, dth, a, m1) + (1 - lw) * single_wiebe(THETA, th0, k * dth, a, m2)
    np.testing.assert_allclose(xb, np.clip(ref, 0, 1), atol=1e-14)


def test_double_wiebe_limits_of_weight():
    args = (THETA, -8.0, 35.0, 5.0, 1.5, 1.0)
    np.testing.assert_allclose(double_wiebe(*args, 1.0, 2.0), single_wiebe(THETA, -8.0, 35.0, 5.0, 1.5))
    np.testing.assert_allclose(double_wiebe(*args, 0.0, 2.0), single_wiebe(THETA, -8.0, 70.0, 5.0, 1.0))


def test_ca50_linear_interpolation():
    theta = np.linspace(0.0, 10.0, 11)
    xb = theta / 10.0
    assert compute_CA50(theta, xb) == pytest.approx(5.0)
    assert compute_CA50(theta + 0.25, xb) == pytest.approx(5.25)


def test_ca50_of_wiebe_matches_closed_form():
    th0, dth, a, m = -10.0, 40.0, 5.0, 2.0
    xb = single_wiebe(THETA, th0, dth, a, m)
    exact = th0 + dth * (np.log(2.0) / a) ** (1.0 / (m + 1.0))
    assert compute_CA50(THETA, xb) == pytest.approx(exact, abs=0.01)


def test_exact_degeneracy_of_a_delta_theta_and_k():
    """(a, dtheta, k) enter only via a/dtheta^(m1+1) and a/(k dtheta)^(m2+1): one free direction."""
    a, d, k, m1, m2 = 4.5, 40.0, 2.0, 2.0, 1.5
    c1, c2 = a / d ** (m1 + 1), a / (k * d) ** (m2 + 1)
    d2 = 30.0
    a2 = c1 * d2 ** (m1 + 1)
    k2 = (a2 / c2) ** (1 / (m2 + 1)) / d2
    x1 = double_wiebe(THETA, 0.0, d, a, m1, m2, 0.6, k)
    x2 = double_wiebe(THETA, 0.0, d2, a2, m1, m2, 0.6, k2)
    assert abs(a2 - a) > 2.0 and abs(d2 - d) == 10.0
    np.testing.assert_allclose(x1, x2, atol=1e-12)
