"""0-D single-zone thermodynamic solver (first law, explicit integration)."""
import numpy as np
import pytest

from src.models.wiebe_functions import double_wiebe
from src.utils.thermodynamics import (
    EngineGeometry,
    EngineOperatingConditions,
    GasProperties,
    calculate_normalized_rmse,
    calculate_r_squared,
    cylinder_volume,
    simulate_pressure_trace,
)

GEOM = EngineGeometry(bore=0.083, stroke=0.0923, connecting_rod=0.1527, compression_ratio=12.5)
THETA = np.round(np.arange(-180.0, 180.0 + 1e-9, 0.1), 1)


def _op(p_int=200e3):
    return EngineOperatingConditions(speed=2000.0, lambda_=1.3, intake_pressure=p_int, intake_temperature=318.0)


def _xb():
    return double_wiebe(THETA, 0.0, 40.0, 4.5, 2.0, 1.5, 0.6, 1.9)


def test_cylinder_volume_at_tdc_and_bdc():
    v = cylinder_volume(np.deg2rad(np.array([0.0, 180.0])),
                        (GEOM.bore, GEOM.stroke, GEOM.connecting_rod, GEOM.clearance_volume))
    assert v[0] == pytest.approx(GEOM.clearance_volume, rel=1e-9)
    assert v[1] == pytest.approx(GEOM.clearance_volume + GEOM.displacement, rel=1e-9)
    assert (v[1] / v[0]) == pytest.approx(GEOM.compression_ratio, rel=1e-9)


def test_motored_adiabatic_compression_is_isentropic():
    """No combustion, no wall heat transfer: p V^gamma must be conserved."""
    gas = GasProperties()
    p, t, _ = simulate_pressure_trace(THETA, np.zeros_like(THETA), GEOM, _op(), gas,
                                      include_heat_transfer=False)
    v = cylinder_volume(np.deg2rad(THETA), (GEOM.bore, GEOM.stroke, GEOM.connecting_rod, GEOM.clearance_volume))
    # p[0] is the prescribed IVC pressure (1.1 p_int); from the first step on
    # the state is p = m R T / V (see docs/MODEL.md, "initial state").
    pv = (p * v ** gas.gamma_u)[1:]
    # explicit first-order scheme on a 0.1 CAD grid: drift below 0.5 %
    assert np.max(np.abs(pv / pv[0] - 1.0)) < 5e-3
    # compression is symmetric about TDC
    assert THETA[np.argmax(p)] == pytest.approx(0.0, abs=0.2)


def test_discrete_energy_balance_closes_without_wall_heat_transfer():
    """U_end = U_0 + Q_chem - sum(p dV) for the adiabatic case (exact for the scheme)."""
    gas = GasProperties()
    p, t, q = simulate_pressure_trace(THETA, _xb(), GEOM, _op(), gas, eta=0.91,
                                      include_heat_transfer=False)
    v = cylinder_volume(np.deg2rad(THETA), (GEOM.bore, GEOM.stroke, GEOM.connecting_rod, GEOM.clearance_volume))
    op = _op()
    mass = 0.85 * op.intake_pressure * GEOM.displacement / (gas.R * op.intake_temperature)
    cv = gas.R / (gas.gamma_u - 1.0)
    u0 = mass * cv * t[0]
    work = np.sum(p[:-1] * np.diff(v))
    u_end = mass * cv * t[-1]
    assert u_end == pytest.approx(u0 + (q[-1] - q[0]) - work, rel=1e-9)


def test_released_heat_matches_fuel_energy():
    gas, op, eta = GasProperties(), _op(), 0.9
    _, _, q = simulate_pressure_trace(THETA, _xb(), GEOM, op, gas, eta=eta)
    mass = 0.85 * op.intake_pressure * GEOM.displacement / (gas.R * op.intake_temperature)
    q_fuel = eta * mass / (1.0 + op.stoich_AFR * op.lambda_) * op.fuel_LHV
    assert q[-1] == pytest.approx(q_fuel * _xb()[-1], rel=1e-9)


def test_wall_heat_transfer_lowers_peak_pressure_monotonically():
    gas = GasProperties()
    peaks = [simulate_pressure_trace(THETA, _xb(), GEOM, _op(), gas, ht_scale_global=s,
                                     ht_scale_combustion=s)[0].max() for s in (0.5, 1.0, 2.0, 3.0)]
    assert all(a > b for a, b in zip(peaks, peaks[1:]))
    p_adiabatic = simulate_pressure_trace(THETA, _xb(), GEOM, _op(), gas, include_heat_transfer=False)[0].max()
    assert p_adiabatic > peaks[0]


def test_combustion_raises_pressure_above_motored():
    gas = GasProperties()
    p_fired = simulate_pressure_trace(THETA, _xb(), GEOM, _op(), gas)[0]
    p_mot = simulate_pressure_trace(THETA, np.zeros_like(THETA), GEOM, _op(), gas)[0]
    assert p_fired.max() > 1.2 * p_mot.max()
    np.testing.assert_allclose(p_fired[THETA < -1.0], p_mot[THETA < -1.0], rtol=1e-12)


def test_pressure_scales_linearly_with_intake_pressure():
    gas = GasProperties()
    p1 = simulate_pressure_trace(THETA, _xb(), GEOM, _op(100e3), gas)[0]
    p2 = simulate_pressure_trace(THETA, _xb(), GEOM, _op(200e3), gas)[0]
    assert p2.max() / p1.max() == pytest.approx(2.0, rel=0.02)


def test_metrics():
    rng = np.random.default_rng(0)
    p = 1e5 + 5e6 * np.sin(np.linspace(0, np.pi, 500))
    assert calculate_r_squared(p, p) == pytest.approx(1.0)
    assert calculate_normalized_rmse(p, p) == pytest.approx(0.0)
    noisy = p + rng.normal(0, 0.01 * p.max(), p.size)
    assert calculate_normalized_rmse(noisy, p) == pytest.approx(1.0, rel=0.1)  # percent of peak


def test_state_is_ideal_gas_consistent_after_first_step():
    gas, op = GasProperties(), _op()
    p, t, _ = simulate_pressure_trace(THETA, _xb(), GEOM, op, gas)
    v = cylinder_volume(np.deg2rad(THETA), (GEOM.bore, GEOM.stroke, GEOM.connecting_rod, GEOM.clearance_volume))
    mass = 0.85 * op.intake_pressure * GEOM.displacement / (gas.R * op.intake_temperature)
    np.testing.assert_allclose(p[1:], mass * gas.R * t[1:] / v[1:], rtol=1e-12)
