"""
Generate a synthetic sample operating point for framework demonstration.

NOTE: The actual experimental dataset used in the paper is sourced from:
    Yuan H, et al. "Thermodynamics-based data-driven combustion modelling for
    modern spark-ignition engines." Energy 2024;313.
    https://doi.org/10.1016/j.energy.2024.134074

This script generates ONE synthetic GDI operating point for code demonstration
purposes only. It does NOT replicate or approximate the original dataset.
"""

import numpy as np
import json

def generate_synthetic_case(
    speed_rpm: float = 2000.0,
    lambda_: float = 1.0,
    intake_pressure_kPa: float = 80.0,
    intake_temp_K: float = 310.0,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic pressure trace using a Double Wiebe model
    with typical GDI parameters. For demonstration only.
    """
    rng = np.random.default_rng(seed)

    # Engine geometry (typical single-cylinder GDI research engine)
    bore = 0.086      # m
    stroke = 0.086    # m
    conn_rod = 0.145  # m
    cr = 10.5

    # Wiebe parameters (typical mid-load GDI)
    theta_0 = -5.0   # SOC deg ATDC
    delta_theta = 35.0
    a = 6.9
    m1 = 1.5
    m2 = 0.5
    lambda_w = 0.3

    theta = np.linspace(-180, 180, 721)

    # Double Wiebe MFB
    def wiebe(th, th0, dth, aa, mm):
        x = np.zeros_like(th)
        mask = th >= th0
        xn = (th[mask] - th0) / dth
        x[mask] = 1.0 - np.exp(-aa * xn ** (mm + 1))
        return np.clip(x, 0, 1)

    xb = lambda_w * wiebe(theta, theta_0, delta_theta * 0.3, a, m1) + \
         (1 - lambda_w) * wiebe(theta, theta_0, delta_theta, a, m2)
    dxb = np.gradient(xb, theta)

    # Simple 0D pressure (adiabatic approximation for demo)
    gamma = 1.35
    Vd = np.pi / 4 * bore**2 * stroke
    V0 = Vd / (cr - 1)

    def volume(th):
        th_r = np.radians(th)
        r = stroke / 2
        l = conn_rod
        s = r * (1 - np.cos(th_r)) + l - np.sqrt(l**2 - (r * np.sin(th_r))**2)
        return V0 + np.pi / 4 * bore**2 * s

    V = volume(theta)
    P = np.zeros_like(theta)
    P[0] = intake_pressure_kPa * 1e3

    Qfuel = 500.0  # J (synthetic)
    for i in range(1, len(theta)):
        dV = V[i] - V[i - 1]
        dQ = Qfuel * dxb[i] * (np.radians(theta[1] - theta[0]))
        P[i] = P[i - 1] - gamma * P[i - 1] / V[i - 1] * dV + (gamma - 1) / V[i - 1] * dQ

    # Add small noise for realism
    P += rng.normal(0, 0.005e5, len(P))
    P = np.clip(P, 0.5e5, None)

    case = {
        "logName": "synthetic_demo_case",
        "note": "SYNTHETIC — generated for framework demonstration only. "
                "Not from the experimental dataset of Yuan et al. (2024).",
        "engine_geometry": {
            "bore_m": bore,
            "stroke_m": stroke,
            "connecting_rod_m": conn_rod,
            "compression_ratio": cr,
        },
        "operating_conditions": {
            "speed_rpm": speed_rpm,
            "lambda": lambda_,
            "intake_pressure_Pa": intake_pressure_kPa * 1e3,
            "intake_temperature_K": intake_temp_K,
            "Tw_wall_K": 450.0,
            "gamma": gamma,
            "fuel_LHV_J_per_kg": 44e6,
            "T_ign_deg": theta_0,
        },
        "pressure_trace": {
            "theta_deg": theta.tolist(),
            "pressure_Pa": P.tolist(),
        },
        "ground_truth_wiebe_params": {
            "note": "Known parameters used to generate this synthetic trace.",
            "theta_0_deg": theta_0,
            "delta_theta_deg": delta_theta,
            "a": a,
            "m1": m1,
            "m2": m2,
            "lambda_w": lambda_w,
        },
    }
    return case


if __name__ == "__main__":
    case = generate_synthetic_case()
    out_path = "synthetic_demo.json"
    with open(out_path, "w") as f:
        json.dump(case, f, indent=2)
    print(f"Saved synthetic case to {out_path}")
    print(f"  Speed: {case['operating_conditions']['speed_rpm']} RPM")
    print(f"  Peak pressure: {max(case['pressure_trace']['pressure_Pa'])/1e5:.2f} bar")
