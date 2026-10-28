"""
Generate a synthetic isoline of operating points with KNOWN double Wiebe and
heat-transfer parameters, for verifying the package without external data.

The traces are produced with the package's own forward model
(``src.models.wiebe_functions.double_wiebe`` and
``src.utils.thermodynamics.simulate_pressure_trace``), on the reference
engine geometry of the associated study (bore 83.0 mm, stroke 92.3 mm,
connecting rod 152.7 mm, compression ratio 12.5).

The ground-truth parameters vary smoothly with intake pressure, so the same
files can be used to (i) check that a calibration method reproduces the
pressure traces and (ii) exercise the compensation-effect (total-variation)
diagnostic on a trajectory whose true TV is known.

NOTE: these are SYNTHETIC cases. They do not replicate or approximate the
experimental dataset of Yuan et al., Energy 313 (2024) 134074, which is not
redistributed with this package.

Usage
-----
    python data/sample/generate_sample.py                  # default: 10 cases
    python data/sample/generate_sample.py --n-cases 20 --noise 0.002
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from src.models.wiebe_functions import double_wiebe  # noqa: E402
from src.utils.thermodynamics import (  # noqa: E402
    EngineGeometry,
    EngineOperatingConditions,
    GasProperties,
    simulate_pressure_trace,
)

PARAM_NAMES = ["theta0", "delta_theta", "a", "m1", "m2", "lambda_w", "k", "ht_g", "ht_c"]

GEOMETRY = {
    "bore_m": 0.083,
    "stroke_m": 0.0923,
    "connecting_rod_m": 0.1527,
    "compression_ratio": 12.5,
}


def ground_truth(s: float) -> dict:
    """Smooth ground-truth parameter trajectory, s in [0, 1] (load axis).

    All values lie inside the stage-1 action box (alpha = 0.5) of the
    environment, so both the bundled v0.x policy and v1.0.0 policies can in
    principle represent them.
    """
    return {
        "theta0": 0.0 + 4.0 * s,          # start of combustion [deg ATDC]
        "delta_theta": 40.0 + 6.0 * s,    # duration of first Wiebe [deg]
        "a": 4.5,                         # efficiency parameter [-]
        "m1": 2.0 - 0.2 * s,              # form factor, fast burn [-]
        "m2": 1.5 + 0.2 * s,              # form factor, slow burn [-]
        "lambda_w": 0.60 - 0.10 * s,      # weight of fast burn [-]
        "k": 1.9 + 0.2 * s,               # duration ratio [-]
        "ht_g": 1.50,                     # global Woschni scale [-]
        "ht_c": 1.40 + 0.20 * s,          # combustion-phase Woschni scale [-]
    }


def simulate_case(params: dict, speed_rpm: float, lam: float,
                  p_int_pa: float, t_int_k: float,
                  theta: np.ndarray) -> np.ndarray:
    geometry = EngineGeometry(GEOMETRY["bore_m"], GEOMETRY["stroke_m"],
                              GEOMETRY["connecting_rod_m"], GEOMETRY["compression_ratio"])
    operating = EngineOperatingConditions(speed=speed_rpm, lambda_=lam,
                                          intake_pressure=p_int_pa,
                                          intake_temperature=t_int_k)
    xb = double_wiebe(theta, params["theta0"], params["delta_theta"], params["a"],
                      params["m1"], params["m2"], params["lambda_w"], params["k"])
    xb = np.clip(xb, 0.0, 1.0)
    # same combustion-efficiency closure as the RL environment (double Wiebe)
    eta = float(np.clip(0.85 + 0.1 * params["lambda_w"], 0.85, 1.0))
    pressure, _, _ = simulate_pressure_trace(
        theta, xb, geometry, operating, GasProperties(), eta=eta,
        ht_scale_global=params["ht_g"], ht_scale_combustion=params["ht_c"])
    return pressure


def generate(out_dir: Path, n_cases: int = 10, speed_rpm: float = 2000.0,
             noise: float = 0.0, seed: int = 42) -> list[Path]:
    """Write ``n_cases`` JSON cases plus ``ground_truth.csv`` into ``out_dir``."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    theta = np.round(np.arange(-180.0, 180.0 + 1e-9, 0.1), 1)   # 3601 points, 0.1 CAD
    p_int = np.linspace(110e3, 290e3, n_cases)

    written, rows = [], []
    for i, pin in enumerate(p_int):
        s = (pin - p_int[0]) / (p_int[-1] - p_int[0]) if n_cases > 1 else 0.0
        params = ground_truth(s)
        lam, t_int = 1.30, 318.0
        pressure = simulate_case(params, speed_rpm, lam, pin, t_int, theta)
        if noise > 0:
            pressure = pressure + rng.normal(0.0, noise * pressure.max(), pressure.size)

        case_id = f"synthetic_{int(speed_rpm)}rpm_{i:02d}"
        case = {
            "logName": case_id,
            "note": ("SYNTHETIC case generated with the package forward model for "
                     "verification only; not from the dataset of Yuan et al. (2024)."),
            "engine_geometry": dict(GEOMETRY),
            "operating_conditions": {
                "speed_rpm": float(speed_rpm),
                "lambda": lam,
                "intake_pressure_Pa": float(pin),
                "intake_temperature_K": t_int,
                "Tw_wall_K": 420.0,
                "gamma": 1.35,
                "fuel_LHV_J_per_kg": 44.0e6,
                "T_ign_deg": float(params["theta0"]),
            },
            "pressure_trace": {
                "theta_deg": theta.tolist(),
                "pressure_Pa": pressure.tolist(),
            },
            "ground_truth_params": params,
        }
        path = out_dir / f"{case_id}.json"
        with open(path, "w") as f:
            json.dump(case, f)
        written.append(path)
        rows.append({"case_id": case_id, "intake_pressure_kPa": pin / 1e3, **params})

    with open(out_dir / "ground_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "intake_pressure_kPa"] + PARAM_NAMES)
        w.writeheader()
        w.writerows(rows)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "synthetic_isoline")
    ap.add_argument("--n-cases", type=int, default=10)
    ap.add_argument("--speed-rpm", type=float, default=2000.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian noise std as a fraction of peak pressure (e.g. 0.002).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    files = generate(args.out_dir, args.n_cases, args.speed_rpm, args.noise, args.seed)
    print(f"Wrote {len(files)} synthetic cases and ground_truth.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
