"""
Calibrate one case or a directory of cases with a trained policy (or the GA).

    python scripts/calibrate_cases.py --cases data/sample/synthetic_isoline \
        --model models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip --out results/calibration.csv

Outputs
-------
<out>.csv   one row per case: case_id, nine calibrated parameters, R2, NRMSE [%],
            peak pressure [Pa], peak temperature [K], final burned fraction,
            predicted CA50 [deg ATDC], success flags, parameters on a bound,
            envelope warnings, actor latency [ms]
<out>.json  the same content plus the metadata of the model used
--plot DIR  measured vs reconstructed pressure trace per case (PNG)

If the cases carry ``ground_truth_params`` (synthetic data) the absolute
parameter errors are added as ``err_<name>`` columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.calibrate import PARAM_NAMES, PolicyCalibrator, success_flags  # noqa: E402
from src.utils.data_loader import load_cases  # noqa: E402


def run_ga(cases, pop, gens, seed):
    from src.baselines.ga import GeneticAlgorithm
    from src.environment.wiebe_env import RewardWeights, WiebeEnvV4, WiebeParams
    env = WiebeEnvV4(wiebe_type="double", cases=list(cases), data_dir="",
                     curriculum={"stages": [0], "schedule_steps": [0], "spans": [1.0], "noise_scales": [0.0]},
                     reward_weights=RewardWeights())
    out = []
    for i, case in enumerate(cases):
        ga = GeneticAlgorithm(wiebe_type="double", population_size=pop, n_generations=gens, seed=seed + i)
        t0 = time.perf_counter()
        best, _, _ = ga.optimize(case, verbose=False)
        dt = (time.perf_counter() - t0) * 1e3
        p = WiebeParams(best["theta0"], best["delta_theta"], best["a"], best["lambda_w"], best["m1"],
                        best["m2"], best["k"], best["ht_g"], best["ht_c"])
        res = env._simulate_episode(case, p)
        r = {"case_id": case["case_id"], "params": {n: float(best[n]) for n in PARAM_NAMES},
             "params_at_bound": [], "ga_time_ms": dt}
        if not res["failed"]:
            info = res["info"]
            r["metrics"] = {k: float(info[k]) for k in ("r2", "nrmse", "pmax", "tmax", "xb_final", "ca50_pred")}
            r.update(success_flags(info))
        out.append(r)
    return out


def flatten(r, case, warnings):
    row = {"case_id": r["case_id"]}
    row.update(r["params"])
    row.update(r.get("metrics", {}))
    for k in ("success", "all_constraints", "peak_pressure_ok", "peak_temperature_ok", "burn_ok"):
        if k in r:
            row[k] = r[k]
    row["params_at_bound"] = ";".join(r.get("params_at_bound", []))
    row["envelope_warnings"] = ";".join(warnings)
    for k in ("actor_time_ms", "ga_time_ms"):
        if k in r:
            row[k] = round(r[k], 4)
    gt = case.get("ground_truth_params")
    if gt:
        for n in PARAM_NAMES:
            row[f"err_{n}"] = r["params"][n] - float(gt[n])
    return row


def plot(cases, results, out_dir: Path, calibrator):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.environment.wiebe_env import WiebeParams
    out_dir.mkdir(parents=True, exist_ok=True)
    env = calibrator._make_env(cases) if calibrator else None
    from src.models.wiebe_functions import double_wiebe
    from src.utils.thermodynamics import (EngineGeometry, EngineOperatingConditions, GasProperties,
                                          simulate_pressure_trace)
    for case, r in zip(cases, results):
        p = r["params"]
        xb = np.clip(double_wiebe(case["crank_angle"], p["theta0"], p["delta_theta"], p["a"], p["m1"],
                                  p["m2"], p["lambda_w"], p["k"]), 0, 1)
        geo = EngineGeometry(case["bore"], case["stroke"], case["connecting_rod"], case["compression_ratio"])
        op = EngineOperatingConditions(case["speed"], case["lambda"], case["intake_pressure"],
                                       case["intake_temperature"])
        eta = float(np.clip(0.85 + 0.1 * p["lambda_w"], 0.85, 1.0))
        pp, _, _ = simulate_pressure_trace(case["crank_angle"], xb, geo, op, GasProperties(), eta=eta,
                                           ht_scale_global=p["ht_g"], ht_scale_combustion=p["ht_c"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(case["crank_angle"], case["pressure"] / 1e5, label="measured", lw=2)
        ax.plot(case["crank_angle"], pp / 1e5, "--", label="calibrated model")
        ax.set_xlim(-60, 90)
        ax.set_xlabel("Crank angle [deg ATDC]")
        ax.set_ylabel("Pressure [bar]")
        m = r.get("metrics", {})
        ax.set_title(f"{case['case_id']}  R2={m.get('r2', float('nan')):.3f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{case['case_id']}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, required=True, help="JSON file or directory of JSON files")
    ap.add_argument("--model", type=Path, default=Path("models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip"))
    ap.add_argument("--metadata", type=Path, default=None, help="sidecar JSON (default: <model>.json)")
    ap.add_argument("--method", choices=["policy", "ga"], default="policy")
    ap.add_argument("--normalization", choices=["frozen", "legacy"], default="frozen")
    ap.add_argument("--ga-pop", type=int, default=50)
    ap.add_argument("--ga-gens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results/calibration.csv"))
    ap.add_argument("--plot", type=Path, default=None, help="directory for per-case pressure plots")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    cal = None
    if args.method == "policy":
        cal = PolicyCalibrator.from_pretrained(args.model, args.metadata, normalization=args.normalization)
        results = cal.calibrate(cases)
        warns = [cal.check_envelope(c) for c in cases]
    else:
        results = run_ga(cases, args.ga_pop, args.ga_gens, args.seed)
        warns = [[] for _ in cases]

    rows = [flatten(r, c, w) for r, c, w in zip(results, cases, warns)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(args.out.with_suffix(".json"), "w") as f:
        json.dump({"method": args.method, "model": str(args.model) if cal else None,
                   "model_metadata": cal.metadata if cal else None, "results": results}, f, indent=2)
    if args.plot:
        plot(cases, results, args.plot, cal)

    n_ok = sum(bool(r.get("success")) for r in results)
    r2 = [r["metrics"]["r2"] for r in results if "r2" in r.get("metrics", {})]
    print(f"Calibrated {len(results)} case(s) with {args.method}: success {n_ok}/{len(results)}, "
          f"mean R2 = {np.mean(r2):.4f}")
    n_warn = sum(bool(w) for w in warns)
    if n_warn:
        print(f"WARNING: {n_warn} case(s) outside the training envelope of the policy; see 'envelope_warnings'.")
    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
