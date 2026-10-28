"""
Inference-latency and memory profiler.

Every latency in the output is MEASURED on the machine that runs the script.
The only estimated quantity is the microcontroller latency, obtained by
scaling a measured mean with a CoreMark ratio; it is written to separate,
explicitly labelled columns and is never mixed with measured values.

Measured rows
-------------
actor-only      one deterministic forward pass of the policy network on a
                real observation (built from a case), batch size 1
observation     building the 93-dim observation from a pressure trace
full-pipeline   observation + actor + one 0-D verification simulation with
                the predicted parameters (what PolicyCalibrator.calibrate does)
xgboost         MultiOutputRegressor.predict on one feature vector
                (only with --xgb-model, i.e. a model trained on GA labels;
                no stand-in model is ever built)
ga              complete genetic-algorithm calibration of one case
                (only with --ga-cases > 0)

Usage
-----
    python scripts/profile_inference.py \\
        --model models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip \\
        --cases data/sample/synthetic_isoline \\
        --repetitions 3000 --ga-cases 3 --out results/profile.csv

Note on the v0.x profiler
-------------------------
The script used for Ref. [1] (realtime_feasibility_wiebe.py) fell back to a
synthetic GA timing distribution (normal(1900 ms, 220 ms), seed 99) when it
found no saved GA timing file, and timed a stand-in XGBoost model fitted on
random data when no model path was given. Both fallbacks are removed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.calibrate import PolicyCalibrator  # noqa: E402
from src.utils.data_loader import load_cases  # noqa: E402

#: CoreMark ratio between a desktop i7 (~35 000) and an Infineon AURIX TC397
#: core at 300 MHz (~5 000). Used ONLY for the clearly labelled estimate.
COREMARK_SCALE = 7.0
HIL_DEADLINE_MS = 10.0


def machine_info() -> Dict[str, str]:
    import torch
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": str(torch.get_num_threads()),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["processor"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def time_call(fn: Callable[[], object], n: int, warmup: int = 50) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        out[i] = (time.perf_counter() - t0) * 1e3
    return out


def row(name: str, t: np.ndarray, **extra) -> Dict[str, object]:
    mean = float(np.mean(t))
    r = {
        "component": name,
        "n": int(t.size),
        "mean_ms": round(mean, 4),
        "std_ms": round(float(np.std(t)), 4),
        "median_ms": round(float(np.median(t)), 4),
        "p99_ms": round(float(np.percentile(t, 99)), 4),
        "max_ms": round(float(np.max(t)), 4),
        "measured": "yes",
        "mcu_estimate_ms (CoreMark x%.0f, NOT measured)" % COREMARK_SCALE: round(mean * COREMARK_SCALE, 2),
        "mcu_estimate_within_10ms_deadline": "yes" if mean * COREMARK_SCALE < HIL_DEADLINE_MS else "no",
    }
    r.update(extra)
    return r


def count_parameters(model) -> Dict[str, int]:
    actor, critic = 0, 0
    for name, p in model.policy.named_parameters():
        if "value_net" in name:
            critic += p.numel()
        else:
            actor += p.numel()   # policy_net, action_net, log_std
    return {"actor": actor, "critic": critic, "total": actor + critic}


def profile(model_path: Path, cases_path: Path, repetitions: int,
            xgb_model: Optional[Path], ga_cases: int, ga_pop: int, ga_gens: int,
            threads: Optional[int]) -> Dict[str, object]:
    import torch
    if threads:
        torch.set_num_threads(threads)

    cal = PolicyCalibrator.from_pretrained(model_path)
    cases = load_cases(cases_path)
    env = cal._make_env(cases)
    obs, _ = env.reset(options={"case_idx": 0})
    model = cal.model

    rows: List[Dict[str, object]] = []

    # actor-only: identical to how the policy is called inside calibrate()
    t = time_call(lambda: model.predict(obs, deterministic=True), repetitions)
    rows.append(row("actor-only (SB3 predict)", t))

    # raw torch forward pass of the actor MLP (no SB3 preprocessing)
    with torch.no_grad():
        x = torch.as_tensor(obs[None, :])
        pol = model.policy

        def fwd():
            with torch.no_grad():
                return pol.action_net(pol.mlp_extractor.forward_actor(x))
        t = time_call(fwd, repetitions)
    rows.append(row("actor-only (raw torch forward)", t))

    # observation construction
    t = time_call(lambda: env._build_observation(cases[0]), repetitions)
    rows.append(row("observation construction", t))

    # full pipeline
    def full():
        o, _ = env.reset(options={"case_idx": 0})
        a, _ = model.predict(o, deterministic=True)
        env.step(a)
    t = time_call(full, max(repetitions // 10, 50), warmup=5)
    rows.append(row("full pipeline (obs + actor + 0-D verification)", t))

    # XGBoost: only a real model
    if xgb_model is not None:
        import joblib
        xgb = joblib.load(xgb_model)
        n_feat = int(getattr(xgb.estimators_[0], "n_features_in_", 6))
        feat = np.zeros((1, n_feat), dtype=np.float32)
        t = time_call(lambda: xgb.predict(feat), repetitions)
        rows.append(row("xgboost predict (trained on GA labels)", t))

    # GA: complete calibrations of real cases
    if ga_cases > 0:
        from src.baselines.ga import GeneticAlgorithm
        tg = []
        for i in range(min(ga_cases, len(cases))):
            ga = GeneticAlgorithm(wiebe_type="double", population_size=ga_pop,
                                  n_generations=ga_gens, seed=42 + i)
            t0 = time.perf_counter()
            ga.optimize(cases[i], verbose=False)
            tg.append((time.perf_counter() - t0) * 1e3)
        rows.append(row(f"genetic algorithm (pop={ga_pop}, gen<={ga_gens}, early stop R2>0.995)",
                        np.asarray(tg)))

    params = count_parameters(model)
    memory = {
        "actor_parameters": params["actor"],
        "critic_parameters": params["critic"],
        "total_parameters": params["total"],
        "actor_float32_kB": round(params["actor"] * 4 / 1024, 1),
        "actor_int8_kB (estimate, not quantised)": round(params["actor"] / 1024, 1),
        "actor_plus_critic_float32_kB": round(params["total"] * 4 / 1024, 1),
        "model_zip_kB": round(model_path.stat().st_size / 1024, 1),
    }
    return {"machine": machine_info(), "rows": rows, "memory": memory,
            "coremark_scale": COREMARK_SCALE, "hil_deadline_ms": HIL_DEADLINE_MS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--model_path", dest="model", type=Path,
                    default=Path("models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip"))
    ap.add_argument("--cases", type=Path, default=Path("data/sample/synthetic_isoline"))
    ap.add_argument("--repetitions", type=int, default=3000)
    ap.add_argument("--xgb-model", type=Path, default=None,
                    help="joblib file of a MultiOutputRegressor trained on GA labels")
    ap.add_argument("--ga-cases", type=int, default=0)
    ap.add_argument("--ga-pop", type=int, default=50)
    ap.add_argument("--ga-gens", type=int, default=100)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/profile.csv"))
    args = ap.parse_args()

    res = profile(args.model, args.cases, args.repetitions, args.xgb_model,
                  args.ga_cases, args.ga_pop, args.ga_gens, args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
        w.writeheader()
        w.writerows(res["rows"])
    with open(args.out.with_suffix(".json"), "w") as f:
        json.dump(res, f, indent=2)

    print("Machine:", json.dumps(res["machine"]))
    for r in res["rows"]:
        print(f"  {r['component']:<62s} mean {r['mean_ms']:>10.4f} ms   p99 {r['p99_ms']:>10.4f} ms  (n={r['n']})")
    print("Memory:", json.dumps(res["memory"]))
    print(f"MCU figures are CoreMark-scaled estimates (x{COREMARK_SCALE:.0f}), not measurements.")
    print(f"Results -> {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
