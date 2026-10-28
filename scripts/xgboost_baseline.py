"""
Stage 4 -- XGBoost baseline comparison (addresses R1-Q1).

Trains an XGBoost (and LightGBM) regressor that maps operating
conditions to Wiebe parameters, using GA-optimised parameters as
pseudo-ground-truth labels. Evaluates on the same four splits used in
stage 2/3 to enable a fair head-to-head comparison with PPO-Double.

Pipeline
--------

  1. For each split_*.json from stage 1, load all cases.
  2. Run GA on EVERY case to obtain per-point "ground-truth" Wiebe
     parameters (9 params for double Wiebe).
     -> cached to disk; running again skips GA if cache exists.
  3. Train XGBoost on the train partition and evaluate on the test
     partition of that split.
  4. For each test case, simulate the pressure trace using the
     predicted Wiebe parameters and compute R^2, NRMSE, success rate.
  5. Produce a comparison figure: XGBoost vs PPO-Double across splits.

Run from the project root:

    python scripts/xgboost_baseline.py \
        --data-dir json_cases \
        --stage1-dir splits \
        --output-dir results/xgboost \
        --ga-pop 50 --ga-gens 100

The first run will take ~30-60 min on CPU because GA runs on 1258 cases.
Subsequent runs reuse the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import WiebeDataLoader  # noqa: E402
from src.models.wiebe_functions import double_wiebe  # noqa: E402
from src.utils.thermodynamics import (  # noqa: E402
    EngineGeometry,
    EngineOperatingConditions,
    GasProperties,
    simulate_pressure_trace,
    calculate_normalized_rmse,
    calculate_r_squared,
)
from src.baselines.ga import GeneticAlgorithm  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature / target schemas
# ---------------------------------------------------------------------------

FEATURE_KEYS = (
    "speed", "lambda", "intake_pressure", "intake_temperature",
    "spark_timing", "compression_ratio",
)
TARGET_KEYS = (
    "theta0", "delta_theta", "a", "m1", "m2", "lambda_w", "k",
    "ht_g", "ht_c",
)


def extract_feature_vector(case: Dict) -> np.ndarray:
    vec = []
    for k in FEATURE_KEYS:
        v = case.get(k, 0.0)
        if v is None:
            v = 0.0
        vec.append(float(v))
    return np.asarray(vec, dtype=np.float64)


def extract_target_vector(params: Dict) -> np.ndarray:
    return np.asarray(
        [float(params[k]) for k in TARGET_KEYS], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# GA-label generation with on-disk caching
# ---------------------------------------------------------------------------

def compute_or_load_ga_labels(
    cases: Sequence[Dict],
    cache_file: Path,
    pop_size: int = 50,
    n_generations: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Return {case_id -> {param_name -> value}}. Cached on disk."""
    if cache_file.exists():
        logger.info("Loading cached GA labels from %s", cache_file)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info("Computing GA labels for %d cases "
                "(pop=%d, gens=%d) -- this can take a while...",
                len(cases), pop_size, n_generations)

    ga = GeneticAlgorithm(
        wiebe_type="double",
        population_size=pop_size,
        n_generations=n_generations,
        seed=seed,
    )

    labels: Dict[str, Dict[str, float]] = {}
    t_start = time.perf_counter()
    for i, case in enumerate(cases):
        best_params, best_fitness, _ = ga.optimize(case, verbose=False)
        labels[case["case_id"]] = dict(best_params)

        if (i + 1) % 50 == 0 or i == len(cases) - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(cases) - i - 1) / rate if rate > 0 else 0
            logger.info("  [%d/%d] R^2=%.4f, %.1f cases/s, ETA %.1f s",
                        i + 1, len(cases), best_fitness, rate, eta)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(labels, f)
    logger.info("Cached GA labels to %s", cache_file)
    return labels


# ---------------------------------------------------------------------------
# XGBoost training & evaluation
# ---------------------------------------------------------------------------

@dataclass
class XGBoostRunResult:
    split_strategy: str
    model_name: str
    n_train: int
    n_test: int
    mean_z_nn: float
    mean_r2: float
    mean_nrmse: float
    success_rate: float
    inference_time_ms: float
    param_smoothness_cv: float   # coefficient of variation among params
    per_case_r2: List[float]
    per_case_nrmse: List[float]


def _build_regressor(model_name: str, seed: int):
    if model_name.lower() == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name.lower() == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_name.lower() == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            solver="adam",
            max_iter=400,
            random_state=seed,
            early_stopping=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_predicted_params(
    cases: Sequence[Dict], pred_params: np.ndarray,
    success_r2: float = 0.95, success_nrmse_pct: float = 5.0,
) -> Tuple[List[float], List[float], float, float]:
    """Given predicted Wiebe params for each test case, simulate
    pressure traces and compute per-case R^2 and NRMSE.
    """
    per_case_r2 = []
    per_case_nrmse = []
    success_count = 0
    t_per_case = []

    gas_props = GasProperties()

    for case, params_vec in zip(cases, pred_params):
        params = dict(zip(TARGET_KEYS, params_vec))

        theta_exp = np.asarray(case["crank_angle"])
        pressure_exp = np.asarray(case["pressure"])

        geometry = EngineGeometry(
            bore=case.get("bore", 0.086),
            stroke=case.get("stroke", 0.086),
            connecting_rod=case.get("connecting_rod", 0.143),
            compression_ratio=case.get("compression_ratio", 10.5),
        )
        operating = EngineOperatingConditions(
            speed=case.get("speed", 2000),
            lambda_=case.get("lambda", 1.0),
            intake_pressure=case.get("intake_pressure", 1e5),
            intake_temperature=case.get("intake_temperature", 300),
        )

        t0 = time.perf_counter()
        try:
            xb = double_wiebe(
                theta_exp,
                params["theta0"], params["delta_theta"], params["a"],
                params["m1"], params["m2"], params["lambda_w"], params["k"],
            )
            eta = 0.85 + 0.15 * params["lambda_w"]
            P_pred, _, _ = simulate_pressure_trace(
                theta_exp, xb, geometry, operating, gas_props,
                eta=eta,
                ht_scale_global=params["ht_g"],
                ht_scale_combustion=params["ht_c"],
            )
            if not np.all(np.isfinite(P_pred)):
                per_case_r2.append(float("nan"))
                per_case_nrmse.append(float("nan"))
                t_per_case.append((time.perf_counter() - t0) * 1000.0)
                continue

            r2 = calculate_r_squared(P_pred, pressure_exp)
            nrmse = calculate_normalized_rmse(P_pred, pressure_exp)
        except Exception as exc:
            logger.debug("Simulation failure: %s", exc)
            r2, nrmse = float("nan"), float("nan")

        t_per_case.append((time.perf_counter() - t0) * 1000.0)
        per_case_r2.append(float(r2))
        per_case_nrmse.append(float(nrmse))

        if np.isfinite(r2) and np.isfinite(nrmse):
            if r2 >= success_r2 and nrmse <= success_nrmse_pct:
                success_count += 1

    valid = [(r, n) for r, n in zip(per_case_r2, per_case_nrmse)
             if np.isfinite(r) and np.isfinite(n)]
    success_rate = 100.0 * success_count / max(len(cases), 1)
    mean_time_ms = float(np.mean(t_per_case)) if t_per_case else 0.0

    return per_case_r2, per_case_nrmse, success_rate, mean_time_ms


def compute_param_smoothness_cv(pred_params: np.ndarray) -> float:
    """A scalar summary of parameter scattering: mean of the
    coefficient of variation across param dimensions.

    Lower = smoother parameters. RL is expected to be smoother than
    tree-based regressors (which produce axis-aligned step functions).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.abs(pred_params).mean(axis=0)
        stds = pred_params.std(axis=0)
        cv = np.where(means > 1e-6, stds / means, 0.0)
    return float(np.mean(cv))


def run_xgboost_on_split(
    split_file: Path, all_cases: Sequence[Dict],
    ga_labels: Dict[str, Dict[str, float]],
    model_name: str, seed: int = 42,
) -> XGBoostRunResult:
    with open(split_file) as f:
        split_payload = json.load(f)

    id_to_case = {c["case_id"]: c for c in all_cases}
    train_cases = [id_to_case[cid] for cid in split_payload["train"]
                   if cid in id_to_case and cid in ga_labels]
    test_cases = [id_to_case[cid] for cid in split_payload["test"]
                  if cid in id_to_case and cid in ga_labels]

    X_train = np.vstack([extract_feature_vector(c) for c in train_cases])
    Y_train = np.vstack([extract_target_vector(ga_labels[c["case_id"]])
                         for c in train_cases])
    X_test = np.vstack([extract_feature_vector(c) for c in test_cases])

    logger.info("[%s/%s] Train: %d, Test: %d",
                split_payload["strategy"], model_name,
                len(train_cases), len(test_cases))

    from sklearn.multioutput import MultiOutputRegressor
    base = _build_regressor(model_name, seed)
    model = MultiOutputRegressor(base, n_jobs=1)

    t0 = time.perf_counter()
    model.fit(X_train, Y_train)
    fit_time = time.perf_counter() - t0
    logger.info("  Fit time: %.1f s", fit_time)

    Y_pred = model.predict(X_test)

    per_r2, per_nrmse, success_rate, mean_ms = evaluate_predicted_params(
        test_cases, Y_pred,
    )

    valid_r2 = [r for r in per_r2 if np.isfinite(r)]
    valid_nrmse = [n for n in per_nrmse if np.isfinite(n)]

    return XGBoostRunResult(
        split_strategy=split_payload["strategy"],
        model_name=model_name,
        n_train=len(train_cases),
        n_test=len(test_cases),
        mean_z_nn=float(
            split_payload.get("diagnostics", {})
                .get("mean_test_to_train_NN_distance_zscore", float("nan"))
        ),
        mean_r2=float(np.mean(valid_r2)) if valid_r2 else float("nan"),
        mean_nrmse=float(np.mean(valid_nrmse)) if valid_nrmse else float("nan"),
        success_rate=success_rate,
        inference_time_ms=mean_ms,
        param_smoothness_cv=compute_param_smoothness_cv(Y_pred),
        per_case_r2=per_r2,
        per_case_nrmse=per_nrmse,
    )


# ---------------------------------------------------------------------------
# Comparison figure: XGBoost vs PPO-Double
# ---------------------------------------------------------------------------

def short_split_name(s: str) -> str:
    import re
    if "random" in s:
        return "Random"
    if "grid_stratified" in s:
        return "Grid-stratified"
    if "speed_isoline" in s:
        m = re.search(r"N=(\d+)rpm", s)
        n = m.group(1) if m else "?"
        return f"Speed-isoline\n(N={n} RPM)"
    if "hlhs" in s:
        return "HL+HS\nextrapolation"
    if "high_load" in s:
        return "High-load\nextrapolation"
    return s


def plot_comparison(
    xgb_df: pd.DataFrame, rl_df: Optional[pd.DataFrame], out_path: Path,
) -> None:
    """Side-by-side bar chart: XGBoost (+ optionally LightGBM/MLP) vs
    PPO-Double."""
    xgb_df = xgb_df.copy()
    xgb_df["split_short"] = xgb_df["split_strategy"].apply(short_split_name)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    colors = {
        "xgboost": "#639922",
        "lightgbm": "#BA7517",
        "mlp": "#5DCAA5",
        "PPO": "#7F77DD",
    }

    models_in_xgb = xgb_df["model_name"].unique().tolist()
    all_methods = models_in_xgb + (["PPO"] if rl_df is not None else [])

    if xgb_df["mean_z_nn"].notna().any():
        split_order_vals = (xgb_df.groupby("split_short")["mean_z_nn"]
                            .mean().sort_values())
        split_order = list(split_order_vals.index)
    else:
        split_order = sorted(xgb_df["split_short"].unique())

    metrics = [
        ("mean_r2", r"$R^2$ (higher is better)", True),
        ("mean_nrmse", "NRMSE (%) (lower is better)", False),
        ("success_rate", "Success rate (%) (higher is better)", True),
    ]

    for ax, (metric, ylabel, _) in zip(axes, metrics):
        x = np.arange(len(split_order))
        n_methods = len(all_methods)
        width = 0.8 / n_methods

        for j, method in enumerate(all_methods):
            offset = (j - n_methods / 2 + 0.5) * width
            vals = []
            for s in split_order:
                if method == "PPO" and rl_df is not None:
                    rl_row = rl_df[rl_df["split_short"] == s]
                    if not rl_row.empty:
                        vals.append(float(rl_row[metric].iloc[0]))
                    else:
                        vals.append(np.nan)
                else:
                    row = xgb_df[(xgb_df["split_short"] == s)
                                 & (xgb_df["model_name"] == method)]
                    if not row.empty:
                        vals.append(float(row[metric].iloc[0]))
                    else:
                        vals.append(np.nan)

            bar_positions = x + offset
            vals_arr = np.asarray(vals)
            finite = ~np.isnan(vals_arr)
            label_pretty = {"xgboost": "XGBoost", "lightgbm": "LightGBM",
                            "mlp": "MLP", "PPO": "PPO-Double"}.get(method, method)
            ax.bar(bar_positions[finite], vals_arr[finite],
                   width=width, color=colors.get(method, "#B4B2A9"),
                   alpha=0.65, edgecolor="white", linewidth=0.8,
                   label=label_pretty)

        ax.set_xticks(x)
        ax.set_xticklabels(split_order, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.set_title(ylabel.split(" (")[0], fontsize=10)

    fig.suptitle(
        "XGBoost / supervised baselines vs PPO-Double (Double Wiebe)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("json_cases"))
    parser.add_argument("--stage1-dir", type=Path,
                        default=Path("splits"))
    parser.add_argument("--stage3-csv", type=Path,
                        default=Path("results/benchmark/results_per_run.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/xgboost"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("results/xgboost/cache"))
    parser.add_argument("--models", type=str, nargs="+",
                        default=["xgboost"],
                        help="Which SL regressors to evaluate.")
    parser.add_argument("--ga-pop", type=int, default=50)
    parser.add_argument("--ga-gens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    loader = WiebeDataLoader(str(args.data_dir))
    all_cases = loader.load_all_cases()
    logger.info("Loaded %d cases.", len(all_cases))

    ga_cache = args.cache_dir / f"ga_labels_pop{args.ga_pop}_gen{args.ga_gens}_seed{args.seed}.pkl"
    ga_labels = compute_or_load_ga_labels(
        all_cases, ga_cache,
        pop_size=args.ga_pop, n_generations=args.ga_gens, seed=args.seed,
    )

    split_files = [
        args.stage1_dir / "split_grid_stratified.json",
        args.stage1_dir / "split_speed_isoline_holdout.json",
        args.stage1_dir / "split_extrapolation_high_load.json",
        args.stage1_dir / "split_extrapolation_hl_hs.json",
    ]

    results: List[XGBoostRunResult] = []
    for split_file in split_files:
        if not split_file.exists():
            logger.warning("Missing split file: %s -- skipping", split_file)
            continue
        for model_name in args.models:
            result = run_xgboost_on_split(
                split_file, all_cases, ga_labels,
                model_name, seed=args.seed,
            )
            results.append(result)
            logger.info(
                "  -> R^2=%.4f  NRMSE=%.2f%%  Success=%.1f%%  "
                "inference=%.2f ms",
                result.mean_r2, result.mean_nrmse,
                result.success_rate, result.inference_time_ms,
            )

    rows = []
    for r in results:
        row = asdict(r)
        row.pop("per_case_r2")
        row.pop("per_case_nrmse")
        rows.append(row)
    xgb_df = pd.DataFrame(rows)
    xgb_df.to_csv(args.output_dir / "results_xgboost.csv", index=False)
    logger.info("Wrote results_xgboost.csv")

    rl_df = None
    if args.stage3_csv.exists():
        stage3 = pd.read_csv(args.stage3_csv)
        stage3 = stage3[stage3["algorithm"].str.upper() == "PPO"]
        stage3 = stage3.rename(columns={"split_short": "split_short"})
        if "split_short" not in stage3.columns:
            stage3["split_short"] = stage3["split_strategy"].apply(
                short_split_name
            )
        rl_df = stage3
        logger.info("Loaded stage3 RL results (%d rows) for comparison.",
                    len(rl_df))

    plot_comparison(xgb_df, rl_df, args.output_dir / "comparison_figure.png")

    write_summary(xgb_df, rl_df, args.output_dir / "summary.md")

    logger.info("Stage 4 complete -> %s", args.output_dir.resolve())


def write_summary(xgb_df: pd.DataFrame, rl_df: Optional[pd.DataFrame],
                  out_path: Path) -> None:
    lines = ["# Stage 4 summary: XGBoost baseline vs PPO-Double\n"]

    xgb_df = xgb_df.copy()
    xgb_df["split_short"] = xgb_df["split_strategy"].apply(short_split_name)

    lines.append("## XGBoost results\n")
    lines.append("| Split | z-NN | n_test | R^2 | NRMSE (%) | Success (%) | Inference (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in xgb_df.iterrows():
        lines.append(
            f"| {r['split_short'].replace(chr(10), ' ')} | "
            f"{r['mean_z_nn']:.3f} | {int(r['n_test'])} | "
            f"{r['mean_r2']:.4f} | {r['mean_nrmse']:.2f} | "
            f"{r['success_rate']:.1f} | {r['inference_time_ms']:.2f} |"
        )
    lines.append("")

    if rl_df is not None and not rl_df.empty:
        lines.append("## Head-to-head comparison (XGBoost vs PPO-Double)\n")
        lines.append("| Split | Model | R^2 | NRMSE (%) | Success (%) |")
        lines.append("|---|---|---|---|---|")
        split_order = list(
            xgb_df.sort_values("mean_z_nn")["split_short"].unique()
        )
        for split_name in split_order:
            for df_label, df_src in [("XGBoost", xgb_df), ("PPO-Double", rl_df)]:
                if df_label == "XGBoost":
                    row = df_src[df_src["split_short"] == split_name]
                    if not row.empty:
                        r = row.iloc[0]
                        lines.append(
                            f"| {split_name.replace(chr(10), ' ')} | {df_label} | "
                            f"{r['mean_r2']:.4f} | {r['mean_nrmse']:.2f} | "
                            f"{r['success_rate']:.1f} |"
                        )
                else:
                    row = df_src[df_src["split_short"] == split_name]
                    if not row.empty:
                        r = row.iloc[0]
                        lines.append(
                            f"| {split_name.replace(chr(10), ' ')} | {df_label} | "
                            f"{r['mean_r2']:.4f} | {r['mean_nrmse']:.2f} | "
                            f"{r['success_rate']:.1f} |"
                        )

    out_path.write_text("\n".join(lines))
    logger.info("Summary -> %s", out_path)


if __name__ == "__main__":
    main()
