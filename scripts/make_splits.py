"""
Stage 1 -- Split diagnostics on the REAL 1258-case GDI dataset.

This is the first artifact for the response-to-reviewers: it runs four
splitting strategies on the actual JSON case files and quantifies how
close train and test points are in the operating-condition space. No
model training is involved here -- the goal is simply to replace the
random split diagnostic with a defensible one and produce the figure
and table that will go into the response document.

Run from the project root (the directory that contains data_loader.py):

    python scripts/make_splits.py \
        --data-dir json_cases \
        --output-dir splits \
        --seed 42

Produces
--------
splits/
    split_random.json              # train/val/test case_ids + diagnostics
    split_grid_stratified.json
    split_extrapolation_high_load.json
    split_extrapolation_hl_hs.json
    split_comparison_table.txt     # table for response-to-reviewer
    split_strategies_figure.png    # 4-panel figure
    summary.md                     # human-readable summary

All outputs are deterministic given the seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Make src/ importable regardless of where this is launched from
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_splits import (  # noqa: E402
    ExtrapolationSplit,
    GridStratifiedSplit,
    KFoldBySpeed,
    RandomSplit,
    SpeedIsolineHoldout,
    SplitResult,
    compare_splits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-dataset loader (minimal, does not depend on the training code path)
# ---------------------------------------------------------------------------

def load_real_cases(data_dir: Path, max_cases: int | None = None
                    ) -> List[Dict[str, object]]:
    """Load case dictionaries directly from the JSON files.

    We deliberately do NOT import WiebeDataLoader here, so this script is
    standalone and will not accidentally trigger the random split that
    lives inside the training pipeline.
    """
    json_files = sorted(data_dir.glob("*.json"))
    if max_cases is not None:
        json_files = json_files[:max_cases]

    if not json_files:
        raise FileNotFoundError(
            f"No JSON case files found in {data_dir}. "
            "Check --data-dir argument."
        )

    cases: List[Dict[str, object]] = []
    skipped = 0
    for path in json_files:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            ops = raw["operating_conditions"]
            case = {
                "case_id": (raw.get("logName") or path.stem).strip(),
                "source_file": path.name,
                "speed": float(ops["speed_rpm"]),
                "lambda": float(ops["lambda"]),
                "intake_pressure": float(ops["intake_pressure_Pa"]),
                "intake_temperature": float(ops["intake_temperature_K"]),
                "spark_timing": float(ops.get("T_ign_deg", 0.0)),
            }
            cases.append(case)
        except (KeyError, ValueError) as exc:  # pragma: no cover
            logger.warning("Skipping %s: %s", path.name, exc)
            skipped += 1

    logger.info("Loaded %d cases from %s (skipped %d).",
                len(cases), data_dir, skipped)
    return cases


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_strategies(cases: Sequence[Dict], results: Sequence[SplitResult],
                    out_path: Path) -> None:
    """Produce the publication-quality figure for the manuscript.

    Supports 3 to 6 strategies: auto-lays out as 2x2 (for 3-4) or 2x3
    (for 5-6). Each panel gets a short "(a)" / "(b)" / "(c)" ... label
    and a friendly title derived from the strategy name.
    """
    speeds = np.asarray([c["speed"] for c in cases])
    pressures_kpa = np.asarray([c["intake_pressure"] for c in cases]) / 1000.0

    n = len(results)
    if n <= 4:
        nrows, ncols = 2, 2
        figsize = (11, 8.5)
    elif n <= 6:
        nrows, ncols = 2, 3
        figsize = (15, 8.5)
    else:
        nrows = (n + 2) // 3
        ncols = 3
        figsize = (15, 4.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    friendly_titles = {
        "random": "Random split (original submission)",
        "grid_stratified": "Grid-stratified split",
    }

    panel_letters = "abcdefghij"

    for idx, (ax, result) in enumerate(zip(axes, results)):
        id_to_set = {cid: "train" for cid in result.train_ids}
        id_to_set.update({cid: "val" for cid in result.val_ids})
        id_to_set.update({cid: "test" for cid in result.test_ids})

        train_mask = np.asarray(
            [id_to_set.get(c["case_id"]) == "train" for c in cases]
        )
        val_mask = np.asarray(
            [id_to_set.get(c["case_id"]) == "val" for c in cases]
        )
        test_mask = np.asarray(
            [id_to_set.get(c["case_id"]) == "test" for c in cases]
        )

        ax.scatter(speeds[train_mask], pressures_kpa[train_mask],
                   s=10, c="#B4B2A9", alpha=0.65, edgecolors="none",
                   label=f"Train (n={train_mask.sum()})")
        ax.scatter(speeds[val_mask], pressures_kpa[val_mask],
                   s=14, c="#EF9F27", alpha=0.85, edgecolors="none",
                   label=f"Val (n={val_mask.sum()})")
        ax.scatter(speeds[test_mask], pressures_kpa[test_mask],
                   s=18, c="#D4537E", alpha=0.85, edgecolors="none",
                   label=f"Test (n={test_mask.sum()})")

        strategy_short = result.strategy.split("[")[0]

        if strategy_short in friendly_titles:
            title_body = friendly_titles[strategy_short]
        elif strategy_short == "extrapolation":
            region = result.diagnostics.get("region_name", "OOD region")
            if region.startswith("high_load_P>="):
                p_thr = region.replace("high_load_P>=", "").replace("kPa", "")
                title_body = (f"High-load extrapolation "
                              f"($P_\\mathrm{{in}} \\geq {p_thr}$ kPa)")
            elif region.startswith("hlhs_"):
                parts = region.split("_")
                p_thr = parts[1].replace("P>=", "").replace("kPa", "")
                n_thr = parts[2].replace("N>=", "").replace("rpm", "")
                title_body = (f"High-load + high-speed "
                              f"($P_\\mathrm{{in}} \\geq {p_thr}$ kPa, "
                              f"$N \\geq {n_thr}$ RPM)")
            else:
                title_body = f"Extrapolation ({region})"
        elif strategy_short == "speed_isoline_holdout":
            speeds_held = result.diagnostics.get("holdout_speeds_rpm", [])
            speeds_str = " + ".join(f"{int(s)}" for s in speeds_held)
            title_body = f"Speed-isoline holdout (N = {speeds_str} RPM)"
        else:
            title_body = strategy_short.replace("_", " ").title()

        mean_d = result.diagnostics.get(
            "mean_test_to_train_NN_distance_zscore", float("nan"))
        min_d = result.diagnostics.get(
            "min_test_to_train_distance_zscore", float("nan"))
        subtitle = (r"$\bar{d}_{z-NN}$ = " + f"{mean_d:.3f} $\\sigma$, "
                    + r"$\min$ = " + f"{min_d:.3f} $\\sigma$")

        ax.set_title(f"({panel_letters[idx]}) {title_body}\n{subtitle}",
                     fontsize=10)
        ax.set_xlabel("Engine speed [RPM]")
        ax.set_ylabel("Intake pressure [kPa]")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Hide any unused axes
    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle(
        "Dataset partitioning strategies on the GDI operating envelope "
        f"(N = {len(cases)} cases)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def write_markdown_summary(cases: Sequence[Dict],
                           results: Sequence[SplitResult],
                           kfold_results: Sequence[SplitResult],
                           out_path: Path) -> None:
    """Human-readable summary, copy-pasteable into the response document."""
    lines: List[str] = []
    lines.append("# Split diagnostics -- real dataset\n")
    lines.append(f"Total cases loaded: **{len(cases)}**\n")

    speeds = np.asarray([c["speed"] for c in cases])
    pressures = np.asarray([c["intake_pressure"] for c in cases])
    lambdas = np.asarray([c["lambda"] for c in cases])

    lines.append("## Operating-envelope statistics\n")
    lines.append("| Variable | Min | Max | Mean | Std |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Speed [RPM] | {speeds.min():.0f} | {speeds.max():.0f} | "
                 f"{speeds.mean():.0f} | {speeds.std():.0f} |")
    lines.append(f"| Intake pressure [kPa] | {pressures.min()/1000:.1f} | "
                 f"{pressures.max()/1000:.1f} | "
                 f"{pressures.mean()/1000:.1f} | "
                 f"{pressures.std()/1000:.1f} |")
    lines.append(f"| Lambda | {lambdas.min():.3f} | {lambdas.max():.3f} | "
                 f"{lambdas.mean():.3f} | {lambdas.std():.3f} |")
    lines.append("")

    lines.append("## Split comparison\n")
    lines.append("```")
    lines.append(compare_splits(results))
    lines.append("```\n")

    lines.append("## K-fold by speed (5 folds)\n")
    lines.append("```")
    lines.append(compare_splits(kfold_results))
    lines.append("```\n")

    lines.append("## Interpretation\n")
    # First index is always random; collect non-random means for comparison.
    random_mean = results[0].diagnostics.get(
        "mean_test_to_train_NN_distance_zscore", float("nan"))
    non_random_means = []
    for r in results[1:]:
        m = r.diagnostics.get("mean_test_to_train_NN_distance_zscore",
                              float("nan"))
        if np.isfinite(m):
            non_random_means.append((r.strategy, m))
    ooD_mean = (float(np.nanmean([m for _, m in non_random_means]))
                if non_random_means else float("nan"))

    lines.append(
        f"- Random split mean nearest-neighbour distance: "
        f"**{random_mean:.3f} sigma**. "
        "The typical test point has a training neighbour very close by; "
        "this confirms Reviewer 2's concern about spatial information "
        "overlap."
    )
    for strategy_name, m in non_random_means:
        ratio = m / max(random_mean, 1e-6)
        lines.append(
            f"- {strategy_name}: **{m:.3f} sigma** "
            f"({ratio:.2f}x larger than random)."
        )
    lines.append(
        "- Model performance should be reported on all splits; if the "
        "OOD R-squared remains above 0.90, the generalisation claim "
        "becomes defensible."
    )

    out_path.write_text("\n".join(lines))
    logger.info("Markdown summary saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Stage 1: split diagnostics on the real GDI dataset. "
                     "No model training."),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("json_cases"),
                        help="Folder with the 1258 JSON case files.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("splits"),
                        help="Where to write splits, figure, summary.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test selection.")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--n-speed-bins", type=int, default=6)
    parser.add_argument("--n-pressure-bins", type=int, default=6)
    parser.add_argument("--high-load-threshold-kpa", type=float,
                        default=270.0,
                        help="P_in >= this -> test set for high-load OOD.")
    parser.add_argument("--hl-hs-load-threshold-kpa", type=float,
                        default=250.0)
    parser.add_argument("--hl-hs-speed-threshold-rpm", type=float,
                        default=3500.0)
    parser.add_argument("--isoline-holdout-speed-rpm", type=float,
                        default=3000.0,
                        help=("Engine speed (RPM) to hold out entirely. "
                              "Default 3000 RPM is a natural mid-range "
                              "isoline in the GDI swept matrix."))
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Optional cap for quick smoke tests.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cases = load_real_cases(args.data_dir, max_cases=args.max_cases)
    if len(cases) < 50:
        raise RuntimeError(
            f"Only {len(cases)} cases loaded -- split diagnostics "
            "are not meaningful. Check --data-dir."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Five primary splits for the response document
    splits = [
        ("random", RandomSplit(
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )),
        ("grid_stratified", GridStratifiedSplit(
            n_speed_bins=args.n_speed_bins,
            n_pressure_bins=args.n_pressure_bins,
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )),
        ("speed_isoline_holdout", SpeedIsolineHoldout.mid_speed(
            speed_rpm=args.isoline_holdout_speed_rpm,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )),
        ("extrapolation_high_load", ExtrapolationSplit.high_load(
            p_threshold_Pa=args.high_load_threshold_kpa * 1000.0,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )),
        ("extrapolation_hl_hs", ExtrapolationSplit.high_load_high_speed(
            p_threshold_Pa=args.hl_hs_load_threshold_kpa * 1000.0,
            speed_threshold_rpm=args.hl_hs_speed_threshold_rpm,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )),
    ]

    results: List[SplitResult] = []
    for name, splitter in splits:
        logger.info("Running split: %s", name)
        result = splitter.split(cases)
        results.append(result)
        print(result.summary())
        result.save(output_dir / f"split_{name}.json")

    # Comparison table
    table = compare_splits(results)
    print("\n" + table)
    (output_dir / "split_comparison_table.txt").write_text(table + "\n")
    logger.info("Table saved -> %s",
                output_dir / "split_comparison_table.txt")

    # K-fold diagnostic (for appendix / robustness)
    logger.info("Running 5-fold K-fold by speed (for appendix)...")
    kfold_splitter = KFoldBySpeed(
        n_folds=5, val_fraction=args.val_fraction, seed=args.seed)
    kfold_results = list(kfold_splitter.folds(cases))
    kfold_table = compare_splits(kfold_results)
    print("\n" + kfold_table)
    (output_dir / "kfold_comparison_table.txt").write_text(kfold_table + "\n")

    # N-panel figure (auto-layout for 3-6 strategies)
    plot_strategies(
        cases, results, output_dir / "split_strategies_figure.png")

    # Markdown summary
    write_markdown_summary(
        cases, results, kfold_results, output_dir / "summary.md")

    logger.info("")
    logger.info("=" * 68)
    logger.info("Stage 1 complete. See output directory:")
    logger.info("  %s", output_dir.resolve())
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
