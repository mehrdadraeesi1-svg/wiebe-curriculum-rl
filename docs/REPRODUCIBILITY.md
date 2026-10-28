# Reproducing the reported results

## What the public release reproduces on its own

| Result | Command | External data |
|---|---|---|
| Solver, burn law, environment, curriculum and pipeline tests | `pytest` | none |
| Calibration of the synthetic isoline with the bundled policy (5/10 successes, R² > 0.98 from 110 to 190 kPa) | `python data/sample/generate_sample.py && python scripts/calibrate_cases.py --cases data/sample/synthetic_isoline` | none |
| Same with the GA baseline | `... --method ga` | none |
| Latency and memory (actor, observation, full pipeline, GA) | `python scripts/profile_inference.py --ga-cases 3` | none (values depend on the CPU) |

## Results of Ref. [1] (need the dataset of Yuan et al. [2])

Convert the dataset to one JSON file per operating point
([DATA_FORMAT.md](DATA_FORMAT.md)) in `json_cases/`.

| Result in Ref. [1] | Command | Notes |
|---|---|---|
| Partition distances (0.161σ … 1.112σ) | `python scripts/make_splits.py --data-dir json_cases --output-dir splits --seed 42` | deterministic |
| PPO / A2C runs (3 seeds × 4 partitions) | `python scripts/train_on_split.py --curriculum legacy --algorithm PPO --data-dir json_cases --split-file splits/split_<name>.json --output-dir runs/<algo>_<name>/seed_<s> --seed <s>` | `legacy` = per-environment curriculum counter of v0.x |
| Table 6 (PPO vs A2C over partitions) | `python scripts/evaluate_all_splits.py --stage2-root runs --data-dir json_cases --normalization test` | `test` = v0.x test-set normalisation |
| XGBoost baseline | `python scripts/xgboost_baseline.py --data-dir json_cases --stage1-dir splits` | GA labels are cached after the first run |
| Compensation effect (TV ratios, 2000 rpm isoline) | `python scripts/compensation_analysis.py --data-dir json_cases --ppo-model runs/ppo_grid_stratified/models/best/best_model.zip --split-file splits/split_grid_stratified.json --normalization test --action-span 0.5` | |
| Speed-isoline result of a trained v0.x model | `PolicyCalibrator(model, metadata_with_eval_action_span_0.5, normalization="legacy")` on the 59 test cases of `split_speed_isoline_holdout.json` | 91.5 %, R² 0.969 (seed 42); the v0.x models are available from the authors |

Training is stochastic across hardware and library versions; evaluation of
a given trained model is deterministic.

### Verified re-evaluation of the 24 published runs

All 24 trained runs of Ref. [1] were re-evaluated with the v1.0.0 code. With
`--normalization test` they reproduce Table 6 of Ref. [1] to within 0.3
percentage points in success rate and 0.004 in R² (Table 6 lists, e.g.,
77.9 % for PPO on the grid partition; the re-evaluation gives 77.6 %); with the v1.0.0
default (`train`) the same models give:

| Partition | Algorithm | R² (test-set norm., as published) | R² (training-set norm.) | Success (published) | Success (training-set norm.) |
|---|---|---|---|---|---|
| Grid-stratified | PPO | 0.970 ± 0.001 | 0.970 ± 0.001 | 77.6 ± 1.0 % | 77.3 ± 1.0 % |
| Grid-stratified | A2C | 0.931 ± 0.002 | 0.931 ± 0.002 | 29.9 ± 5.9 % | 29.9 ± 5.6 % |
| Speed isoline (3000 rpm) | PPO | 0.971 ± 0.004 | 0.988 ± 0.002 | 91.5 ± 0.0 % | 100.0 ± 0.0 % |
| Speed isoline (3000 rpm) | A2C | 0.974 ± 0.004 | 0.974 ± 0.004 | 73.4 ± 14.0 % | 74.0 ± 14.8 % |
| High load (≥ 270 kPa) | PPO | 0.864 ± 0.014 | 0.927 ± 0.003 | 0.9 ± 0.3 % | 24.4 ± 1.7 % |
| High load (≥ 270 kPa) | A2C | 0.836 ± 0.043 | 0.841 ± 0.040 | 6.4 ± 5.1 % | 6.7 ± 5.1 % |
| High load + high speed | PPO | 0.914 ± 0.005 | 0.961 ± 0.001 | 6.2 ± 0.7 % | 50.2 ± 1.4 % |
| High load + high speed | A2C | 0.892 ± 0.026 | 0.896 ± 0.027 | 26.9 ± 4.4 % | 27.9 ± 5.0 % |

Mean ± standard deviation over seeds 42, 123 and 2024 (population std).
Success: R² ≥ 0.95, NRMSE ≤ 5 %, peak pressure within 2–200 bar. The
peak-temperature and burn constraints held at every successful point.

### Curriculum

In v0.x the curriculum counter was per sub-environment, so all runs of Ref.
[1] trained in stage 1 (α = 0.5, β = 0.3). Evaluating those policies with the
full stage-3 box gives R² < 0 — they are only valid in the box they were
trained in, which is why v1.0.0 stores `eval_action_span` in each model's
sidecar. Retraining with the corrected scheduler: see
[CURRICULUM_RETRAINING.md](CURRICULUM_RETRAINING.md).

### Latency

| Component | Measured (Intel Core i7-10700K) |
|---|---|
| PPO actor, SB3 `predict` | 0.26 ms (Ref. [1], host); 0.23 ms (Linux VM, 1 thread) |
| PPO actor, raw PyTorch forward | 0.058 ms |
| Full pipeline (observation + actor + 0-D verification) | 1.72 ms (Ref. [1]); 1.66 ms |
| XGBoost trained on GA labels (300 trees × 9 outputs) | 3.9 ms |
| GA, population 50, ≤ 100 generations, early stop R² > 0.995 | 1.86 ± 1.10 s (252 points, Ref. [1] run) |

Microcontroller values are CoreMark-scaled estimates (× 7): actor ≈ 1.8 ms,
full pipeline ≈ 12 ms. Note that the v0.x profiler reported a GA timing drawn
from a placeholder distribution (mean 1898.7 ms) and timed an XGBoost model
fitted on random data (2.26 ms); both fallbacks are removed in v1.0.0.
