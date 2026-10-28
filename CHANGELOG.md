# Changelog

## v1.0.0 (2026-10)

First tagged release, accompanying the Software Impacts article. Results of
the associated research article (Ref. [1], Energy Convers. Manag. X 31 (2026)
101986) were produced with the pre-release code (v0.x); every v0.x behaviour
that affects those numbers is kept available behind an explicit option, so
the published numbers remain reproducible.

### Fixed

* **Import error.** `src/environment/wiebe_env.py` imported `data_loader`
  as a top-level module, so the package could not be imported from a fresh
  clone.
* **Curriculum never advanced in v0.x.** Each sub-environment of a
  vectorised environment counted only its own steps. With the configuration
  of Ref. [1] (n_envs = 4, 300 000 steps, thresholds at 100 000 and
  200 000) every sub-environment reached 75 000 steps, so training stayed in
  stage 1 (α = 0.50, β = 0.30) for the whole run. `CurriculumSyncCallback`
  (`src/curriculum.py`) now drives all sub-environments with the learner's
  total step count; `train_on_split.py --curriculum synced` is the default.
  `--curriculum legacy` reproduces v0.x; `--curriculum fixed-final` trains
  without a curriculum. A regression test covers both behaviours.
* **Evaluation normalisation.** The v0.x evaluation scripts built the
  environment from the test cases, so operating-condition z-scores and
  pressure scales were computed from the test set rather than from the
  training set the policy was trained on. Normalisation statistics are now
  frozen at training time, stored in a sidecar JSON next to every model and
  used by default (`normalization="frozen"` / `--normalization train`).
  `legacy` / `test` reproduces Ref. [1].
* **Profiler fallbacks removed.** `realtime_feasibility_wiebe.py` (v0.x)
  substituted a synthetic GA timing distribution (normal(1900, 220) ms, seed
  99) when no GA timing file was found, and timed an XGBoost model fitted on
  random data when no model path was given. `scripts/profile_inference.py`
  reports measured values only; MCU figures are in separate columns
  labelled as CoreMark-scaled estimates.
* `config/default_config.yaml` now matches the code and is read by
  `train_on_split.py --config`.
* `scripts/compensation_analysis.py` is the total-variation version used
  for Ref. [1] (v0.x shipped an earlier CV-based version) and now reports how
  often each PPO parameter lies on an admissible bound.
* The optional `ca50_deg` field is passed from the input file to the
  environment. Without it the CA50 reward term is zero; it was zero in all
  runs of Ref. [1] because the dataset has no CA50 field.

### Added

* `src/calibrate.py`: `PolicyCalibrator` API (load model + sidecar,
  calibrate cases, success flags, bound flags, training-envelope warnings).
* `scripts/calibrate_cases.py`: command-line calibration with CSV/JSON output
  and plots; also runs the GA.
* Pretrained policy `models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip`
  with its sidecar: PPO trained with the corrected (synchronised) curriculum
  on the speed-isoline split of Ref. [1] (seed 2024, best validation
  checkpoint). Selected among three seeds on the 120 validation cases
  (61.7 % success, R² 0.952); 100 % strict success and R² 0.980 on the 59
  held-out test points (three-seed mean 83.1 ± 18.3 %).
* Synthetic verification set (`data/sample/generate_sample.py`): ten
  operating points generated with the package forward model and known
  parameters.
* Ported from the research code: split generation (`make_splits.py`,
  `src/utils/data_splits.py`), PPO/A2C training on a split
  (`train_on_split.py`), GA baseline (`src/baselines/ga.py`), XGBoost
  baseline (`xgboost_baseline.py`).
* Test suite (`tests/`, pytest) and GitHub Actions CI on Linux, Windows and
  macOS with Python 3.9 and 3.12.
* `docs/DATA_FORMAT.md`, `docs/MODEL.md`, `docs/REPRODUCIBILITY.md`,
  `CITATION.cff`.

### Removed

* `scripts/evaluate_agent.py`, `src/utils/data_utils.py`,
  `src/utils/visualization.py` (v0.x leftovers, not used by any workflow).

### Known limitations (unchanged on purpose)

See `docs/MODEL.md`: inconsistent first pressure sample, single γ, fixed
wall temperature, GA search box and efficiency closure different from the
RL environment.
