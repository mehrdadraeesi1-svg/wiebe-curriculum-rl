# Input and output formats

## Input: one JSON file per operating point

A case is a single ensemble-averaged cycle at one operating point. The
loader (`src.utils.data_loader.load_case` / `load_cases`) reads one file or
every `*.json` file in a directory.

```json
{
  "logName": "my_engine_2000rpm_150kPa",
  "engine_geometry": {
    "bore_m": 0.083,
    "stroke_m": 0.0923,
    "connecting_rod_m": 0.1527,
    "compression_ratio": 12.5
  },
  "operating_conditions": {
    "speed_rpm": 2000.0,
    "lambda": 1.30,
    "intake_pressure_Pa": 150000.0,
    "intake_temperature_K": 318.0,
    "ca50_deg": 8.5
  },
  "pressure_trace": {
    "theta_deg":   [-180.0, -179.9, "...", 180.0],
    "pressure_Pa": [ 1.62e5,  1.62e5, "...", 3.1e5]
  }
}
```

| Field | Unit | Required | Notes |
|---|---|---|---|
| `engine_geometry.bore_m`, `stroke_m`, `connecting_rod_m` | m | yes | |
| `engine_geometry.compression_ratio` | – | yes | geometric |
| `operating_conditions.speed_rpm` | rpm | yes | |
| `operating_conditions.lambda` | – | yes | relative air–fuel ratio |
| `operating_conditions.intake_pressure_Pa` | **Pa** (absolute) | yes | not kPa |
| `operating_conditions.intake_temperature_K` | K | yes | |
| `pressure_trace.theta_deg` | deg ATDC (firing TDC = 0) | yes | monotonic; the study used −180…180 deg at 0.1 deg (3601 points) |
| `pressure_trace.pressure_Pa` | Pa (absolute) | yes | same length as `theta_deg`; pegged absolute pressure |
| `logName` | – | no | case identifier (file stem otherwise) |
| `operating_conditions.ca50_deg` (or top-level `ca50_deg`) | deg ATDC | no | measured CA50; **activates the combustion-phasing reward term** (`w_ca50`). The dataset of the associated study did not contain it, so that term was inactive there. |
| `ground_truth_params` | – | no | known parameters (synthetic cases); used for error columns |
| `Tw_wall_K`, `gamma`, `fuel_LHV_J_per_kg`, `T_ign_deg`, `T_inj_deg` | – | no | read and stored, **not used by the v1.0.0 solver** (constants: γ = 1.35, LHV = 44 MJ/kg, wall temperature 400 K; see docs/MODEL.md) |

Requirements for a meaningful calibration:

* the trace must cover the compression stroke from at least −180 deg (the
  solver starts from the first sample, treated as IVC) to past the end of
  combustion;
* the policy observes 20 consecutive samples centred on the pressure peak;
  with a resolution other than 0.1 deg this window covers a different crank
  interval than in training;
* the bundled policy was trained on a single engine (geometry above) and on
  1500–4500 rpm, 100.5–301.5 kPa, λ = 0.94–2.07 and intake temperature
  316–320 K. `PolicyCalibrator.check_envelope()` and
  `scripts/calibrate_cases.py` warn when a case is outside this envelope.

A CSV of many traces can be converted with a few lines of Python: build the
dictionary above per operating point and `json.dump` it.

## Output of `scripts/calibrate_cases.py`

`<out>.csv`, one row per case:

| Column | Unit | Meaning |
|---|---|---|
| `theta0` | deg ATDC | start of combustion |
| `delta_theta` | deg | duration of the first (premixed) Wiebe |
| `a` | – | Wiebe efficiency parameter (shared) |
| `m1`, `m2` | – | form factors of the fast and slow Wiebe |
| `lambda_w` | – | weight of the fast Wiebe, x_b = λ_w x₁ + (1−λ_w) x₂ |
| `k` | – | duration ratio, Δθ₂ = k Δθ |
| `ht_g`, `ht_c` | – | scale factors of the global and combustion-phase Woschni terms |
| `r2`, `nrmse` | –, % of peak | reconstruction of the measured trace by the 0-D model with these parameters |
| `pmax`, `tmax` | Pa, K | peak pressure and peak mean gas temperature of the reconstruction |
| `xb_final` | – | burned fraction at the end of the trace |
| `ca50_pred` | deg ATDC | CA50 of the calibrated burn law |
| `success` | bool | criterion of Ref. [1]: R² ≥ 0.95, NRMSE ≤ 5 %, peak pressure within 2–200 bar |
| `all_constraints` | bool | `success` and peak temperature ≤ 4000 K and burn completeness/monotonicity penalty ≤ 0.05 |
| `params_at_bound` | – | parameters lying on an admissible bound (not identified by the data) |
| `envelope_warnings` | – | inputs outside the policy's training envelope or geometry |
| `actor_time_ms` | ms | measured actor forward-pass time for this case |
| `err_<param>` | – | calibrated minus true value (synthetic cases only) |

`<out>.json` holds the same results and the metadata of the model used;
`--plot DIR` writes measured vs reconstructed pressure for each case.

## Model files

A trained policy is a Stable-Baselines3 `.zip` plus a sidecar `.json` with
the same stem. The sidecar stores the training-set normalisation statistics,
the action span the policy was trained with, the reward weights and the
training envelope. `scripts/train_on_split.py` writes it automatically.
