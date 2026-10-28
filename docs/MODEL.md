# Model and implementation notes

## 0-D thermodynamic model (`src/utils/thermodynamics.py`)

Single-zone, closed-system first-law model from the first sample of the
trace (treated as intake-valve closing) to the last:

* volume from slider-crank kinematics;
* trapped mass m = 0.85 p_int V_d / (R T_int);
* fuel energy Q = η m_f LHV with m_f = m / (1 + 14.7 λ), LHV = 44 MJ/kg and
  combustion efficiency η = 0.85 + 0.1 λ_w (double Wiebe; RL environment);
* chemical heat release Q_chem(θ) = x_b(θ) Q;
* internal-energy update **U_i = U_{i−1} + ΔQ_chem − p_{i−1} ΔV − ΔQ_wall**
  (explicit first-order integration on the crank-angle grid), T = U(γ−1)/(mR)
  with γ = 1.35, p = mRT/V, T ≥ 300 K;
* wall heat transfer: Woschni-type correlation
  h = 130 c_g B^−0.2 p[bar]^0.8 T^−0.53 w^0.8,
  w = 2.28 S_p + c_c · 0.00324 (V_d T_ref / (p V_ref)) (p − p_mot) after the
  start of combustion, applied to the piston area with a 400 K wall.
  c_g = `ht_g` and c_c = `ht_c` are calibrated.

Implemented in Numba (`@jit(nopython=True)`).

Known simplifications, kept unchanged in v1.0.0 so that published results
remain reproducible:

* the first pressure sample is set to 1.1 p_int while the state from the
  first step on follows p = mRT/V with T_IVC = 1.05 T_int, so samples 0 and 1
  are not consistent; only the first sample is affected;
* single γ, no residual gas, no crevice or blow-by model;
* `gamma`, `Tw_wall_K` and `fuel_LHV_J_per_kg` from the input file are not
  used.

The test suite checks this model against analytical limits (isentropic
motored compression, exact discrete energy balance, released heat equal to
fuel energy, monotonic effect of the heat-transfer scales, linear scaling
with intake pressure).

## Burn law (`src/models/wiebe_functions.py`)

x_b = λ_w [1 − exp(−a ((θ−θ₀)/Δθ)^(m₁+1))] + (1−λ_w) [1 − exp(−a ((θ−θ₀)/(kΔθ))^(m₂+1))]

### Identifiability

Each Wiebe term depends on a and its duration only through
a / Δθ^(m+1). With a shared a, the double Wiebe law therefore depends on
(a, Δθ, k) only through two combinations, a/Δθ^(m₁+1) and a/(kΔθ)^(m₂+1),
so for fixed m₁, m₂ there is a one-dimensional family of (a, Δθ, k) giving
exactly the same burn law. Individual values of a, Δθ and k are not
identifiable from a pressure trace; compare calibrations through the burn
law (e.g. CA10/CA50/CA90) rather than through these parameters. This is
one exact source of the compensation effect.

## Environment (`src/environment/wiebe_env.py`)

One-step episodes: observe a trace, output nine actions in [−1, 1], receive
the reward of the reconstruction.

* Observation (93 values): 20 samples centred on the pressure peak, each
  (p/p_scale, (dp/dθ)/dp_scale, normalised θ, normalised cumulative p dV);
  z-scored speed, λ, intake pressure and intake temperature; previous action.
* Action mapping onto the stage's admissible box: linear (θ₀, k, ht_g,
  ht_c), logarithmic (Δθ, a, m₁, m₂), sigmoid (λ_w).
* Physical limits (full box, stage 3): θ₀ ∈ [−20, 40] deg ATDC,
  Δθ ∈ [5, 120] deg, a ∈ [0.1, 8], m ∈ [0.5, 4], λ_w ∈ [0.05, 0.95],
  k ∈ [1, 4], ht ∈ [0.5, 3].
* Stage k uses the central fraction α_k of each range: α = 0.50, 0.75, 1.00,
  and operating-condition noise scaled by β = 0.30, 0.65, 1.00.
* Reward = −w_nrmse NRMSE/100 + w_r2 R² − w_peakP (peak-pressure excursion
  outside 2–200 bar) − w_tb (peak temperature above 4000 K) − w_xb (|x_b,end −
  0.98| + monotonicity violation) − w_ca50 |CA50 error|/50 (only when the case
  has `ca50_deg`).

### Admissible box of v0.x policies (α = 0.5)

The policies of Ref. [1] were trained in stage 1 only (see CHANGELOG.md) and
act in θ₀ ∈ [−5, 25] deg ATDC, Δθ ∈ [33.75, 91.25] deg, a ∈ [2.08, 6.03],
m ∈ [1.38, 3.13], λ_w ∈ [0.28, 0.73], k ∈ [1.75, 3.25], ht ∈ [1.13, 2.38];
their sidecar must therefore carry `eval_action_span: 0.5`. The bundled
v1.0.0 policy was trained with the synchronised curriculum and acts in the
full box (`eval_action_span: 1.0`). It places `a` on its lower bound (0.1)
for every case, which is allowed by the exact (a, Δθ, k) degeneracy above.

## Baselines

* **GA** (`src/baselines/ga.py`): real-coded GA, population 50, up to 100
  generations, tournament selection, elitism 5, early stop at R² > 0.995.
  Its search box differs from the RL box (θ₀ ∈ [−40, 20], Δθ ∈ [30, 100],
  a ∈ [3, 8], m ∈ [1, 4], λ_w ∈ [0.3, 0.8], k ∈ [2, 15]) and it uses
  η = 0.85 + 0.15 λ_w. Its fitness calls `calculate_r_squared(measured,
  predicted)`, i.e. the total sum of squares is taken about the mean of the
  predicted trace. These choices are those of Ref. [1] and are kept.
* **XGBoost** (`scripts/xgboost_baseline.py`): multi-output regressor from
  (speed, λ, p_int, T_int, spark timing, compression ratio) to the nine GA
  labels.
