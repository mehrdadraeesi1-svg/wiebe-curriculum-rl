# Retraining with the corrected curriculum (v1.0.0)

Split: speed-isoline holdout (3000 rpm withheld; 1079 train / 120 val / 59
test cases), PPO, 300 000 steps, 4 environments, hyper-parameters and reward
weights of Ref. [1], seeds 42 / 123 / 2024. Evaluation on the 59 test cases
with training-set normalisation and the action box the policy was trained
for. Success: R² ≥ 0.95, NRMSE ≤ 5 %, peak pressure 2–200 bar (temperature
and burn constraints also held at every successful point).

| Training configuration | Action box at the end | Success, best val. checkpoint | Success, final model | R² (best) |
|---|---|---|---|---|
| Published runs of Ref. [1] (v0.x; curriculum stalled in stage 1) | α = 0.5 | 100.0 ± 0.0 % | – | 0.988 ± 0.002 |
| `--curriculum synced` (stages switch at 100k / 200k total steps) | α = 1.0 | 83.1 ± 18.3 % (57.6 / 91.5 / 100) | 88.1 ± 8.4 % | 0.972 ± 0.009 |
| `--curriculum fixed-final` (full box and noise from step 0) | α = 1.0 | 50.8 ± 8.6 % (54.2 / 39.0 / 59.3) | 60.5 ± 1.6 % | 0.959 ± 0.008 |

Mean ± population standard deviation over three seeds.

Reading:

* Towards the full parameter box, the staged curriculum improves the
  strict success rate by about 30 percentage points over training without
  it.
* Training restricted to the central half of each parameter range
  (what the v0.x runs effectively did) is the most accurate on this
  holdout, at the price of parameters saturating on the restricted bounds
  (Δθ on its lower bound at 99 % of the 2000 rpm isoline points).
* In all curriculum runs the efficiency parameter `a` ends on its lower
  bound (0.1) at every test point. This is the exact (a, Δθ, k)
  degeneracy of the double Wiebe law (docs/MODEL.md): the policy may move
  along the degenerate direction without changing the burn law.

Bundled policy: `models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip`
is the `synced` seed 2024 run, selected on the 120 validation cases
(success 50.0 / 60.0 / 61.7 %, R² 0.942 / 0.944 / 0.952 for seeds
42 / 123 / 2024). Its test result (100 %, R² 0.980) is the best of the three
and is therefore optimistic; quote the three-seed mean as the expected
performance.

Reproduce:

```bash
python scripts/train_on_split.py --curriculum synced --data-dir json_cases \
    --split-file splits/split_speed_isoline_holdout.json \
    --output-dir runs/synced/seed_42 --seed 42
```
