"""
High-level calibration API.

    from src.calibrate import PolicyCalibrator
    from src.utils.data_loader import load_cases

    cal = PolicyCalibrator.from_pretrained("models/pretrained/ppo_speed_isoline_curriculum_seed2024.zip")
    results = cal.calibrate(load_cases("data/sample/synthetic_isoline"))
    results[0]["params"]      # nine calibrated parameters
    results[0]["metrics"]     # R2, NRMSE, peak pressure/temperature, CA50, ...
    results[0]["success"]     # strict success criterion of Ref. [1]

Every result is produced by one forward pass of the actor network followed by
one evaluation of the 0-D thermodynamic model with the predicted parameters.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.environment.wiebe_env import PhysLimits, RewardWeights, WiebeEnvV4

PARAM_NAMES = ["theta0", "delta_theta", "a", "m1", "m2", "lambda_w", "k", "ht_g", "ht_c"]
_ATTR = {"theta0": "theta0", "delta_theta": "delta_theta", "a": "a", "m1": "m1",
         "m2": "m2", "lambda_w": "lambda_w", "k": "k",
         "ht_g": "ht_scale_global", "ht_c": "ht_scale_combustion"}

# Success criterion of Ref. [1]
SUCCESS_R2 = 0.95
SUCCESS_NRMSE_PCT = 5.0


def success_flags(info: Dict[str, Any]) -> Dict[str, bool]:
    """Evaluate the success criterion on an environment ``info`` dict.

    ``success`` is the criterion reported in Ref. [1]: R2 >= 0.95,
    NRMSE <= 5 % and peak pressure inside the admissible band.
    ``all_constraints`` additionally requires the peak-temperature and the
    burn-completeness / monotonicity constraints to be satisfied.
    """
    r2_ok = bool(info["r2"] >= SUCCESS_R2)
    nrmse_ok = bool(info["nrmse"] <= SUCCESS_NRMSE_PCT)
    peak_ok = bool(info["peak_penalty"] <= 1e-3)
    temp_ok = bool(info["tb_penalty"] <= 0.0)
    burn_ok = bool(info["burn_penalty"] <= 0.05)
    return {
        "r2_ok": r2_ok, "nrmse_ok": nrmse_ok, "peak_pressure_ok": peak_ok,
        "peak_temperature_ok": temp_ok, "burn_ok": burn_ok,
        "success": r2_ok and nrmse_ok and peak_ok,
        "all_constraints": r2_ok and nrmse_ok and peak_ok and temp_ok and burn_ok,
    }


def _params_dict(p) -> Dict[str, float]:
    return {name: float(getattr(p, _ATTR[name])) for name in PARAM_NAMES}


_LIMIT_FIELD = {"theta0": "theta0", "delta_theta": "dtheta", "a": "a", "m1": "m", "m2": "m",
                "lambda_w": "lambda_w", "k": "k", "ht_g": "ht_global", "ht_c": "ht_combustion"}


def effective_bounds(env: WiebeEnvV4) -> Dict[str, tuple]:
    """Admissible range of each parameter for the env's current curriculum stage."""
    return {n: env._effective_bounds(getattr(env.phys_limits, f)) for n, f in _LIMIT_FIELD.items()}


def params_at_bound(env: WiebeEnvV4, params: Dict[str, float], rel_tol: float = 1e-3) -> List[str]:
    """Names of parameters lying on (within ``rel_tol`` of) an admissible bound.

    A parameter pinned to a bound is not identified by the data; its smooth
    trajectory along an isoline must not be read as regularisation.
    """
    out = []
    for n, (lo, hi) in effective_bounds(env).items():
        tol = rel_tol * (hi - lo)
        if params[n] <= lo + tol:
            out.append(f"{n}:lower")
        elif params[n] >= hi - tol:
            out.append(f"{n}:upper")
    return out


class PolicyCalibrator:
    """Calibrate double Wiebe + heat-transfer parameters with a trained policy.

    Parameters
    ----------
    model:
        A loaded Stable-Baselines3 PPO or A2C model.
    metadata:
        The sidecar dictionary saved next to the model (``*.json``). Its
        ``normalization`` entry holds the training-set statistics.
    normalization:
        ``"frozen"`` (default) normalises new cases with the training-set
        statistics stored in ``metadata``. ``"legacy"`` recomputes the
        statistics from the cases being calibrated, which is how the
        evaluation scripts of Ref. [1] were run; use it only to reproduce
        the numbers published there.
    """

    def __init__(self, model, metadata: Dict[str, Any], normalization: str = "frozen"):
        if normalization not in {"frozen", "legacy"}:
            raise ValueError("normalization must be 'frozen' or 'legacy'")
        if normalization == "frozen" and "normalization" not in metadata:
            raise ValueError("metadata has no 'normalization' entry; "
                             "export it with scripts/export_normalization.py")
        self.model = model
        self.metadata = metadata
        self.normalization = normalization
        self.reward_weights = RewardWeights(**metadata.get("reward_weights", {}))

    @classmethod
    def from_pretrained(cls, model_path, metadata_path=None, normalization: str = "frozen",
                        device: str = "cpu") -> "PolicyCalibrator":
        from stable_baselines3 import A2C, PPO

        model_path = Path(model_path)
        metadata_path = Path(metadata_path) if metadata_path else model_path.with_suffix(".json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        algo = PPO if metadata.get("algorithm", "PPO").upper() == "PPO" else A2C
        # lr_schedule / clip_range are only needed for further training
        custom = {"lr_schedule": lambda _: 0.0, "learning_rate": 0.0, "clip_range": lambda _: 0.2}
        model = algo.load(str(model_path), device=device, custom_objects=custom)
        return cls(model, metadata, normalization)

    def _make_env(self, cases: Sequence[Dict[str, Any]]) -> WiebeEnvV4:
        norm = self.metadata["normalization"] if self.normalization == "frozen" else None
        env = WiebeEnvV4(
            wiebe_type=self.metadata.get("wiebe_type", "double"),
            seq_len=int(self.metadata.get("seq_len", 20)),
            cases=list(cases), data_dir="",
            phys_limits=PhysLimits(), reward_weights=self.reward_weights,
            # action bounds the policy was trained for (0.5 for v0.x models,
            # whose curriculum never left stage 1; 1.0 for v1.0.0 'synced')
            curriculum={"stages": [0], "schedule_steps": [0],
                        "spans": [float(self.metadata.get("eval_action_span", 0.5))],
                        "noise_scales": [0.0]},
            use_enhanced_obs=bool(self.metadata.get("use_enhanced_obs", True)),
            seed=0, normalization=norm,
        )
        return env

    def calibrate(self, cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        env = self._make_env(cases)
        out = []
        for i, case in enumerate(cases):
            obs, _ = env.reset(options={"case_idx": i})
            t0 = time.perf_counter()
            action, _ = self.model.predict(obs, deterministic=True)
            t_actor = (time.perf_counter() - t0) * 1e3
            params = env._action_to_params(np.asarray(action, dtype=np.float32))
            _, reward, _, _, info = env.step(action)
            res: Dict[str, Any] = {
                "case_id": case["case_id"],
                "params": _params_dict(params),
                "params_at_bound": params_at_bound(env, _params_dict(params)),
                "actor_time_ms": t_actor,
            }
            if "r2" in info:
                res["metrics"] = {k: float(info[k]) for k in
                                  ("r2", "nrmse", "pmax", "tmax", "xb_final", "ca50_pred")}
                res["metrics"]["reward"] = float(reward)
                res.update(success_flags(info))
            else:
                res["metrics"] = {"failure_reason": info.get("failure_reason", "unknown")}
                res["success"] = res["all_constraints"] = False
            out.append(res)
        return out

    def check_envelope(self, case: Dict[str, Any]) -> List[str]:
        """Return warnings for inputs outside the training envelope of the policy."""
        env = self.metadata.get("training_envelope", {})
        checks = [("speed_rpm", case.get("speed"), 1.0),
                  ("intake_pressure_kPa", case.get("intake_pressure"), 1e-3),
                  ("lambda", case.get("lambda"), 1.0),
                  ("intake_temperature_K", case.get("intake_temperature"), 1.0)]
        warnings = []
        for key, val, scale in checks:
            if key in env and val is not None:
                lo, hi = env[key]
                v = float(val) * scale
                if v < lo or v > hi:
                    warnings.append(f"{key}={v:.4g} outside training range [{lo:.4g}, {hi:.4g}]")
        geom = self.metadata.get("normalization", {}).get("reference_geometry")
        if geom:
            for key in ("bore", "stroke", "connecting_rod", "compression_ratio"):
                if key in case and abs(float(case[key]) - geom[key]) > 1e-6 * max(1.0, geom[key]):
                    warnings.append(f"{key}={case[key]} differs from the training engine ({geom[key]})")
        return warnings
