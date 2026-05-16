"""
Wiebe Environment v4 - Enhanced Observation Space
- Includes: Pressure, dP/dθ, Cumulative Heat Release, Previous Action
- Improved Double Wiebe support
- Better curriculum learning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_loader import WiebeDataLoader
from src.models.wiebe_functions import single_wiebe, double_wiebe
from src.utils.thermodynamics import (
    EngineGeometry,
    EngineOperatingConditions,
    GasProperties,
    simulate_pressure_trace,
    calculate_normalized_rmse,
    calculate_rms_error,
    calculate_r_squared,
    compute_CA50,
)

logger = logging.getLogger(__name__)


@dataclass
class PhysLimits:
    """Physical limits for Wiebe parameters and thermodynamic constraints."""

    theta0: Tuple[float, float] = (-20.0, 40.0)
    dtheta: Tuple[float, float] = (5.0, 120.0)
    m: Tuple[float, float] = (0.5, 4.0)
    a: Tuple[float, float] = (0.1, 8.0)
    lambda_w: Tuple[float, float] = (0.05, 0.95)
    
    # Double Wiebe specific limits
    k: Tuple[float, float] = (1.0, 4.0)
    
    # Heat Transfer Scaling Limits
    ht_global: Tuple[float, float] = (0.5, 3.0) 
    ht_combustion: Tuple[float, float] = (0.5, 3.0)

    pmax_soft: Tuple[float, float] = (2.0e5, 2.0e7)
    tb_max_soft: float = 4000.0
    xb_target: float = 0.98


@dataclass
class RewardWeights:
    """Weights for composite reward components."""
    w_nrmse: float = 1.0
    w_r2: float = 0.1
    w_peakP: float = 0.2
    w_ca50: float = 0.2
    w_tb: float = 0.1
    w_xb: float = 0.2


@dataclass
class WiebeParams:
    """Container for decoded Wiebe parameters + Heat Transfer scales."""
    theta0: float
    delta_theta: float
    a: float
    lambda_w: float
    # Wiebe shape parameters
    m1: float         # Used as 'm' for single wiebe
    m2: float = 2.0   # Used only for double wiebe
    k: float = 2.0    # Used only for double wiebe
    
    # Heat Transfer
    ht_scale_global: float = 1.0
    ht_scale_combustion: float = 1.0


class WiebeEnvV4(gym.Env):
    """
    Gymnasium environment v4 with enhanced observation space.
    
    New features:
    - Cumulative heat release in observation
    - Previous action for temporal consistency
    - Improved normalization
    - Better curriculum strategy
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        wiebe_type: str = "single",
        seq_len: int = 20,
        cases: Optional[List[Dict[str, Any]]] = None,
        data_dir: str = "json_cases",
        phys_limits: Optional[PhysLimits] = None,
        reward_weights: Optional[RewardWeights] = None,
        curriculum: Optional[Dict[str, Any]] = None,
        randomization_base: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        ca50_penalty: bool = True,
        use_enhanced_obs: bool = True,  # NEW: toggle for enhanced observation
    ) -> None:
        super().__init__()

        self.wiebe_type = wiebe_type.lower()
        if self.wiebe_type not in {"single", "double"}:
            raise ValueError("wiebe_type must be 'single' or 'double'")

        self.seq_len = seq_len
        self.phys_limits = phys_limits or PhysLimits()
        self.reward_weights = reward_weights or RewardWeights()
        self.ca50_penalty = ca50_penalty
        self.use_enhanced_obs = use_enhanced_obs

        self.randomization_base = randomization_base or {
            "speed": 200.0,
            "lambda": 0.08,
            "intake_pressure": 2.0e4,
            "intake_temperature": 20.0,
        }

        self.curriculum_cfg = curriculum or {"stages": [0], "schedule_steps": [0]}
        schedule = list(self.curriculum_cfg.get("schedule_steps", [0]))
        if len(schedule) == 0:
            schedule = [0]
        self.curriculum_schedule = sorted(int(s) for s in schedule)
        self.stage_count = max(len(self.curriculum_schedule), 1)
        self.curriculum_stage = 0
        self.curriculum_spans = np.linspace(0.5, 1.0, self.stage_count)
        self.randomization_scales = np.linspace(0.3, 1.0, self.stage_count)
        self.global_step = 0

        self.data_dir = data_dir
        self.cases = cases if cases is not None else self._load_cases(data_dir)
        if len(self.cases) == 0:
            raise ValueError("WiebeEnvV4 requires at least one case")

        self.dataset_stats = self._compute_dataset_stats(self.cases)
        self.pressure_scale, self.dp_scale = self._compute_pressure_scales(self.cases)
        self.reference_geometry = self._infer_reference_geometry(self.cases)
        self.external_traces = self._load_external_traces(Path("data/pressure_traces"))
        self.external_mix_prob = 0.25 if self.external_traces else 0.0

        # === ENHANCED OBSERVATION SPACE ===
        if self.use_enhanced_obs:
            # seq_len * 4 (P, dP, theta, CHR) + 4 (operating) + n_actions (prev_action)
            if self.wiebe_type == 'single':
                self.n_actions = 7  # 5 Wiebe + 2 HT
            else:
                self.n_actions = 9  # 7 Wiebe + 2 HT
            
            obs_dim = self.seq_len * 4 + 4 + self.n_actions
        else:
            # Legacy: seq_len * 3 + 4
            if self.wiebe_type == 'single':
                self.n_actions = 7
            else:
                self.n_actions = 9
            obs_dim = self.seq_len * 3 + 4
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )

        self.gas_props = GasProperties()
        self.current_case: Optional[Dict[str, Any]] = None
        self.last_info: Dict[str, Any] = {}
        self.previous_action: Optional[np.ndarray] = None  # NEW: store previous action
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Data loading (same as v3)
    # ------------------------------------------------------------------
    def _load_cases(self, data_dir: str) -> List[Dict[str, Any]]:
        loader = WiebeDataLoader(data_dir)
        return loader.load_all_cases()

    def _compute_dataset_stats(self, cases: List[Dict[str, Any]]) -> Dict[str, float]:
        speeds = [c.get("speed", 2000.0) for c in cases]
        lambdas = [c.get("lambda", 1.0) for c in cases]
        pressures = [c.get("intake_pressure", 1e5) for c in cases]
        temps = [c.get("intake_temperature", 300.0) for c in cases]

        return {
            "speed_mean": float(np.mean(speeds)),
            "speed_std": float(np.std(speeds)) + 1e-6,
            "lambda_mean": float(np.mean(lambdas)),
            "lambda_std": float(np.std(lambdas)) + 1e-6,
            "p_mean": float(np.mean(pressures)),
            "p_std": float(np.std(pressures)) + 1e-6,
            "t_mean": float(np.mean(temps)),
            "t_std": float(np.std(temps)) + 1e-6,
        }

    def _compute_pressure_scales(self, cases: List[Dict[str, Any]]) -> Tuple[float, float]:
        all_p, all_dp = [], []
        for c in cases:
            p = np.asarray(c["pressure"])
            theta = np.asarray(c["crank_angle"])
            dp = np.gradient(p, theta)
            all_p.append(np.max(p))
            all_dp.append(np.max(np.abs(dp)))

        p_scale = float(np.median(all_p))
        dp_scale = float(np.median(all_dp))
        return max(p_scale, 1e5), max(dp_scale, 1e3)

    def _infer_reference_geometry(self, cases: List[Dict[str, Any]]) -> Dict[str, float]:
        bore = np.median([c.get("bore", 0.086) for c in cases])
        stroke = np.median([c.get("stroke", 0.086) for c in cases])
        rod = np.median([c.get("connecting_rod", 0.143) for c in cases])
        cr = np.median([c.get("compression_ratio", 10.5) for c in cases])
        return {"bore": float(bore), "stroke": float(stroke), 
                "connecting_rod": float(rod), "compression_ratio": float(cr)}

    def _load_external_traces(self, path: Path) -> List[Dict[str, Any]]:
        # Placeholder for external pressure traces
        return []

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        # Update curriculum stage based on global step
        for stage_idx, threshold in enumerate(self.curriculum_schedule):
            if self.global_step >= threshold:
                self.curriculum_stage = stage_idx

        self.current_case = self._sample_case(options)
        
        # Reset previous action to zeros
        self.previous_action = np.zeros(self.n_actions, dtype=np.float32)
        
        obs = self._build_observation(self.current_case)
        info = {
            "case_id": self.current_case["case_id"],
            "curriculum_stage": self.curriculum_stage,
        }
        self.last_info = info
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_case is None:
            raise RuntimeError("reset() must be called before step()")

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        params = self._action_to_params(action)
        result = self._simulate_episode(self.current_case, params)

        self.global_step += 1

        if result.get("failed", False):
            reward = -10.0
            info = result.get("info", {})
            info["case_id"] = self.current_case["case_id"]
            info["curriculum_stage"] = self.curriculum_stage
            self.last_info = info
            
            # Store action even on failure
            self.previous_action = action
            
            obs = self._build_observation(self.current_case)
            return obs, reward, True, False, info

        reward = result["reward"]
        info = result["info"]
        info["case_id"] = self.current_case["case_id"]
        info["curriculum_stage"] = self.curriculum_stage
        self.last_info = info
        
        # Store current action for next observation
        self.previous_action = action
        
        obs = self._build_observation(self.current_case)
        return obs, reward, True, False, info

    # ------------------------------------------------------------------
    # Observation building with ENHANCED FEATURES
    # ------------------------------------------------------------------
    def _build_observation(self, case: Dict[str, Any]) -> np.ndarray:
        """
        Build observation with enhanced features:
        - Pressure window
        - Pressure derivative (dP/dθ)
        - Crank angle (normalized)
        - Cumulative Heat Release (NEW)
        - Operating conditions
        - Previous action (NEW)
        """
        theta = np.asarray(case["crank_angle"], dtype=np.float64)
        pressure = np.asarray(case["pressure"], dtype=np.float64)
        dp = np.gradient(pressure, theta)

        # Compute cumulative heat release (simplified approximation)
        # CHR ≈ integral of (P * dV), normalized by peak
        V = self._compute_volume_approx(theta)
        dV = np.gradient(V, theta)
        heat_release = np.cumsum(pressure * dV)
        chr = heat_release / (np.max(np.abs(heat_release)) + 1e-6)  # normalized

        if theta.size < self.seq_len:
            pad = self.seq_len - theta.size
            theta = np.pad(theta, (0, pad), "edge")
            pressure = np.pad(pressure, (0, pad), "edge")
            dp = np.pad(dp, (0, pad), "edge")
            chr = np.pad(chr, (0, pad), "edge")

        idx = self._select_window_indices(pressure)
        theta_norm = 2.0 * (theta - theta[0]) / (theta[-1] - theta[0] + 1e-6) - 1.0

        window = []
        if self.use_enhanced_obs:
            # Enhanced: P, dP, theta, CHR
            for i in idx:
                window.extend([
                    float(pressure[i] / self.pressure_scale),
                    float(dp[i] / self.dp_scale),
                    float(theta_norm[i]),
                    float(chr[i])  # NEW
                ])
        else:
            # Legacy: P, dP, theta
            for i in idx:
                window.extend([
                    float(pressure[i] / self.pressure_scale),
                    float(dp[i] / self.dp_scale),
                    float(theta_norm[i]),
                ])

        cond = self._operating_obs(case)
        
        if self.use_enhanced_obs:
            # Add previous action to observation
            prev_action_list = self.previous_action.tolist() if self.previous_action is not None else [0.0] * self.n_actions
            obs = np.asarray(window + cond + prev_action_list, dtype=np.float32)
        else:
            obs = np.asarray(window + cond, dtype=np.float32)
            
        return obs

    def _compute_volume_approx(self, theta: np.ndarray) -> np.ndarray:
        """Approximate cylinder volume for CHR calculation."""
        # Simple slider-crank approximation
        theta_rad = np.deg2rad(theta)
        bore = self.reference_geometry["bore"]
        stroke = self.reference_geometry["stroke"]
        rod = self.reference_geometry["connecting_rod"]
        cr = self.reference_geometry["compression_ratio"]
        
        cyl_area = np.pi * (bore / 2.0)**2
        crank_radius = stroke / 2.0
        clearance_vol = (cyl_area * stroke) / (cr - 1.0)
        
        # Piston position
        x = crank_radius * ((1.0 - np.cos(theta_rad)) + 
                           (crank_radius / (4.0 * rod)) * (1.0 - np.cos(2.0 * theta_rad)))
        
        V = clearance_vol + cyl_area * x
        return V

    def _operating_obs(self, case: Dict[str, Any]) -> List[float]:
        stats = self.dataset_stats
        speed = (case.get("speed", stats["speed_mean"]) - stats["speed_mean"]) / stats["speed_std"]
        lam = (case.get("lambda", stats["lambda_mean"]) - stats["lambda_mean"]) / stats["lambda_std"]
        p_int = (case.get("intake_pressure", stats["p_mean"]) - stats["p_mean"]) / stats["p_std"]
        t_int = (case.get("intake_temperature", stats["t_mean"]) - stats["t_mean"]) / stats["t_std"]
        return [float(speed), float(lam), float(p_int), float(t_int)]

    def _select_window_indices(self, pressure: np.ndarray) -> np.ndarray:
        peak_idx = int(np.argmax(pressure))
        half = self.seq_len // 2
        start = np.clip(peak_idx - half, 0, max(len(pressure) - self.seq_len, 0))
        return np.arange(start, start + self.seq_len)

    def _sample_case(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # Same as v3
        if options and "case_idx" in options:
            return self.cases[options["case_idx"]]
        
        case_idx = self._rng.integers(0, len(self.cases))
        case = self.cases[case_idx].copy()
        
        # Apply randomization based on curriculum
        scale = float(self.randomization_scales[self.curriculum_stage])
        for key in ["speed", "lambda", "intake_pressure", "intake_temperature"]:
            if key in self.randomization_base:
                base_val = case.get(key, self.dataset_stats.get(f"{key}_mean", 0))
                noise_std = self.randomization_base[key] * scale
                noise = self._rng.normal(0, noise_std)
                case[key] = float(base_val + noise)
        
        return case

    # ------------------------------------------------------------------
    # Physics helpers (same as v3)
    # ------------------------------------------------------------------
    def _effective_bounds(self, bounds: Tuple[float, float]) -> Tuple[float, float]:
        minimum, maximum = bounds
        center = 0.5 * (minimum + maximum)
        span = 0.5 * (maximum - minimum) * float(self.curriculum_spans[self.curriculum_stage])
        eff_min = max(minimum, center - span)
        eff_max = min(maximum, center + span)
        if eff_min >= eff_max:
            eff_min, eff_max = minimum, maximum
        return eff_min, eff_max

    def _scale_linear(self, val: float, bounds: Tuple[float, float]) -> float:
        eff_min, eff_max = self._effective_bounds(bounds)
        return float(eff_min + 0.5 * (val + 1.0) * (eff_max - eff_min))

    def _scale_log(self, val: float, bounds: Tuple[float, float]) -> float:
        eff_min, eff_max = self._effective_bounds(bounds)
        eff_min = max(eff_min, 1e-3)
        eff_max = max(eff_max, eff_min + 1e-3)
        log_min = np.log(eff_min)
        log_max = np.log(eff_max)
        return float(np.exp(log_min + 0.5 * (val + 1.0) * (log_max - log_min)))

    def _scale_sigmoid(self, val: float, bounds: Tuple[float, float]) -> float:
        eff_min, eff_max = self._effective_bounds(bounds)
        sig = 1.0 / (1.0 + np.exp(-3.0 * val))
        return float(eff_min + sig * (eff_max - eff_min))

    def _action_to_params(self, action: np.ndarray) -> WiebeParams:
        # Same as v3
        theta0 = self._scale_linear(float(action[0]), self.phys_limits.theta0)
        delta = self._scale_log(float(action[1]), self.phys_limits.dtheta)
        a_val = self._scale_log(float(action[3]), self.phys_limits.a)
        lambda_w = self._scale_sigmoid(float(action[4]), self.phys_limits.lambda_w)
        
        if self.wiebe_type == 'single':
            m1 = self._scale_log(float(action[2]), self.phys_limits.m)
            m2 = 2.0
            k = 2.0
            ht_g = self._scale_linear(float(action[5]), self.phys_limits.ht_global)
            ht_c = self._scale_linear(float(action[6]), self.phys_limits.ht_combustion)
        else:
            m1 = self._scale_log(float(action[2]), self.phys_limits.m)
            m2 = self._scale_log(float(action[5]), self.phys_limits.m)
            k = self._scale_linear(float(action[6]), self.phys_limits.k)
            ht_g = self._scale_linear(float(action[7]), self.phys_limits.ht_global)
            ht_c = self._scale_linear(float(action[8]), self.phys_limits.ht_combustion)
            
        return WiebeParams(theta0, delta, a_val, lambda_w, m1, m2, k, ht_g, ht_c)

    def _build_mass_fraction(self, theta: np.ndarray, params: WiebeParams) -> Tuple[np.ndarray, float]:
        if self.wiebe_type == "single":
            xb = single_wiebe(theta, params.theta0, params.delta_theta, params.a, params.m1)
            eta = np.clip(0.85 + 0.15 * params.lambda_w, 0.85, 1.0)
        else:
            xb = double_wiebe(
                theta, params.theta0, params.delta_theta, params.a,
                params.m1, params.m2,
                np.clip(params.lambda_w, *self.phys_limits.lambda_w),
                params.k,
            )
            eta = np.clip(0.85 + 0.1 * params.lambda_w, 0.85, 1.0)
        xb = np.clip(xb, 0.0, 1.0)
        return xb, float(eta)

    def _simulate_episode(self, case: Dict[str, Any], params: WiebeParams) -> Dict[str, Any]:
        theta = np.asarray(case["crank_angle"], dtype=np.float64)
        pressure_true = np.asarray(case["pressure"], dtype=np.float64)

        geometry = EngineGeometry(
            bore=float(case.get("bore", self.reference_geometry["bore"])),
            stroke=float(case.get("stroke", self.reference_geometry["stroke"])),
            connecting_rod=float(case.get("connecting_rod", self.reference_geometry["connecting_rod"])),
            compression_ratio=float(case.get("compression_ratio", self.reference_geometry["compression_ratio"])),
        )
        operating = EngineOperatingConditions(
            speed=float(case.get("speed", self.dataset_stats["speed_mean"])),
            lambda_=float(case.get("lambda", self.dataset_stats["lambda_mean"])),
            intake_pressure=float(case.get("intake_pressure", self.dataset_stats["p_mean"])),
            intake_temperature=float(case.get("intake_temperature", self.dataset_stats["t_mean"])),
        )

        xb, eta = self._build_mass_fraction(theta, params)

        try:
            P_pred, T_pred, _ = simulate_pressure_trace(
                theta, xb, geometry, operating, self.gas_props, 
                eta=eta,
                ht_scale_global=params.ht_scale_global,
                ht_scale_combustion=params.ht_scale_combustion
            )
        except Exception as exc:
            logger.warning("Simulation failure: %s", exc)
            info = {"failure_reason": str(exc)}
            return {"failed": True, "info": info}

        if not np.all(np.isfinite(P_pred)) or not np.all(np.isfinite(T_pred)):
            info = {"failure_reason": "non-finite simulation output"}
            return {"failed": True, "info": info}

        reward, info = self._compute_reward(case, pressure_true, P_pred, T_pred, xb, params)
        return {"failed": False, "reward": reward, "info": info}

    def _compute_reward(
        self,
        case: Dict[str, Any],
        pressure_true: np.ndarray,
        pressure_pred: np.ndarray,
        temp_pred: np.ndarray,
        xb: np.ndarray,
        params: WiebeParams,
    ) -> Tuple[float, Dict[str, Any]]:
        nrmse = calculate_normalized_rmse(pressure_pred, pressure_true)
        rmse = calculate_rms_error(pressure_pred, pressure_true)
        r2 = calculate_r_squared(pressure_pred, pressure_true)
        peak_pressure = float(np.max(pressure_pred))
        peak_temp = float(np.max(temp_pred))
        xb_final = float(xb[-1])
        monotonic_violation = float(max(0.0, -np.min(np.diff(xb))))

        target_ca50 = case.get("target_ca50_deg") or case.get("ca50_deg")
        ca50_pred = compute_CA50(case["crank_angle"], xb)
        ca50_error = np.nan
        if target_ca50 is not None:
            ca50_error = ca50_pred - float(target_ca50)

        pmin, pmax = self.phys_limits.pmax_soft
        if peak_pressure < pmin:
            peak_penalty = (pmin - peak_pressure) / pmin
        elif peak_pressure > pmax:
            peak_penalty = (peak_pressure - pmax) / pmax
        else:
            peak_penalty = 0.0

        temp_penalty = max(0.0, (peak_temp - self.phys_limits.tb_max_soft) / self.phys_limits.tb_max_soft)
        burn_penalty = abs(xb_final - self.phys_limits.xb_target) + monotonic_violation

        components: Dict[str, float] = {}
        components["nrmse"] = -self.reward_weights.w_nrmse * (nrmse / 100.0)
        components["r2"] = self.reward_weights.w_r2 * float(r2)
        components["peak"] = -self.reward_weights.w_peakP * peak_penalty
        components["temp"] = -self.reward_weights.w_tb * temp_penalty
        components["xb"] = -self.reward_weights.w_xb * burn_penalty

        if self.ca50_penalty and not np.isnan(ca50_error):
            components["ca50"] = -self.reward_weights.w_ca50 * (abs(ca50_error) / 50.0)
        else:
            components["ca50"] = 0.0

        reward = float(sum(components.values()))
        safety_violation = peak_penalty > 0.0 or temp_penalty > 0.0 or burn_penalty > 0.05

        info = {
            "reward_components": components,
            "nrmse": float(nrmse),
            "rmse": float(rmse),
            "r2": float(r2),
            "pmax": peak_pressure,
            "tmax": peak_temp,
            "xb_final": xb_final,
            "tb_penalty": temp_penalty,
            "peak_penalty": peak_penalty,
            "burn_penalty": burn_penalty,
            "ca50_pred": float(ca50_pred) if np.isfinite(ca50_pred) else float("nan"),
            "ca50_target": float(target_ca50) if target_ca50 is not None else float("nan"),
            "params": params.__dict__,
            "safety_violation": safety_violation,
        }
        return reward, info

# Alias for backward compatibility
WiebeEnvironment = WiebeEnvV4
