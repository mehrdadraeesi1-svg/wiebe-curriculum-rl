"""
Train PPO or A2C (double Wiebe) on a split produced by scripts/make_splits.py.

    python scripts/train_on_split.py --algorithm PPO \
        --data-dir path/to/json_cases \
        --split-file splits/split_speed_isoline_holdout.json \
        --output-dir runs/ppo_speed_isoline/seed_42 --seed 42

Curriculum modes (``--curriculum``)
-----------------------------------
synced (default, v1.0.0)
    Three stages, alpha = [0.50, 0.75, 1.00], beta = [0.30, 0.65, 1.00],
    switched at 100 000 and 200 000 *total* environment steps
    (CurriculumSyncCallback). Model selection and evaluation use the
    final-stage bounds.
legacy
    Behaviour of v0.x used for the results of Ref. [1]: the stage counter is
    per sub-environment, so with n_envs = 4 and 300 000 steps training never
    leaves stage 1 (alpha = 0.50, beta = 0.30). Use only for reproduction.
fixed-final
    No curriculum: final-stage bounds and noise from the first step.

Outputs: models/best/best_model.zip, models/<prefix>_final.zip, a sidecar
``models/best/best_model.json`` (training-set normalisation, action span,
training envelope -- everything src.calibrate.PolicyCalibrator needs),
data_splits.json and training_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import __version__  # noqa: E402
from src.utils.data_loader import WiebeDataLoader  # noqa: E402
from src.environment.wiebe_env import (  # noqa: E402
    WiebeEnvV4,
    RewardWeights,
    PhysLimits,
)

from stable_baselines3 import PPO, A2C  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from src.curriculum import (  # noqa: E402
    DEFAULT_SCHEDULE,
    LEGACY_EFFECTIVE_SCHEDULE,
    CurriculumSyncCallback,
)

logger = logging.getLogger(__name__)


DEFAULT_HYPERPARAMS: Dict[str, float] = {
    "total_timesteps": 300_000,
    "n_envs": 4,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "checkpoint_freq": 10_000,
    "eval_freq": 5_000,
    "w_nrmse": 1.0,
    "w_r2": 0.2,
    "w_peakP": 0.5,
    "w_ca50": 0.3,
    "w_tb": 0.1,
    "w_xb": 0.2,
}


def load_cases_by_split(
    data_dir: str, split_file: Path
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    with open(split_file, "r") as f:
        split_payload = json.load(f)

    logger.info("Loaded split file: %s", split_file)
    logger.info("  Strategy: %s", split_payload.get("strategy", "unknown"))
    logger.info("  |train|=%d, |val|=%d, |test|=%d",
                len(split_payload["train"]),
                len(split_payload["val"]),
                len(split_payload["test"]))

    loader = WiebeDataLoader(data_dir)
    all_cases = loader.load_all_cases()
    id_to_case = {c["case_id"]: c for c in all_cases}

    def _fetch(ids: List[str], bucket: str) -> List[Dict]:
        missing = [cid for cid in ids if cid not in id_to_case]
        if missing:
            raise KeyError(
                f"{len(missing)} case_ids in the {bucket} split are not "
                f"present in {data_dir}. First missing: {missing[:3]}"
            )
        return [id_to_case[cid] for cid in ids]

    train = _fetch(split_payload["train"], "train")
    val = _fetch(split_payload["val"], "val")
    test = _fetch(split_payload["test"], "test")

    all_assigned = (set(split_payload["train"]) | set(split_payload["val"])
                    | set(split_payload["test"]))
    if len(all_assigned) != (len(train) + len(val) + len(test)):
        raise RuntimeError("Split file has overlapping case_ids across sets.")

    metadata = {
        "strategy": split_payload.get("strategy", "unknown"),
        "diagnostics": split_payload.get("diagnostics", {}),
        "source_file": str(split_file),
    }
    return train, val, test, metadata


def curriculum_configs(mode: str, schedule: Dict = None) -> Tuple[Dict, Dict, float, float]:
    """Return (train_cfg, eval_cfg, eval_action_span, eval_noise_scale)."""
    schedule = schedule or DEFAULT_SCHEDULE
    if mode == "synced":
        train = {"stages": list(range(len(schedule["schedule_steps"]))),
                 "schedule_steps": list(schedule["schedule_steps"]),
                 "spans": list(schedule["spans"]), "noise_scales": list(schedule["noise_scales"])}
        span = train["spans"][-1]
    elif mode == "legacy":
        # exactly what v0.x passed; the per-env counter keeps it in stage 1
        train = {"stages": [0, 1, 2], "schedule_steps": [0, 100_000, 200_000]}
        span = LEGACY_EFFECTIVE_SCHEDULE["spans"][0]
    elif mode == "fixed-final":
        train = {"stages": [0], "schedule_steps": [0],
                 "spans": [DEFAULT_SCHEDULE["spans"][-1]],
                 "noise_scales": [DEFAULT_SCHEDULE["noise_scales"][-1]]}
        span = DEFAULT_SCHEDULE["spans"][-1]
    else:
        raise ValueError(f"unknown curriculum mode {mode!r}")
    if mode == "legacy":
        evalc = {"stages": [0, 1, 2], "schedule_steps": [0, 100_000, 200_000]}
    else:
        evalc = {"stages": [0], "schedule_steps": [0], "spans": [span], "noise_scales": [0.0]}
    return train, evalc, span, 0.0


def make_env(cases, data_dir, reward_weights, phys_limits,
             wiebe_type="double", use_enhanced_obs=True, rank=0, seed=0,
             curriculum_cfg=None, normalization=None):
    def _init():
        env = WiebeEnvV4(
            wiebe_type=wiebe_type,
            seq_len=20,
            cases=cases,
            data_dir=data_dir,
            phys_limits=phys_limits,
            reward_weights=reward_weights,
            curriculum=curriculum_cfg,
            use_enhanced_obs=use_enhanced_obs,
            seed=seed + rank,
            normalization=normalization,
        )
        env = Monitor(env)
        return env
    return _init


def train_on_split(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    algo_name = args.algorithm.upper()
    if algo_name not in {"PPO", "A2C"}:
        raise ValueError(f"--algorithm must be PPO or A2C, got {algo_name}")

    logger.info("=" * 70)
    logger.info("STAGE 2 TRAINING (v3)")
    logger.info("  Algorithm: %s (double Wiebe)", algo_name)
    logger.info("  Split file: %s", args.split_file)
    logger.info("  Seed: %d", args.seed)
    logger.info("  Output:     %s", output_dir)
    logger.info("=" * 70)

    train_cases, val_cases, test_cases, split_meta = load_cases_by_split(
        data_dir=args.data_dir, split_file=args.split_file,
    )

    with open(output_dir / "data_splits.json", "w") as f:
        json.dump({
            "strategy": split_meta["strategy"],
            "diagnostics": split_meta["diagnostics"],
            "train": [c["case_id"] for c in train_cases],
            "val": [c["case_id"] for c in val_cases],
            "test": [c["case_id"] for c in test_cases],
        }, f, indent=2)

    reward_weights = RewardWeights(
        w_nrmse=args.w_nrmse, w_r2=args.w_r2,
        w_peakP=args.w_peakP, w_ca50=args.w_ca50,
        w_tb=args.w_tb, w_xb=args.w_xb,
    )
    phys_limits = PhysLimits()

    env_kwargs = {
        "data_dir": args.data_dir,
        "reward_weights": reward_weights,
        "phys_limits": phys_limits,
        "wiebe_type": "double",
        "use_enhanced_obs": True,
        "seed": args.seed,
    }

    train_cfg, eval_cfg, eval_span, eval_noise = curriculum_configs(args.curriculum, args.schedule)
    # training-set normalisation, frozen and shared by train/eval envs and
    # saved with the model for inference on new data
    normalization = WiebeEnvV4(wiebe_type="double", cases=train_cases,
                               data_dir=args.data_dir).get_normalization()

    logger.info("Curriculum mode: %s", args.curriculum)
    logger.info("Creating %d parallel training environments...", args.n_envs)
    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_env(train_cases, rank=i, curriculum_cfg=train_cfg,
                     normalization=None if args.curriculum == "legacy" else normalization,
                     **env_kwargs)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([make_env(
            train_cases, rank=0, curriculum_cfg=train_cfg,
            normalization=None if args.curriculum == "legacy" else normalization,
            **env_kwargs)])

    eval_env = None
    if val_cases:
        eval_env = DummyVecEnv([make_env(
            val_cases, rank=100, curriculum_cfg=eval_cfg,
            normalization=None if args.curriculum == "legacy" else normalization,
            **env_kwargs)])

    try:
        import tensorboard  # noqa: F401
        formats = ["stdout", "csv", "tensorboard"]
    except ImportError:
        formats = ["stdout", "csv"]
    new_logger = configure(str(logs_dir), formats)

    common_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logs_dir) if "tensorboard" in formats else None,
        policy_kwargs={"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
    )

    if algo_name == "PPO":
        model = PPO(
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            **common_kwargs,
        )
    else:  # A2C
        model = A2C(
            n_steps=args.n_steps,
            **common_kwargs,
        )

    model.set_logger(new_logger)

    split_slug = split_meta["strategy"].split("[")[0]
    name_prefix = f"{algo_name.lower()}_double_{split_slug}_seed{args.seed}"
    sync_cb = CurriculumSyncCallback(verbose=1) if args.curriculum == "synced" else None
    callbacks = [] if sync_cb is None else [sync_cb]
    callbacks += [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(models_dir),
            name_prefix=name_prefix,
        )
    ]
    if eval_env is not None:
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=str(models_dir / "best"),
            log_path=str(logs_dir / "eval"),
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
        ))

    logger.info("Begin training (%d timesteps)...", args.total_timesteps)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=args.progress_bar,
        )
        final_path = models_dir / f"{name_prefix}_final"
        model.save(final_path)
        logger.info("Training completed. Final model: %s", final_path)
    except KeyboardInterrupt:
        interrupt_path = models_dir / f"{name_prefix}_interrupted"
        model.save(interrupt_path)
        logger.warning("Interrupted. Partial model: %s", interrupt_path)
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    metadata = {
        "stage": "train_on_split",
        "algorithm": algo_name,
        "wiebe_type": "double",
        "split_strategy": split_meta["strategy"],
        "split_diagnostics": split_meta["diagnostics"],
        "n_train": len(train_cases),
        "n_val": len(val_cases),
        "n_test": len(test_cases),
        "hyperparameters": {
            k: getattr(args, k) for k in DEFAULT_HYPERPARAMS if hasattr(args, k)
        },
        "reward_weights": {
            "w_nrmse": reward_weights.w_nrmse,
            "w_r2": reward_weights.w_r2,
            "w_peakP": reward_weights.w_peakP,
            "w_ca50": reward_weights.w_ca50,
            "w_tb": reward_weights.w_tb,
            "w_xb": reward_weights.w_xb,
        },
        "seed": args.seed,
        "curriculum_mode": args.curriculum,
        "curriculum": train_cfg,
        "curriculum_stage_history": (sync_cb.stage_history if sync_cb else None),
        "eval_action_span": eval_span,
        "package_version": __version__,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    sidecar = build_sidecar(metadata, normalization, train_cases)
    for zip_path in [models_dir / "best" / "best_model.zip",
                     models_dir / f"{name_prefix}_final.zip"]:
        if zip_path.exists():
            with open(zip_path.with_suffix(".json"), "w") as f:
                json.dump(sidecar, f, indent=2)


def build_sidecar(metadata: Dict, normalization: Dict, train_cases: List[Dict]) -> Dict:
    """Everything PolicyCalibrator needs to calibrate new cases."""
    def rng(key, scale=1.0):
        vals = [float(c[key]) * scale for c in train_cases]
        return [min(vals), max(vals)]
    return {
        "algorithm": metadata["algorithm"],
        "wiebe_type": "double",
        "seq_len": 20,
        "use_enhanced_obs": True,
        "split_strategy": metadata["split_strategy"],
        "n_train": metadata["n_train"],
        "reward_weights": metadata["reward_weights"],
        "curriculum_mode": metadata["curriculum_mode"],
        "eval_action_span": metadata["eval_action_span"],
        "training_envelope": {
            "speed_rpm": rng("speed"),
            "intake_pressure_kPa": rng("intake_pressure", 1e-3),
            "lambda": rng("lambda"),
            "intake_temperature_K": rng("intake_temperature"),
        },
        "normalization": normalization,
        "package_version": metadata["package_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "A2C", "ppo", "a2c"])
    parser.add_argument("--data-dir", type=str, default="json_cases")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", choices=["synced", "legacy", "fixed-final"],
                        default="synced",
                        help="see module docstring; 'legacy' reproduces Ref. [1]")

    for key, default in DEFAULT_HYPERPARAMS.items():
        name = "--" + key.replace("_", "-")
        typ = type(default)
        parser.add_argument(name, type=typ, default=default)

    parser.add_argument("--progress-bar", action="store_true",
                        help="show a progress bar (needs tqdm and rich)")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML file (see config/default_config.yaml); CLI flags override it")
    parser.add_argument("--log-level", type=str, default="INFO")

    pre, _ = parser.parse_known_args()
    args_schedule = None
    if pre.config is not None:
        import yaml
        cfg = yaml.safe_load(pre.config.read_text())
        flat = dict(cfg.get("hyperparameters", {}))
        flat.update(cfg.get("reward_weights", {}))
        if "algorithm" in cfg:
            flat["algorithm"] = cfg["algorithm"]
        cur = cfg.get("curriculum", {})
        if "mode" in cur:
            flat["curriculum"] = cur["mode"]
        if {"schedule_steps", "spans", "noise_scales"} <= set(cur):
            args_schedule = {k: cur[k] for k in ("schedule_steps", "spans", "noise_scales")}
        parser.set_defaults(**flat)
    args = parser.parse_args()
    args.schedule = args_schedule

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_on_split(args)


if __name__ == "__main__":
    main()
