"""
Stage 2 -- Retrain PPO-Double on a specified split.

This script reuses the entire training stack of train_rl_v4_enhanced.py
(same environment, same hyperparameters, same reward) but replaces the
internal random split with one loaded from a JSON file produced by
stage 1. This keeps comparability with the original submission results
while allowing the reviewer's OOD concerns to be addressed.

Run from the project root:

    python -m scripts.stage2_train_ppo_on_split \
        --data-dir json_cases \
        --split-file response_revision/stage1/split_extrapolation_high_load.json \
        --output-dir response_revision/stage2/ppo_double_extrapolation_high_load

Reproducibility
---------------
Hyperparameters are hardcoded here to match train_rl_v4_enhanced.py defaults
so that the only intentional difference between runs is the split. Any
override is logged clearly in the run's metadata.json.
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

# These imports come from the original project code, not from stage 1.
from src.utils.data_loader import WiebeDataLoader  # noqa: E402
from src.environment.wiebe_env import (  # noqa: E402
    WiebeEnvV4,
    RewardWeights,
    PhysLimits,
)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters -- match train_rl_v4_enhanced.py defaults exactly
# ---------------------------------------------------------------------------

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
    # Reward weights (from training script defaults)
    "w_nrmse": 1.0,
    "w_r2": 0.2,
    "w_peakP": 0.5,
    "w_ca50": 0.3,
    "w_tb": 0.1,
    "w_xb": 0.2,
}


# ---------------------------------------------------------------------------
# Split loading
# ---------------------------------------------------------------------------

def load_cases_by_split(
    data_dir: str, split_file: Path
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Load all cases, then partition them by the case_id lists in the
    split file. Returns (train, val, test, split_metadata).

    Raises if any case_id in the split file is missing from the dataset,
    or if the dataset has unassigned cases (both are integrity errors).
    """
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

    # Sanity: no case assigned to multiple buckets
    all_assigned = set(split_payload["train"]) | set(split_payload["val"]) \
        | set(split_payload["test"])
    if len(all_assigned) != (len(train) + len(val) + len(test)):
        raise RuntimeError(
            "Split file contains overlapping case_ids across sets."
        )

    metadata = {
        "strategy": split_payload.get("strategy", "unknown"),
        "diagnostics": split_payload.get("diagnostics", {}),
        "source_file": str(split_file),
    }
    return train, val, test, metadata


# ---------------------------------------------------------------------------
# Environment factory (identical to train_rl_v4_enhanced.py)
# ---------------------------------------------------------------------------

def make_env(cases, data_dir, reward_weights, phys_limits,
             wiebe_type="double", use_enhanced_obs=True, rank=0, seed=0):
    def _init():
        curriculum_cfg = {
            "stages": [0, 1, 2],
            "schedule_steps": [0, 100_000, 200_000],
        }
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
        )
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_on_split(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 2 TRAINING")
    logger.info("  Split file: %s", args.split_file)
    logger.info("  Output:     %s", output_dir)
    logger.info("=" * 70)

    train_cases, val_cases, test_cases, split_meta = load_cases_by_split(
        data_dir=args.data_dir, split_file=args.split_file,
    )

    # Persist the split alongside the trained model so downstream eval
    # can reload the exact same assignment.
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

    logger.info("")
    logger.info("Reward weights: nrmse=%.2f r2=%.2f peakP=%.2f ca50=%.2f "
                "tb=%.2f xb=%.2f",
                reward_weights.w_nrmse, reward_weights.w_r2,
                reward_weights.w_peakP, reward_weights.w_ca50,
                reward_weights.w_tb, reward_weights.w_xb)

    env_kwargs = {
        "data_dir": args.data_dir,
        "reward_weights": reward_weights,
        "phys_limits": phys_limits,
        "wiebe_type": "double",
        "use_enhanced_obs": True,
        "seed": args.seed,
    }

    logger.info("Creating %d parallel training environments...", args.n_envs)
    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_env(train_cases, rank=i, **env_kwargs)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([make_env(train_cases, rank=0, **env_kwargs)])

    eval_env = None
    if val_cases:
        logger.info("Creating validation environment (%d cases)...",
                    len(val_cases))
        eval_env = DummyVecEnv([make_env(val_cases, rank=100, **env_kwargs)])

    new_logger = configure(str(logs_dir), ["stdout", "csv", "tensorboard"])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(logs_dir),
        policy_kwargs={
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        },
    )
    model.set_logger(new_logger)

    split_slug = split_meta["strategy"].split("[")[0]
    name_prefix = f"ppo_double_{split_slug}"
    callbacks = [
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

    logger.info("")
    logger.info("=" * 70)
    logger.info("BEGIN TRAINING (%d timesteps)", args.total_timesteps)
    logger.info("=" * 70)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
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
        "stage": "stage2_train_ppo_on_split",
        "algorithm": "PPO",
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
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata -> %s", output_dir / "training_metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Train PPO-Double on a fixed split produced by "
                     "stage 1."),
    )
    parser.add_argument("--data-dir", type=str, default="json_cases")
    parser.add_argument("--split-file", type=Path, required=True,
                        help="Path to a split_*.json file from stage 1.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where checkpoints, logs, and metadata go.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total-timesteps", type=int,
                        default=DEFAULT_HYPERPARAMS["total_timesteps"])
    parser.add_argument("--n-envs", type=int,
                        default=DEFAULT_HYPERPARAMS["n_envs"])
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_HYPERPARAMS["learning_rate"])
    parser.add_argument("--n-steps", type=int,
                        default=DEFAULT_HYPERPARAMS["n_steps"])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_HYPERPARAMS["batch_size"])
    parser.add_argument("--n-epochs", type=int,
                        default=DEFAULT_HYPERPARAMS["n_epochs"])
    parser.add_argument("--gamma", type=float,
                        default=DEFAULT_HYPERPARAMS["gamma"])
    parser.add_argument("--gae-lambda", type=float,
                        default=DEFAULT_HYPERPARAMS["gae_lambda"])
    parser.add_argument("--clip-range", type=float,
                        default=DEFAULT_HYPERPARAMS["clip_range"])
    parser.add_argument("--ent-coef", type=float,
                        default=DEFAULT_HYPERPARAMS["ent_coef"])
    parser.add_argument("--checkpoint-freq", type=int,
                        default=DEFAULT_HYPERPARAMS["checkpoint_freq"])
    parser.add_argument("--eval-freq", type=int,
                        default=DEFAULT_HYPERPARAMS["eval_freq"])

    parser.add_argument("--w-nrmse", type=float,
                        default=DEFAULT_HYPERPARAMS["w_nrmse"])
    parser.add_argument("--w-r2", type=float,
                        default=DEFAULT_HYPERPARAMS["w_r2"])
    parser.add_argument("--w-peakP", type=float,
                        default=DEFAULT_HYPERPARAMS["w_peakP"])
    parser.add_argument("--w-ca50", type=float,
                        default=DEFAULT_HYPERPARAMS["w_ca50"])
    parser.add_argument("--w-tb", type=float,
                        default=DEFAULT_HYPERPARAMS["w_tb"])
    parser.add_argument("--w-xb", type=float,
                        default=DEFAULT_HYPERPARAMS["w_xb"])

    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_on_split(args)


if __name__ == "__main__":
    main()
