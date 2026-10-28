"""Curriculum scheduling with vectorised environments (regression for the v0.x bug)."""
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.curriculum import CurriculumSyncCallback
from src.environment.wiebe_env import WiebeEnvV4

SCHED = {"stages": [0, 1, 2], "schedule_steps": [0, 128, 256]}


def _vec(cases, n):
    return DummyVecEnv([lambda i=i: WiebeEnvV4(wiebe_type="double", cases=cases, data_dir="",
                                              curriculum=dict(SCHED), seed=i) for i in range(n)])


def _train(cases, sync, total=512, n_envs=4):
    env = _vec(cases, n_envs)
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=32, n_epochs=1, seed=0, verbose=0,
                policy_kwargs={"net_arch": [dict(pi=[16], vf=[16])]})
    cb = CurriculumSyncCallback() if sync else None
    model.learn(total_timesteps=total, callback=cb)
    return max(env.get_attr("curriculum_stage")), cb


def test_unsynchronised_counter_stalls(synthetic_cases):
    """Without synchronisation each sub-env sees total/n_envs steps (the v0.x bug)."""
    stage, _ = _train(synthetic_cases, sync=False)
    assert stage == 1           # 512 / 4 = 128 per env: stage 2 reached, stage 3 never


def test_synchronised_counter_reaches_final_stage(synthetic_cases):
    stage, cb = _train(synthetic_cases, sync=True)
    assert stage == 2
    assert [s for _, s in cb.stage_history] == [0, 1, 2]
    steps = [t for t, _ in cb.stage_history]
    assert steps[1] >= 128 and steps[2] >= 256
