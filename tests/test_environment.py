"""Gymnasium environment: spaces, action mapping, normalisation, curriculum."""
import numpy as np
import pytest

from src.calibrate import effective_bounds
from src.environment.wiebe_env import PhysLimits, WiebeEnvV4

THREE_STAGE = {"stages": [0, 1, 2], "schedule_steps": [0, 100, 200]}


def make(cases, **kw):
    kw.setdefault("curriculum", THREE_STAGE)
    return WiebeEnvV4(wiebe_type="double", seq_len=20, cases=cases, data_dir="", seed=0, **kw)


def test_spaces(synthetic_cases):
    env = make(synthetic_cases)
    assert env.observation_space.shape == (20 * 4 + 4 + 9,)
    assert env.action_space.shape == (9,)
    obs, info = env.reset(options={"case_idx": 0})
    assert obs.shape == (93,) and np.all(np.isfinite(obs))
    assert np.all(obs[-9:] == 0.0)                       # previous action is zero after reset


def test_actions_map_inside_physical_bounds(synthetic_cases):
    env = make(synthetic_cases)
    rng = np.random.default_rng(1)
    lim = PhysLimits()
    for stage in range(3):
        env.curriculum_stage = stage
        bounds = effective_bounds(env)
        for a in list(rng.uniform(-1, 1, (200, 9))) + [-np.ones(9), np.ones(9)]:
            p = env._action_to_params(np.asarray(a, dtype=np.float32))
            vals = {"theta0": p.theta0, "delta_theta": p.delta_theta, "a": p.a, "m1": p.m1, "m2": p.m2,
                    "lambda_w": p.lambda_w, "k": p.k, "ht_g": p.ht_scale_global, "ht_c": p.ht_scale_combustion}
            for n, v in vals.items():
                lo, hi = bounds[n]
                assert lo - 1e-9 <= v <= hi + 1e-9, (stage, n, v)
    # final stage spans the full physical box
    env.curriculum_stage = 2
    assert effective_bounds(env)["delta_theta"] == pytest.approx(lim.dtheta)


def test_curriculum_widens_bounds_and_noise(synthetic_cases):
    env = make(synthetic_cases)
    np.testing.assert_allclose(env.curriculum_spans, [0.5, 0.75, 1.0])
    np.testing.assert_allclose(env.randomization_scales, [0.3, 0.65, 1.0])
    widths = []
    for stage in range(3):
        env.curriculum_stage = stage
        lo, hi = effective_bounds(env)["theta0"]
        widths.append(hi - lo)
    assert widths[0] < widths[1] < widths[2]


def test_single_stage_configuration_uses_stage1_bounds(synthetic_cases):
    """Documents the v0.x evaluation behaviour: linspace(0.5, 1, 1) == [0.5]."""
    env = make(synthetic_cases, curriculum={"stages": [2], "schedule_steps": [0]})
    assert env.curriculum_spans.tolist() == [0.5]


def test_set_global_step_advances_stage_on_reset(synthetic_cases):
    env = make(synthetic_cases)
    for step, stage in [(0, 0), (99, 0), (100, 1), (250, 2)]:
        env.set_global_step(step)
        env.reset()
        assert env.curriculum_stage == stage


def test_frozen_normalization_overrides_case_statistics(synthetic_cases):
    ref = make(synthetic_cases).get_normalization()
    sub = synthetic_cases[:2]
    recomputed = make(sub)
    frozen = make(sub, normalization=ref)
    assert frozen.get_normalization() == ref
    assert recomputed.get_normalization()["dataset_stats"] != ref["dataset_stats"]
    o1, _ = frozen.reset(options={"case_idx": 0})
    o2, _ = make(synthetic_cases, normalization=ref).reset(options={"case_idx": 0})
    np.testing.assert_allclose(o1, o2)


def test_step_with_true_parameters_reproduces_trace(synthetic_cases):
    """Generator and environment share one forward model: truth gives R2 ~ 1."""
    from src.environment.wiebe_env import WiebeParams
    env = make(synthetic_cases)
    for i in (0, 5, 9):
        c = synthetic_cases[i]
        g = c["ground_truth_params"]
        p = WiebeParams(g["theta0"], g["delta_theta"], g["a"], g["lambda_w"], g["m1"], g["m2"], g["k"],
                        g["ht_g"], g["ht_c"])
        res = env._simulate_episode(c, p)
        assert not res["failed"]
        assert res["info"]["r2"] > 0.999999
        assert res["info"]["nrmse"] < 1e-3


def test_reward_penalises_bad_parameters(synthetic_cases):
    env = make(synthetic_cases)
    env.reset(options={"case_idx": 3})
    _, r_good, done, trunc, info = env.step(np.zeros(9, dtype=np.float32))
    env.reset(options={"case_idx": 3})
    _, r_bad, *_ = env.step(np.full(9, -1.0, dtype=np.float32))
    assert done and not trunc
    assert np.isfinite(r_good) and np.isfinite(r_bad)
    assert set(info["reward_components"]) == {"nrmse", "r2", "peak", "temp", "xb", "ca50"}
    assert info["reward_components"]["ca50"] == 0.0          # inactive without a measured CA50


def test_ca50_term_active_when_target_given(synthetic_cases):
    case = dict(synthetic_cases[3])
    case["ca50_deg"] = 30.0
    env = make([case])
    env.reset(options={"case_idx": 0})
    _, _, _, _, info = env.step(np.zeros(9, dtype=np.float32))
    assert info["reward_components"]["ca50"] < 0.0
