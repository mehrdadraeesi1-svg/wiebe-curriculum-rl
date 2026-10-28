"""End-to-end calibration with the bundled pretrained policy and with the GA."""
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import PRETRAINED, ROOT
from src.calibrate import PARAM_NAMES, PolicyCalibrator


@pytest.fixture(scope="module")
def calibrator():
    return PolicyCalibrator.from_pretrained(PRETRAINED)


def test_sidecar_metadata(calibrator):
    m = calibrator.metadata
    assert m["eval_action_span"] == 1.0
    assert m["curriculum_mode"] == "synced"
    assert set(m["normalization"]) == {"dataset_stats", "pressure_scale", "dp_scale", "reference_geometry"}
    assert m["training_envelope"]["speed_rpm"] == [1500.0, 4500.0]


def test_policy_calibration_is_deterministic_and_physical(calibrator, synthetic_cases):
    r1 = calibrator.calibrate(synthetic_cases)
    r2 = calibrator.calibrate(synthetic_cases)
    for a, b in zip(r1, r2):
        assert a["params"] == b["params"]
    for r in r1:
        assert set(r["params"]) == set(PARAM_NAMES)
        assert all(np.isfinite(v) for v in r["params"].values())
        assert r["metrics"]["xb_final"] <= 1.0


def test_policy_reproduces_synthetic_isoline(calibrator, synthetic_cases):
    """Regression test on the bundled policy (values recorded for v1.0.0).

    The five points from 110 to 190 kPa are reconstructed with R2 > 0.98;
    the points above 200 kPa fail the success criterion, consistent with the
    high-load limitation documented in the README.
    """
    res = calibrator.calibrate(synthetic_cases)
    r2 = np.array([r["metrics"]["r2"] for r in res])
    assert np.all(r2[:5] > 0.98)
    assert [r["success"] for r in res] == [True] * 5 + [False] * 5
    assert all("a:lower" in r["params_at_bound"] for r in res[:7])


def test_frozen_vs_legacy_normalisation_differ(synthetic_cases):
    a = PolicyCalibrator.from_pretrained(PRETRAINED, normalization="frozen").calibrate(synthetic_cases[:3])
    b = PolicyCalibrator.from_pretrained(PRETRAINED, normalization="legacy").calibrate(synthetic_cases[:3])
    assert any(x["params"] != y["params"] for x, y in zip(a, b))


def test_envelope_warnings(calibrator, synthetic_cases):
    c = dict(synthetic_cases[0])
    assert calibrator.check_envelope(c) == []
    c["speed"] = 6000.0
    c["bore"] = 0.09
    w = calibrator.check_envelope(c)
    assert any("speed_rpm" in s for s in w) and any("bore" in s for s in w)


def test_ga_recovers_synthetic_trace(synthetic_cases):
    from src.baselines.ga import GeneticAlgorithm
    ga = GeneticAlgorithm(wiebe_type="double", population_size=30, n_generations=40, seed=0)
    best, fitness, hist = ga.optimize(synthetic_cases[4], verbose=False)
    assert fitness > 0.98
    assert set(best) == set(PARAM_NAMES)


def test_cli_end_to_end(tmp_path, synthetic_dir):
    out = tmp_path / "cal.csv"
    cmd = [sys.executable, str(ROOT / "scripts" / "calibrate_cases.py"), "--cases", str(synthetic_dir),
           "--model", str(PRETRAINED), "--out", str(out), "--plot", str(tmp_path / "plots")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 11
    header = lines[0].split(",")
    for col in PARAM_NAMES + ["r2", "nrmse", "success", "err_delta_theta"]:
        assert col in header
    payload = json.loads(out.with_suffix(".json").read_text())
    assert len(payload["results"]) == 10
    assert len(list((tmp_path / "plots").glob("*.png"))) == 10
