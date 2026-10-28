"""Input format (docs/DATA_FORMAT.md) and loader."""
import json

import numpy as np
import pytest

from src.utils.data_loader import load_case, load_cases

REQUIRED_CASE_KEYS = {"case_id", "speed", "lambda", "intake_pressure", "intake_temperature",
                      "bore", "stroke", "connecting_rod", "compression_ratio",
                      "crank_angle", "pressure"}


def test_synthetic_files_follow_schema(synthetic_dir):
    files = sorted(synthetic_dir.glob("*.json"))
    assert len(files) == 10
    raw = json.loads(files[0].read_text())
    assert set(raw) >= {"engine_geometry", "operating_conditions", "pressure_trace"}
    assert set(raw["pressure_trace"]) == {"theta_deg", "pressure_Pa"}
    assert (synthetic_dir / "ground_truth.csv").exists()


def test_loader_converts_to_environment_format(synthetic_cases):
    c = synthetic_cases[0]
    assert REQUIRED_CASE_KEYS <= set(c)
    assert c["crank_angle"].shape == c["pressure"].shape == (3601,)
    assert c["crank_angle"][0] == -180.0 and c["crank_angle"][-1] == 180.0
    assert c["intake_pressure"] > 1e4                 # Pa, not kPa
    assert set(c["ground_truth_params"]) == {"theta0", "delta_theta", "a", "m1", "m2",
                                              "lambda_w", "k", "ht_g", "ht_c"}


def test_optional_ca50_is_passed_through(tmp_path, synthetic_dir):
    raw = json.loads(sorted(synthetic_dir.glob("*.json"))[0].read_text())
    raw["operating_conditions"]["ca50_deg"] = 8.5
    f = tmp_path / "with_ca50.json"
    f.write_text(json.dumps(raw))
    assert load_case(f)["ca50_deg"] == 8.5


def test_mismatched_trace_lengths_are_rejected(tmp_path, synthetic_dir):
    raw = json.loads(sorted(synthetic_dir.glob("*.json"))[0].read_text())
    raw["pressure_trace"]["pressure_Pa"] = raw["pressure_trace"]["pressure_Pa"][:-5]
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_case(f)


def test_load_cases_accepts_file_or_directory(synthetic_dir):
    assert len(load_cases(synthetic_dir)) == 10
    assert len(load_cases(sorted(synthetic_dir.glob("*.json"))[0])) == 1
