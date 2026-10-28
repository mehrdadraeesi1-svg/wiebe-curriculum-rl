"""The training CLI runs end to end and writes a usable model + sidecar."""
import json
import subprocess
import sys

from conftest import ROOT


def test_train_cli_writes_model_and_sidecar(tmp_path, synthetic_dir, synthetic_cases):
    ids = [c["case_id"] for c in synthetic_cases]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"strategy": "synthetic", "diagnostics": {},
                                 "train": ids[:7], "val": ids[7:8], "test": ids[8:]}))
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "scripts" / "train_on_split.py"),
           "--config", str(ROOT / "config" / "default_config.yaml"),
           "--data-dir", str(synthetic_dir), "--split-file", str(split), "--output-dir", str(out),
           "--total-timesteps", "256", "--n-envs", "2", "--n-steps", "64", "--batch-size", "64",
           "--n-epochs", "1", "--eval-freq", "64", "--checkpoint-freq", "10000"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    meta = json.loads((out / "training_metadata.json").read_text())
    assert meta["curriculum_mode"] == "synced"
    assert meta["eval_action_span"] == 1.0
    sidecar = out / "models" / "best" / "best_model.json"
    assert sidecar.exists()
    side = json.loads(sidecar.read_text())
    assert side["normalization"]["reference_geometry"]["bore"] == 0.083

    from src.calibrate import PolicyCalibrator
    res = PolicyCalibrator.from_pretrained(out / "models" / "best" / "best_model.zip").calibrate(synthetic_cases[8:])
    assert len(res) == 2

