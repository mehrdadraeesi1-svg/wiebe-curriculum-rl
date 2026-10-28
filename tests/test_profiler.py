"""The latency profiler measures every value it reports."""
import json
import subprocess
import sys

from conftest import PRETRAINED, ROOT


def test_profiler_reports_measured_rows_only(tmp_path, synthetic_dir):
    out = tmp_path / "profile.csv"
    cmd = [sys.executable, str(ROOT / "scripts" / "profile_inference.py"), "--model", str(PRETRAINED),
           "--cases", str(synthetic_dir), "--repetitions", "100", "--ga-cases", "1",
           "--ga-pop", "10", "--ga-gens", "3", "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    res = json.loads(out.with_suffix(".json").read_text())
    names = [r["component"] for r in res["rows"]]
    assert any(n.startswith("actor-only") for n in names)
    assert any(n.startswith("genetic algorithm") for n in names)
    assert not any("xgboost" in n for n in names)          # no stand-in model without --xgb-model
    for r in res["rows"]:
        assert r["measured"] == "yes" and r["mean_ms"] > 0
    mem = res["memory"]
    assert mem["actor_parameters"] == 92178
    assert mem["total_parameters"] == 182291
