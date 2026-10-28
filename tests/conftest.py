import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message=".*Could not deserialize object.*")

PRETRAINED = ROOT / "models" / "pretrained" / "ppo_speed_isoline_curriculum_seed2024.zip"


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """Fresh noise-free synthetic isoline generated with the package forward model."""
    from data.sample.generate_sample import generate
    out = tmp_path_factory.mktemp("synthetic")
    generate(out, n_cases=10, speed_rpm=2000.0, noise=0.0, seed=42)
    return out


@pytest.fixture(scope="session")
def synthetic_cases(synthetic_dir):
    from src.utils.data_loader import load_cases
    return load_cases(synthetic_dir)
