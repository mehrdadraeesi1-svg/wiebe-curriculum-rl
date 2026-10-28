"""Train PPO on a split (shortcut for ``train_on_split.py --algorithm PPO``)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_on_split import main  # noqa: E402

if __name__ == "__main__":
    if "--algorithm" not in sys.argv:
        sys.argv[1:1] = ["--algorithm", "PPO"]
    main()
