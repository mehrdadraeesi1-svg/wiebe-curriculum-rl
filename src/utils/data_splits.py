"""
Dataset splitting strategies for RL-based Wiebe calibration.

This module addresses reviewer concerns about random splits leaking information
due to the high spatial correlation of operating points. It provides four
split strategies, each answering a different question:

    1. RandomSplit         -- baseline, kept for backward compatibility
    2. GridStratifiedSplit -- inlier generalization (honest interpolation test)
    3. ExtrapolationSplit  -- OOD generalization (held-out region)
    4. KFoldBySpeed        -- speed-wise cross-validation

All splits return a common SplitResult dataclass carrying the train/val/test
case lists plus diagnostic information suitable for direct inclusion in the
response-to-reviewer document.

Usage
-----

    from src.utils.data_loader import WiebeDataLoader
    from src.utils.data_splits import GridStratifiedSplit

    loader = WiebeDataLoader('json_cases')
    cases = loader.load_all_cases()

    splitter = GridStratifiedSplit(
        n_speed_bins=6, n_pressure_bins=6,
        test_fraction=0.20, val_fraction=0.10, seed=42,
    )
    result = splitter.split(cases)
    print(result.summary())

The splitter never mutates the input cases and is deterministic given a seed.

Author: Mehrdad Raeesi
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

OpsKey = Tuple[str, ...]
"""Tuple of operating-condition field names used to position a case in space."""

DEFAULT_OPS_KEYS: OpsKey = ("speed", "lambda", "intake_pressure", "intake_temperature")


@dataclass
class SplitResult:
    """Container for a single train / val / test partition.

    Attributes
    ----------
    strategy : str
        Short name identifying which splitter produced this result.
    train, val, test : list[dict]
        The case dictionaries assigned to each set.
    train_ids, val_ids, test_ids : list[str]
        The ``case_id`` strings, cached for saving to JSON.
    diagnostics : dict
        Free-form statistics useful for the response-to-reviewers document,
        e.g. held-out region bounds, minimum train-test distance, etc.
    """

    strategy: str
    train: List[Dict[str, Any]] = field(default_factory=list)
    val: List[Dict[str, Any]] = field(default_factory=list)
    test: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def train_ids(self) -> List[str]:
        return [c["case_id"] for c in self.train]

    @property
    def val_ids(self) -> List[str]:
        return [c["case_id"] for c in self.val]

    @property
    def test_ids(self) -> List[str]:
        return [c["case_id"] for c in self.test]

    def summary(self) -> str:
        """One-page textual summary suitable for logging."""
        lines = [
            "=" * 68,
            f"Split strategy: {self.strategy}",
            "=" * 68,
            f"  Train: {len(self.train):5d} cases",
            f"  Val:   {len(self.val):5d} cases",
            f"  Test:  {len(self.test):5d} cases",
            f"  Total: {len(self.train) + len(self.val) + len(self.test):5d} cases",
        ]
        if self.diagnostics:
            lines.append("-" * 68)
            lines.append("Diagnostics:")
            for k, v in self.diagnostics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
        lines.append("=" * 68)
        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        """Persist the split so downstream training scripts can re-read it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "diagnostics": self.diagnostics,
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved split '%s' -> %s", self.strategy, path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case_vector(case: Dict[str, Any], keys: OpsKey) -> np.ndarray:
    """Extract the operating-condition vector in a fixed key order."""
    return np.asarray([float(case[k]) for k in keys], dtype=np.float64)


def _stack_cases(
    cases: Sequence[Dict[str, Any]], keys: OpsKey = DEFAULT_OPS_KEYS
) -> np.ndarray:
    """Shape (N, len(keys)) matrix of operating conditions."""
    return np.vstack([_case_vector(c, keys) for c in cases])


def _standardize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization column-wise. Returns (Z, mean, std)."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    z = (matrix - mean) / std
    return z, mean, std


def _min_pairwise_distance(a: np.ndarray, b: np.ndarray) -> float:
    """For each row of a, find min euclidean distance to rows of b; return
    the smallest such distance overall.

    This is the quantity reviewers care about most: if it is near zero, the
    test set has a nearly-identical neighbour in the training set and the
    random-split generalization claim is weak.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Chunked computation to keep memory modest for 1000+ cases.
    chunk = 128
    best = np.inf
    for start in range(0, a.shape[0], chunk):
        sub = a[start:start + chunk]
        # (chunk, 1, d) - (1, M, d) -> (chunk, M, d)
        diffs = sub[:, None, :] - b[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        best = min(best, float(dists.min()))
    return best


def _mean_nearest_neighbor_distance(
    target: np.ndarray, pool: np.ndarray
) -> float:
    """Average of each target point's nearest-neighbour distance in pool."""
    if target.size == 0 or pool.size == 0:
        return float("nan")
    chunk = 128
    mins = []
    for start in range(0, target.shape[0], chunk):
        sub = target[start:start + chunk]
        diffs = sub[:, None, :] - pool[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        mins.append(dists.min(axis=1))
    return float(np.concatenate(mins).mean())


# ---------------------------------------------------------------------------
# 1. Random split (baseline)
# ---------------------------------------------------------------------------

class RandomSplit:
    """Baseline random split -- kept so reported results remain reproducible.

    This is the split used in the original submission. It is retained for
    transparency; the response-to-reviewers compares the three stratified
    alternatives against this one.
    """

    def __init__(
        self,
        test_fraction: float = 0.20,
        val_fraction: float = 0.10,
        seed: int = 42,
        ops_keys: OpsKey = DEFAULT_OPS_KEYS,
    ) -> None:
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.ops_keys = ops_keys

    def split(self, cases: Sequence[Dict[str, Any]]) -> SplitResult:
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(cases))
        rng.shuffle(idx)

        n_total = len(cases)
        n_test = int(round(n_total * self.test_fraction))
        n_val = int(round((n_total - n_test) * self.val_fraction))

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        train = [cases[i] for i in train_idx]
        val = [cases[i] for i in val_idx]
        test = [cases[i] for i in test_idx]

        # Leakage diagnostic
        z_all, _, _ = _standardize(_stack_cases(cases, self.ops_keys))
        z_train = z_all[train_idx]
        z_test = z_all[test_idx]

        diagnostics = {
            "test_fraction": float(self.test_fraction),
            "val_fraction": float(self.val_fraction),
            "seed": int(self.seed),
            "min_test_to_train_distance_zscore": _min_pairwise_distance(
                z_test, z_train
            ),
            "mean_test_to_train_NN_distance_zscore": _mean_nearest_neighbor_distance(
                z_test, z_train
            ),
        }
        return SplitResult(
            strategy="random",
            train=train,
            val=val,
            test=test,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 2. Grid-stratified split (honest inlier generalization)
# ---------------------------------------------------------------------------

class GridStratifiedSplit:
    """Balanced random split across a 2-D operating-condition grid.

    The operating envelope is partitioned into a ``n_speed_bins x
    n_pressure_bins`` grid. From *each* non-empty cell we draw the same
    fraction of cases for test and val. This guarantees that every region
    of the map is represented in the test set with similar density,
    addressing the reviewer concern that random splits may over-sample
    the densest part of the envelope.

    The minimum train-test distance diagnostic lets us quantify exactly
    how close the nearest train point is for each test point, in z-score
    units. We report this in the manuscript as the primary evidence that
    this split is not a trivial interpolation task.
    """

    def __init__(
        self,
        n_speed_bins: int = 6,
        n_pressure_bins: int = 6,
        test_fraction: float = 0.20,
        val_fraction: float = 0.10,
        seed: int = 42,
        ops_keys: OpsKey = DEFAULT_OPS_KEYS,
        speed_key: str = "speed",
        pressure_key: str = "intake_pressure",
    ) -> None:
        self.n_speed_bins = n_speed_bins
        self.n_pressure_bins = n_pressure_bins
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.seed = seed
        self.ops_keys = ops_keys
        self.speed_key = speed_key
        self.pressure_key = pressure_key

    def _assign_bins(self, cases: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Return integer cell index for each case."""
        speeds = np.asarray([c[self.speed_key] for c in cases])
        pressures = np.asarray([c[self.pressure_key] for c in cases])

        speed_edges = np.linspace(
            speeds.min() - 1e-6, speeds.max() + 1e-6, self.n_speed_bins + 1
        )
        pressure_edges = np.linspace(
            pressures.min() - 1e-6, pressures.max() + 1e-6, self.n_pressure_bins + 1
        )

        speed_bin = np.digitize(speeds, speed_edges[1:-1])
        pressure_bin = np.digitize(pressures, pressure_edges[1:-1])
        return speed_bin * self.n_pressure_bins + pressure_bin

    def split(self, cases: Sequence[Dict[str, Any]]) -> SplitResult:
        rng = np.random.default_rng(self.seed)
        n = len(cases)
        bins = self._assign_bins(cases)

        test_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)

        # For each bin, draw the same fraction for test and val.
        unique_bins = np.unique(bins)
        bin_sizes: Dict[int, int] = {}
        for b in unique_bins:
            members = np.where(bins == b)[0]
            bin_sizes[int(b)] = int(len(members))
            if len(members) < 3:
                # Too few cases in this cell; keep everything for training
                # to preserve coverage.
                continue
            rng.shuffle(members)
            n_test_local = max(1, int(round(len(members) * self.test_fraction)))
            n_val_local = max(
                0, int(round((len(members) - n_test_local) * self.val_fraction))
            )
            test_mask[members[:n_test_local]] = True
            val_mask[members[n_test_local:n_test_local + n_val_local]] = True

        train_idx = np.where(~test_mask & ~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        train = [cases[i] for i in train_idx]
        val = [cases[i] for i in val_idx]
        test = [cases[i] for i in test_idx]

        # Leakage diagnostic
        z_all, _, _ = _standardize(_stack_cases(cases, self.ops_keys))
        z_train = z_all[train_idx]
        z_test = z_all[test_idx]

        diagnostics = {
            "n_speed_bins": int(self.n_speed_bins),
            "n_pressure_bins": int(self.n_pressure_bins),
            "n_nonempty_cells": int(len(unique_bins)),
            "min_cell_size": int(min(bin_sizes.values())),
            "max_cell_size": int(max(bin_sizes.values())),
            "test_fraction": float(self.test_fraction),
            "val_fraction": float(self.val_fraction),
            "seed": int(self.seed),
            "min_test_to_train_distance_zscore": _min_pairwise_distance(
                z_test, z_train
            ),
            "mean_test_to_train_NN_distance_zscore": _mean_nearest_neighbor_distance(
                z_test, z_train
            ),
        }
        return SplitResult(
            strategy="grid_stratified",
            train=train,
            val=val,
            test=test,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 3. Extrapolation holdout (OOD generalization)
# ---------------------------------------------------------------------------

class ExtrapolationSplit:
    """Hold out a contiguous region of the operating envelope.

    Test cases are those satisfying a user-supplied predicate (for
    instance, the high-load / high-speed corner of the map). The
    remaining cases are partitioned randomly into train/val. This
    directly answers the reviewer's ``out-of-distribution generalization
    validation`` question: the model must predict parameters for
    operating points that live in a region of the envelope it has never
    seen during training.

    Predicate options
    -----------------
    1. Pass a callable ``predicate(case) -> bool``.
    2. Use one of the built-in predicates created by class methods:
         ``ExtrapolationSplit.high_load(p_threshold_Pa=270_000)``
         ``ExtrapolationSplit.high_speed(speed_threshold_rpm=3500)``
         ``ExtrapolationSplit.lean_burn(lambda_threshold=1.5)``
         ``ExtrapolationSplit.high_load_high_speed(...)``

    Notes
    -----
    Because the test region is fixed, the ``seed`` controls only the
    train/val split. The test set is deterministic once a predicate is
    chosen.
    """

    def __init__(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        region_name: str,
        val_fraction: float = 0.10,
        seed: int = 42,
        ops_keys: OpsKey = DEFAULT_OPS_KEYS,
    ) -> None:
        self.predicate = predicate
        self.region_name = region_name
        self.val_fraction = val_fraction
        self.seed = seed
        self.ops_keys = ops_keys

    # ---- Built-in region factories ------------------------------------
    @classmethod
    def high_load(
        cls,
        p_threshold_Pa: float = 270_000.0,
        val_fraction: float = 0.10,
        seed: int = 42,
    ) -> "ExtrapolationSplit":
        return cls(
            predicate=lambda c: c["intake_pressure"] >= p_threshold_Pa,
            region_name=f"high_load_P>={p_threshold_Pa/1000:.0f}kPa",
            val_fraction=val_fraction,
            seed=seed,
        )

    @classmethod
    def high_speed(
        cls,
        speed_threshold_rpm: float = 3500.0,
        val_fraction: float = 0.10,
        seed: int = 42,
    ) -> "ExtrapolationSplit":
        return cls(
            predicate=lambda c: c["speed"] >= speed_threshold_rpm,
            region_name=f"high_speed_N>={speed_threshold_rpm:.0f}rpm",
            val_fraction=val_fraction,
            seed=seed,
        )

    @classmethod
    def lean_burn(
        cls,
        lambda_threshold: float = 1.5,
        val_fraction: float = 0.10,
        seed: int = 42,
    ) -> "ExtrapolationSplit":
        return cls(
            predicate=lambda c: c["lambda"] >= lambda_threshold,
            region_name=f"lean_burn_lambda>={lambda_threshold:.2f}",
            val_fraction=val_fraction,
            seed=seed,
        )

    @classmethod
    def high_load_high_speed(
        cls,
        p_threshold_Pa: float = 250_000.0,
        speed_threshold_rpm: float = 3500.0,
        val_fraction: float = 0.10,
        seed: int = 42,
    ) -> "ExtrapolationSplit":
        return cls(
            predicate=lambda c: (
                c["intake_pressure"] >= p_threshold_Pa
                and c["speed"] >= speed_threshold_rpm
            ),
            region_name=(
                f"hlhs_P>={p_threshold_Pa/1000:.0f}kPa_N>={speed_threshold_rpm:.0f}rpm"
            ),
            val_fraction=val_fraction,
            seed=seed,
        )

    # ---- Main split ---------------------------------------------------
    def split(self, cases: Sequence[Dict[str, Any]]) -> SplitResult:
        test_idx = np.asarray(
            [i for i, c in enumerate(cases) if self.predicate(c)], dtype=int
        )
        remaining_idx = np.asarray(
            [i for i, c in enumerate(cases) if not self.predicate(c)], dtype=int
        )

        if test_idx.size == 0:
            raise ValueError(
                f"Extrapolation predicate '{self.region_name}' selected "
                "zero cases. Adjust threshold."
            )
        if test_idx.size >= 0.5 * len(cases):
            logger.warning(
                "Extrapolation predicate '%s' selected %d/%d cases (>=50%%). "
                "Test region is unusually large.",
                self.region_name, test_idx.size, len(cases),
            )

        rng = np.random.default_rng(self.seed)
        rng.shuffle(remaining_idx)
        n_val = int(round(len(remaining_idx) * self.val_fraction))
        val_idx = remaining_idx[:n_val]
        train_idx = remaining_idx[n_val:]

        train = [cases[i] for i in train_idx]
        val = [cases[i] for i in val_idx]
        test = [cases[i] for i in test_idx]

        # Region geometry -- what bounds does the held-out set cover?
        ops_test = _stack_cases(test, self.ops_keys)
        ops_train = _stack_cases(train, self.ops_keys)

        z_all, _, _ = _standardize(_stack_cases(cases, self.ops_keys))
        z_train = z_all[train_idx]
        z_test = z_all[test_idx]

        diagnostics = {
            "region_name": self.region_name,
            "val_fraction": float(self.val_fraction),
            "seed": int(self.seed),
            "test_region_bounds": {
                self.ops_keys[j]: {
                    "min": float(ops_test[:, j].min()),
                    "max": float(ops_test[:, j].max()),
                }
                for j in range(len(self.ops_keys))
            },
            "train_region_bounds": {
                self.ops_keys[j]: {
                    "min": float(ops_train[:, j].min()),
                    "max": float(ops_train[:, j].max()),
                }
                for j in range(len(self.ops_keys))
            },
            "min_test_to_train_distance_zscore": _min_pairwise_distance(
                z_test, z_train
            ),
            "mean_test_to_train_NN_distance_zscore": _mean_nearest_neighbor_distance(
                z_test, z_train
            ),
        }
        return SplitResult(
            strategy=f"extrapolation[{self.region_name}]",
            train=train,
            val=val,
            test=test,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 4. Speed-isoline holdout (exploits the grid-sampled dataset structure)
# ---------------------------------------------------------------------------

class SpeedIsolineHoldout:
    """Hold out every case at one or more specific engine speeds.

    This strategy is tailored to the Yuan et al. GDI dataset, which was
    acquired as a swept operating matrix at six discrete engine speeds
    (1500, 2000, 2500, 3000, 4000, 4500 RPM). Random and grid-stratified
    splits inevitably place training and test points on the same isoline,
    producing near-identical neighbours in the standardized feature space.

    By removing entire speed isolines from the training set, this splitter
    forces the agent to extrapolate to a rotational regime it has never
    seen, providing the strongest single-axis generalization test the
    dataset can support.

    Parameters
    ----------
    holdout_speeds : sequence of float
        Engine speeds (in RPM) to hold out. Matching is performed with a
        tolerance of ``speed_tolerance_rpm`` so that near-identical values
        like 1999.9 and 2000.1 are collapsed onto the same isoline.
    speed_tolerance_rpm : float
        Numerical tolerance for matching a case to a held-out speed.
    val_fraction : float
        Fraction of the remaining (training) cases that become validation.

    Built-in factories
    ------------------
    ``SpeedIsolineHoldout.mid_speed()`` -> 3000 RPM (edge of mid-range)
    ``SpeedIsolineHoldout.redline()``   -> 4500 RPM (upper limit)
    """

    def __init__(
        self,
        holdout_speeds: Sequence[float],
        speed_tolerance_rpm: float = 25.0,
        val_fraction: float = 0.10,
        seed: int = 42,
        ops_keys: OpsKey = DEFAULT_OPS_KEYS,
    ) -> None:
        if not holdout_speeds:
            raise ValueError("holdout_speeds must be non-empty")
        self.holdout_speeds = [float(s) for s in holdout_speeds]
        self.speed_tolerance_rpm = speed_tolerance_rpm
        self.val_fraction = val_fraction
        self.seed = seed
        self.ops_keys = ops_keys

    # ---- Built-in factories ------------------------------------------
    @classmethod
    def mid_speed(cls, speed_rpm: float = 3000.0,
                  val_fraction: float = 0.10,
                  seed: int = 42) -> "SpeedIsolineHoldout":
        return cls([speed_rpm], val_fraction=val_fraction, seed=seed)

    @classmethod
    def redline(cls, speed_rpm: float = 4500.0,
                val_fraction: float = 0.10,
                seed: int = 42) -> "SpeedIsolineHoldout":
        return cls([speed_rpm], val_fraction=val_fraction, seed=seed)

    # ---- Main split --------------------------------------------------
    def split(self, cases: Sequence[Dict[str, Any]]) -> SplitResult:
        speeds = np.asarray([c["speed"] for c in cases])

        test_mask = np.zeros(len(cases), dtype=bool)
        for holdout in self.holdout_speeds:
            test_mask |= np.abs(speeds - holdout) <= self.speed_tolerance_rpm

        test_idx = np.where(test_mask)[0]
        remaining_idx = np.where(~test_mask)[0]

        if test_idx.size == 0:
            raise ValueError(
                f"No cases match holdout_speeds={self.holdout_speeds} "
                f"within {self.speed_tolerance_rpm} RPM tolerance. "
                "Check the dataset's unique speed values."
            )
        if test_idx.size >= 0.5 * len(cases):
            logger.warning(
                "Speed-isoline holdout selected %d/%d cases (>=50%%).",
                test_idx.size, len(cases),
            )

        rng = np.random.default_rng(self.seed)
        rng.shuffle(remaining_idx)
        n_val = int(round(len(remaining_idx) * self.val_fraction))
        val_idx = remaining_idx[:n_val]
        train_idx = remaining_idx[n_val:]

        train = [cases[i] for i in train_idx]
        val = [cases[i] for i in val_idx]
        test = [cases[i] for i in test_idx]

        z_all, _, _ = _standardize(_stack_cases(cases, self.ops_keys))
        z_train = z_all[train_idx]
        z_test = z_all[test_idx]

        holdout_label = "+".join(f"{s:.0f}" for s in self.holdout_speeds)
        diagnostics = {
            "holdout_speeds_rpm": list(self.holdout_speeds),
            "speed_tolerance_rpm": float(self.speed_tolerance_rpm),
            "n_test_per_holdout_speed": [
                int(np.sum(np.abs(speeds[test_idx] - s)
                           <= self.speed_tolerance_rpm))
                for s in self.holdout_speeds
            ],
            "unique_test_speeds": sorted(
                set(round(float(s), 1) for s in speeds[test_idx])
            )[:10],  # cap list length
            "val_fraction": float(self.val_fraction),
            "seed": int(self.seed),
            "min_test_to_train_distance_zscore": _min_pairwise_distance(
                z_test, z_train
            ),
            "mean_test_to_train_NN_distance_zscore":
                _mean_nearest_neighbor_distance(z_test, z_train),
        }
        return SplitResult(
            strategy=f"speed_isoline_holdout[N={holdout_label}rpm]",
            train=train,
            val=val,
            test=test,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 5. K-fold cross-validation stratified by speed
# ---------------------------------------------------------------------------

class KFoldBySpeed:
    """K-fold CV where each fold is a contiguous speed band.

    This exposes the model to the strongest form of speed extrapolation:
    one fold at a time is excluded from training. Useful for robustness
    checks and reporting variance across folds.

    Iterate using ``for fold in splitter.folds(cases): ...``.
    """

    def __init__(
        self,
        n_folds: int = 5,
        val_fraction: float = 0.10,
        seed: int = 42,
        ops_keys: OpsKey = DEFAULT_OPS_KEYS,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        self.n_folds = n_folds
        self.val_fraction = val_fraction
        self.seed = seed
        self.ops_keys = ops_keys

    def folds(self, cases: Sequence[Dict[str, Any]]) -> Iterator[SplitResult]:
        speeds = np.asarray([c["speed"] for c in cases])
        edges = np.quantile(
            speeds, np.linspace(0.0, 1.0, self.n_folds + 1)
        )
        edges[0] -= 1e-6
        edges[-1] += 1e-6

        z_all, _, _ = _standardize(_stack_cases(cases, self.ops_keys))

        for k in range(self.n_folds):
            test_idx = np.where(
                (speeds >= edges[k]) & (speeds < edges[k + 1])
            )[0]
            remaining_idx = np.setdiff1d(np.arange(len(cases)), test_idx)

            rng = np.random.default_rng(self.seed + k)
            rng.shuffle(remaining_idx)
            n_val = int(round(len(remaining_idx) * self.val_fraction))
            val_idx = remaining_idx[:n_val]
            train_idx = remaining_idx[n_val:]

            train = [cases[i] for i in train_idx]
            val = [cases[i] for i in val_idx]
            test = [cases[i] for i in test_idx]

            diagnostics = {
                "fold_index": int(k),
                "n_folds": int(self.n_folds),
                "test_speed_range_rpm": (
                    float(edges[k]),
                    float(edges[k + 1]),
                ),
                "val_fraction": float(self.val_fraction),
                "seed": int(self.seed + k),
                "min_test_to_train_distance_zscore": _min_pairwise_distance(
                    z_all[test_idx], z_all[train_idx]
                ),
                "mean_test_to_train_NN_distance_zscore":
                    _mean_nearest_neighbor_distance(
                        z_all[test_idx], z_all[train_idx]
                    ),
            }
            yield SplitResult(
                strategy=f"kfold_by_speed[fold={k + 1}/{self.n_folds}]",
                train=train,
                val=val,
                test=test,
                diagnostics=diagnostics,
            )


# ---------------------------------------------------------------------------
# Utility: side-by-side comparison table for response-to-reviewers
# ---------------------------------------------------------------------------

def compare_splits(results: Sequence[SplitResult]) -> str:
    """Produce a compact textual table summarising several splits.

    The most important row is
    ``min_test_to_train_distance_zscore`` -- this is the number we
    quote back to the reviewer to show that the random split is a
    trivial neighbour-lookup task while the stratified and OOD splits
    put genuine distance between train and test points.
    """
    cols = [r.strategy for r in results]
    rows = [
        ("n_train", [len(r.train) for r in results]),
        ("n_val", [len(r.val) for r in results]),
        ("n_test", [len(r.test) for r in results]),
        ("min_NN_dist (z)", [
            r.diagnostics.get("min_test_to_train_distance_zscore", float("nan"))
            for r in results
        ]),
        ("mean_NN_dist (z)", [
            r.diagnostics.get("mean_test_to_train_NN_distance_zscore", float("nan"))
            for r in results
        ]),
    ]

    col_widths = [max(12, len(c) + 2) for c in cols]
    header = f"{'metric':<22} | " + " | ".join(
        f"{c:<{w}}" for c, w in zip(cols, col_widths)
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, values in rows:
        formatted = []
        for v, w in zip(values, col_widths):
            if isinstance(v, float):
                formatted.append(f"{v:<{w}.4f}")
            else:
                formatted.append(f"{v:<{w}}")
        lines.append(f"{name:<22} | " + " | ".join(formatted))
    lines.append(sep)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    # Smoke test on synthetic data.
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    synthetic = [
        {
            "case_id": f"syn_{i:04d}",
            "speed": float(rng.uniform(1500, 4500)),
            "lambda": float(rng.uniform(0.9, 2.0)),
            "intake_pressure": float(rng.uniform(100_000, 300_000)),
            "intake_temperature": float(rng.uniform(315, 320)),
        }
        for i in range(500)
    ]
    print(RandomSplit().split(synthetic).summary())
    print(GridStratifiedSplit().split(synthetic).summary())
    print(ExtrapolationSplit.high_load().split(synthetic).summary())
