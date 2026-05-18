"""Constraint-aware split optimization for splitsmith.

Treats train/val/test assignment as an optimization problem: find the
allocation that minimizes a composite penalty across group exclusivity,
temporal ordering, class balance, row ratios, and covariate drift.

Uses greedy iterative refinement (swap-based local search) so we keep
the zero-extra-dependency constraint (no scipy/cvxpy needed).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .types import SplitResult
from ._meta import build_metadata


def _class_balance_penalty(y: pd.Series, assignments: np.ndarray) -> float:
    """Sum of absolute deviations from overall class prevalence across splits."""
    overall = y.value_counts(normalize=True, dropna=False)
    penalty = 0.0
    for split_id in range(3):
        mask = assignments == split_id
        if mask.sum() == 0:
            penalty += 1.0
            continue
        split_dist = y[mask].value_counts(normalize=True, dropna=False)
        for label in overall.index:
            penalty += abs(overall[label] - split_dist.get(label, 0.0))
    return penalty


def _ratio_penalty(assignments: np.ndarray, target_ratios: Tuple[float, ...]) -> float:
    """Deviation of actual row ratios from target ratios."""
    n = len(assignments)
    if n == 0:
        return 1.0
    penalty = 0.0
    for split_id, target_r in enumerate(target_ratios):
        actual_r = (assignments == split_id).sum() / n
        penalty += abs(actual_r - target_r)
    return penalty


def _time_ordering_penalty(
    timestamps: pd.Series, assignments: np.ndarray,
) -> float:
    """Penalty for temporal violations: train samples after val, val after test."""
    ts = timestamps.values
    penalty = 0.0

    train_mask = assignments == 0
    val_mask = assignments == 1
    test_mask = assignments == 2

    if train_mask.any() and val_mask.any():
        train_max = ts[train_mask].max()
        val_min = ts[val_mask].min()
        if train_max > val_min:
            penalty += 1.0

    if val_mask.any() and test_mask.any():
        val_max = ts[val_mask].max()
        test_min = ts[test_mask].min()
        if val_max > test_min:
            penalty += 1.0

    if train_mask.any() and test_mask.any():
        train_max = ts[train_mask].max()
        test_min = ts[test_mask].min()
        if train_max > test_min:
            penalty += 1.0

    return penalty


def _drift_penalty(
    df: pd.DataFrame, columns: List[str], assignments: np.ndarray,
) -> float:
    """Mean absolute difference of column means between train and other splits."""
    train_mask = assignments == 0
    if train_mask.sum() == 0:
        return 1.0

    penalty = 0.0
    count = 0
    train_data = df[columns].iloc[train_mask.nonzero()[0]]
    train_means = train_data.mean()
    train_stds = train_data.std().replace(0, 1)

    for split_id in [1, 2]:
        mask = assignments == split_id
        if mask.sum() == 0:
            continue
        split_data = df[columns].iloc[mask.nonzero()[0]]
        split_means = split_data.mean()
        # normalized difference
        diffs = ((split_means - train_means) / train_stds).abs()
        penalty += diffs.mean()
        count += 1

    return penalty / max(count, 1)


def _composite_penalty(
    y: pd.Series,
    assignments: np.ndarray,
    target_ratios: Tuple[float, ...],
    timestamps: Optional[pd.Series],
    drift_df: Optional[pd.DataFrame],
    drift_columns: Optional[List[str]],
    weights: Dict[str, float],
) -> float:
    """Weighted sum of all penalty components."""
    total = 0.0
    total += weights.get("class_balance", 1.0) * _class_balance_penalty(y, assignments)
    total += weights.get("ratio", 1.0) * _ratio_penalty(assignments, target_ratios)

    if timestamps is not None:
        total += weights.get("time_order", 2.0) * _time_ordering_penalty(timestamps, assignments)

    if drift_df is not None and drift_columns:
        total += weights.get("drift", 0.5) * _drift_penalty(drift_df, drift_columns, assignments)

    return total


def optimize_split(
    df: pd.DataFrame,
    target: str,
    *,
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    drift_columns: Optional[Sequence[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    max_iterations: int = 1000,
    patience: int = 50,
) -> SplitResult:
    """Find a split assignment that minimizes a composite constraint penalty.

    Unlike split(), which applies a single deterministic rule, this function
    searches for the assignment that best satisfies multiple constraints at once:
    group exclusivity, temporal ordering, class balance, row ratios, and
    covariate drift minimization.

    Parameters
    ----------
    df : DataFrame
    target : target column name
    groups : if given, entire groups are assigned together (no group leakage)
    time_col : if given, temporal ordering is enforced as a soft constraint
    ratios : target (train, val, test) row ratios
    seed : random seed
    drift_columns : numeric columns to minimize distribution shift on
    weights : penalty weights dict. Keys: "class_balance", "ratio",
        "time_order", "drift". Higher weight = more important constraint.
        Defaults: class_balance=1.0, ratio=1.0, time_order=2.0, drift=0.5
    max_iterations : maximum swap iterations
    patience : stop early if no improvement for this many iterations

    Returns
    -------
    SplitResult with metadata including optimization diagnostics
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if groups is not None and groups not in df.columns:
        raise ValueError(f"groups column '{groups}' not found in DataFrame")
    if time_col is not None and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")
    if not isinstance(ratios, (tuple, list)) or len(ratios) != 3:
        raise ValueError("ratios must have length 3")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    n = len(df)
    if n < 3:
        raise ValueError("dataset must have at least 3 rows")

    w = {"class_balance": 1.0, "ratio": 1.0, "time_order": 2.0, "drift": 0.5}
    if weights is not None:
        w.update(weights)

    rng = np.random.default_rng(seed)
    y = df[target]
    timestamps = pd.to_datetime(df[time_col]) if time_col else None

    drift_cols = None
    drift_df = None
    if drift_columns is not None:
        drift_cols = list(drift_columns)
        missing = [c for c in drift_cols if c not in df.columns]
        if missing:
            raise ValueError(f"drift_columns not found: {missing}")
        drift_df = df[drift_cols].apply(pd.to_numeric, errors="coerce")

    # group-level or row-level optimization
    if groups is not None:
        result, opt_meta = _optimize_grouped(
            df, y, groups, timestamps, drift_df, drift_cols,
            ratios, w, rng, max_iterations, patience,
        )
    else:
        result, opt_meta = _optimize_ungrouped(
            df, y, timestamps, drift_df, drift_cols,
            ratios, w, rng, max_iterations, patience,
        )

    # build metadata
    params = {
        "target": target, "groups": groups, "time_col": time_col,
        "ratios": tuple(ratios), "seed": seed,
        "drift_columns": drift_cols, "weights": w,
        "max_iterations": max_iterations, "patience": patience,
    }
    result.metadata = {
        "strategy": "optimized",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "split_sizes": {
            "train": len(result.train_idx),
            "val": len(result.val_idx),
            "test": len(result.test_idx),
        },
        "achieved_ratios": {
            "train": round(len(result.train_idx) / n, 4),
            "val": round(len(result.val_idx) / n, 4),
            "test": round(len(result.test_idx) / n, 4),
        },
        "optimization": opt_meta,
        "reproducibility": build_metadata(df, params),
    }

    return result


def _optimize_grouped(df, y, groups, timestamps, drift_df, drift_cols,
                      ratios, weights, rng, max_iter, patience):
    """Optimize by swapping entire groups between splits."""
    group_col = df[groups]
    unique_groups = np.array(group_col.unique().tolist())
    n_groups = len(unique_groups)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 groups for optimization, got {n_groups}")

    # map group -> row indices
    group_to_rows = {}
    for g in unique_groups:
        group_to_rows[g] = np.where(group_col == g)[0]

    # initial assignment: random group allocation respecting ratios
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n = len(df)
    group_assignments = {}  # group -> split_id
    current = {0: 0, 1: 0, 2: 0}
    targets = {0: ratios[0] * n, 1: ratios[1] * n, 2: ratios[2] * n}

    for g in shuffled:
        sz = len(group_to_rows[g])
        best = max(range(3), key=lambda s: targets[s] - current[s])
        group_assignments[g] = best
        current[best] += sz

    # row-level assignment array
    assignments = np.zeros(n, dtype=int)
    for g, split_id in group_assignments.items():
        assignments[group_to_rows[g]] = split_id

    best_penalty = _composite_penalty(
        y, assignments, ratios, timestamps, drift_df, drift_cols, weights
    )
    best_assignments = group_assignments.copy()
    no_improve = 0
    iterations_used = 0

    for iteration in range(max_iter):
        # pick a random group and try moving it to a different split
        g = rng.choice(unique_groups)
        old_split = group_assignments[g]
        new_split = rng.choice([s for s in range(3) if s != old_split])

        group_assignments[g] = new_split
        assignments[group_to_rows[g]] = new_split

        new_penalty = _composite_penalty(
            y, assignments, ratios, timestamps, drift_df, drift_cols, weights
        )

        if new_penalty < best_penalty:
            best_penalty = new_penalty
            best_assignments = group_assignments.copy()
            no_improve = 0
        else:
            # revert
            group_assignments[g] = old_split
            assignments[group_to_rows[g]] = old_split
            no_improve += 1

        iterations_used = iteration + 1
        if no_improve >= patience:
            break

    # apply best assignment
    for g, split_id in best_assignments.items():
        assignments[group_to_rows[g]] = split_id

    # ensure every split has at least one group
    for split_id in range(3):
        if (assignments == split_id).sum() == 0:
            donor = int(np.argmax([(assignments == s).sum() for s in range(3)]))
            donor_groups = [g for g, s in best_assignments.items() if s == donor]
            if donor_groups:
                moved = rng.choice(donor_groups)
                best_assignments[moved] = split_id
                assignments[group_to_rows[moved]] = split_id

    train_idx = np.where(assignments == 0)[0].astype(int)
    val_idx = np.where(assignments == 1)[0].astype(int)
    test_idx = np.where(assignments == 2)[0].astype(int)

    opt_meta = {
        "iterations": iterations_used,
        "final_penalty": round(best_penalty, 6),
        "converged": no_improve >= patience,
        "mode": "grouped",
        "n_groups": n_groups,
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx), opt_meta


def _optimize_ungrouped(df, y, timestamps, drift_df, drift_cols,
                        ratios, weights, rng, max_iter, patience):
    """Optimize by swapping individual rows between splits."""
    n = len(df)

    # initial assignment based on ratios
    assignments = np.zeros(n, dtype=int)
    shuffled = np.arange(n)
    rng.shuffle(shuffled)
    train_n = int(ratios[0] * n)
    val_n = int(ratios[1] * n)
    assignments[shuffled[:train_n]] = 0
    assignments[shuffled[train_n:train_n + val_n]] = 1
    assignments[shuffled[train_n + val_n:]] = 2

    best_penalty = _composite_penalty(
        y, assignments, ratios, timestamps, drift_df, drift_cols, weights
    )
    best_assignments = assignments.copy()
    no_improve = 0
    iterations_used = 0

    for iteration in range(max_iter):
        # pick two random rows from different splits and swap
        i = rng.integers(n)
        old_split_i = assignments[i]
        new_split_i = rng.choice([s for s in range(3) if s != old_split_i])

        assignments[i] = new_split_i

        new_penalty = _composite_penalty(
            y, assignments, ratios, timestamps, drift_df, drift_cols, weights
        )

        if new_penalty < best_penalty:
            best_penalty = new_penalty
            best_assignments = assignments.copy()
            no_improve = 0
        else:
            assignments[i] = old_split_i
            no_improve += 1

        iterations_used = iteration + 1
        if no_improve >= patience:
            break

    assignments = best_assignments

    train_idx = np.where(assignments == 0)[0].astype(int)
    val_idx = np.where(assignments == 1)[0].astype(int)
    test_idx = np.where(assignments == 2)[0].astype(int)

    opt_meta = {
        "iterations": iterations_used,
        "final_penalty": round(best_penalty, 6),
        "converged": no_improve >= patience,
        "mode": "ungrouped",
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx), opt_meta
