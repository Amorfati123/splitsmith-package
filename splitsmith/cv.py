"""K-fold cross-validation for splitsmith."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import CVResult, FoldResult


_VALID_STRATEGIES = ("basic", "stratified", "group", "time", "group_time")


def k_fold(
    df: pd.DataFrame,
    target: str,
    k: int = 5,
    strategy: str = "basic",
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    seed: int = 42,
) -> CVResult:
    """K-fold cross-validation with multiple strategies.

    Parameters
    ----------
    df : DataFrame
    target : target column name
    k : number of folds (>= 2)
    strategy : "basic", "stratified", "group", "time", or "group_time"
    groups : column name for group-aware strategies
    time_col : column name for time-aware strategies
    seed : random seed for determinism

    Returns
    -------
    CVResult
    """
    # --- Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if not isinstance(k, int):
        raise TypeError("k must be an int")
    if k < 2:
        raise ValueError("k must be >= 2")
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from: {', '.join(_VALID_STRATEGIES)}")

    n = len(df)
    if n < k:
        raise ValueError(f"k must be <= number of rows ({n}), got {k}")

    if strategy in ("group", "group_time"):
        if groups is None:
            raise ValueError(f"groups parameter is required when strategy='{strategy}'")
        if groups not in df.columns:
            raise ValueError(f"groups column '{groups}' not found in DataFrame")
        n_groups = int(df[groups].nunique(dropna=False))
        if n_groups < k:
            raise ValueError(f"k must be <= number of unique groups ({n_groups}), got {k}")

    if strategy in ("time", "group_time"):
        if time_col is None:
            raise ValueError(f"time_col parameter is required when strategy='{strategy}'")
        if time_col not in df.columns:
            raise ValueError(f"time_col column '{time_col}' not found in DataFrame")

    if strategy == "time" and n < (k + 1):
        raise ValueError(f"time strategy requires at least k+1 rows, got n={n}, k={k}")

    if strategy == "group_time":
        n_groups = int(df[groups].nunique(dropna=False))
        if n_groups < (k + 1):
            raise ValueError(f"group_time strategy requires at least k+1 unique groups, got {n_groups}, k={k}")

    # --- Dispatch ---
    if strategy == "basic":
        folds = _basic_folds(n, k, seed)
        meta = {"strategy": "basic", "k": k, "seed": seed, "n_rows": n}

    elif strategy == "stratified":
        folds = _stratified_folds(df, target, k, seed)
        meta = {
            "strategy": "stratified", "k": k, "seed": seed, "n_rows": n,
            "stratified": True,
            "target_nunique": int(df[target].nunique(dropna=False)),
        }

    elif strategy == "group":
        folds, ng = _group_folds(df, groups, k, seed)
        meta = {"strategy": "group", "k": k, "seed": seed, "n_rows": n, "n_groups": ng}

    elif strategy == "time":
        folds, time_range = _time_folds(df, time_col, k, seed)
        meta = {"strategy": "time", "k": k, "seed": seed, "n_rows": n, "time_range": time_range}

    else:  # group_time
        folds, ng = _group_time_folds(df, groups, time_col, k, seed)
        meta = {"strategy": "group_time", "k": k, "seed": seed, "n_rows": n, "n_groups": ng}

    return CVResult(folds=folds, metadata=meta)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _basic_folds(n: int, k: int, seed: int) -> List[FoldResult]:
    rng = np.random.default_rng(seed)
    shuffled = np.arange(n, dtype=int)
    rng.shuffle(shuffled)
    val_blocks = [b.astype(int, copy=False) for b in np.array_split(shuffled, k)]

    folds = []
    for i in range(k):
        val_idx = val_blocks[i]
        train_idx = np.concatenate([val_blocks[j] for j in range(k) if j != i]).astype(int, copy=False)
        folds.append(FoldResult(
            fold=i, train_idx=train_idx, val_idx=val_idx,
            metadata={"fold": i, "strategy": "basic", "seed": seed},
        ))
    return folds


def _stratified_folds(df: pd.DataFrame, target: str, k: int, seed: int) -> List[FoldResult]:
    rng = np.random.default_rng(seed)
    per_fold: List[List[np.ndarray]] = [[] for _ in range(k)]

    grouped = df[target].groupby(df[target], dropna=False)
    for label, label_idx in grouped.groups.items():
        class_indices = np.asarray(label_idx, dtype=int)
        if len(class_indices) < k:
            raise ValueError(f"class {label!r} has {len(class_indices)} samples, fewer than k={k}")
        rng.shuffle(class_indices)
        parts = [p.astype(int, copy=False) for p in np.array_split(class_indices, k)]
        for i in range(k):
            per_fold[i].append(parts[i])

    val_blocks = []
    for fold_parts in per_fold:
        if fold_parts:
            val_idx = np.concatenate(fold_parts).astype(int, copy=False)
            rng.shuffle(val_idx)
        else:
            val_idx = np.array([], dtype=int)
        val_blocks.append(val_idx)

    folds = []
    for i in range(k):
        val_idx = val_blocks[i]
        train_idx = np.concatenate([val_blocks[j] for j in range(k) if j != i]).astype(int, copy=False)
        meta = {"fold": i, "strategy": "stratified", "seed": seed, "stratified": True}
        folds.append(FoldResult(fold=i, train_idx=train_idx, val_idx=val_idx, metadata=meta))
    return folds


def _group_folds(df: pd.DataFrame, groups: str, k: int, seed: int) -> Tuple[List[FoldResult], int]:
    rng = np.random.default_rng(seed)
    unique_groups = np.asarray(df[groups].unique())
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    group_blocks = [block.tolist() for block in np.array_split(shuffled, k)]
    group_col = df[groups]

    folds = []
    for i in range(k):
        val_groups = set(group_blocks[i])
        train_groups = set(g for j, block in enumerate(group_blocks) if j != i for g in block)
        val_idx = np.where(group_col.isin(val_groups))[0].astype(int)
        train_idx = np.where(group_col.isin(train_groups))[0].astype(int)
        meta = {
            "fold": i, "strategy": "group", "seed": seed,
            "n_groups_val": len(val_groups), "n_groups_train": len(train_groups),
        }
        folds.append(FoldResult(fold=i, train_idx=train_idx, val_idx=val_idx, metadata=meta))

    return folds, len(unique_groups)


def _iso_range(values: pd.Series) -> Dict[str, str]:
    return {"min": values.min().isoformat(), "max": values.max().isoformat()}


def _time_folds(df: pd.DataFrame, time_col: str, k: int, seed: int) -> Tuple[List[FoldResult], Dict[str, str]]:
    """Forward-chaining: train grows each fold, val is always the next block."""
    time_series = pd.to_datetime(df[time_col])
    sorted_idx = np.argsort(time_series.values, kind="stable").astype(int)
    row_blocks = [b.astype(int, copy=False) for b in np.array_split(sorted_idx, k + 1)]

    folds = []
    for i in range(k):
        train_idx = np.concatenate(row_blocks[: i + 1]).astype(int, copy=False)
        val_idx = row_blocks[i + 1]
        train_time = pd.to_datetime(df.iloc[train_idx][time_col])
        val_time = pd.to_datetime(df.iloc[val_idx][time_col])
        meta = {
            "fold": i, "strategy": "time", "seed": seed,
            "train_time_range": _iso_range(train_time),
            "val_time_range": _iso_range(val_time),
        }
        folds.append(FoldResult(fold=i, train_idx=train_idx, val_idx=val_idx, metadata=meta))

    return folds, _iso_range(time_series)


def _group_time_folds(
    df: pd.DataFrame, groups: str, time_col: str, k: int, seed: int,
) -> Tuple[List[FoldResult], int]:
    """Forward-chaining on groups sorted by their max timestamp."""
    time_series = pd.to_datetime(df[time_col])
    max_times = time_series.groupby(df[groups]).max()

    group_order = pd.DataFrame({"group": max_times.index, "max_time": max_times.values})
    group_order = group_order.sort_values(["max_time", "group"], kind="mergesort")
    sorted_groups = group_order["group"].values

    group_blocks = [block.tolist() for block in np.array_split(sorted_groups, k + 1)]
    group_col = df[groups]

    folds = []
    for i in range(k):
        train_groups = set(g for block in group_blocks[: i + 1] for g in block)
        val_groups = set(group_blocks[i + 1])
        train_idx = np.where(group_col.isin(train_groups))[0].astype(int)
        val_idx = np.where(group_col.isin(val_groups))[0].astype(int)
        train_time = pd.to_datetime(df.iloc[train_idx][time_col])
        val_time = pd.to_datetime(df.iloc[val_idx][time_col])
        meta = {
            "fold": i, "strategy": "group_time", "seed": seed,
            "n_groups_train": len(train_groups), "n_groups_val": len(val_groups),
            "train_time_range": _iso_range(train_time),
            "val_time_range": _iso_range(val_time),
        }
        folds.append(FoldResult(fold=i, train_idx=train_idx, val_idx=val_idx, metadata=meta))

    return folds, len(sorted_groups)
