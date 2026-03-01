from __future__ import annotations

import numpy as np
import pandas as pd

from .types import SplitResult


def _validate_and_normalize_sizes(n: int, ratios) -> tuple[int, int, int]:
    train_size = int(np.floor(ratios[0] * n))
    val_size = int(np.floor(ratios[1] * n))
    test_size = n - train_size - val_size

    sizes = [train_size, val_size, test_size]

    while any(s == 0 for s in sizes):
        largest = sizes.index(max(sizes))
        smallest = sizes.index(min(sizes))
        if sizes[largest] <= 1:
            raise ValueError("Cannot allocate at least 1 sample to each split with given ratios.")
        sizes[largest] -= 1
        sizes[smallest] += 1

    return sizes[0], sizes[1], sizes[2]


def _random_split(df, target, ratios, seed, stratify):
    """Random (optionally stratified) train/val/test split."""
    n = len(df)
    rng = np.random.default_rng(seed)

    train_size, val_size, test_size = _validate_and_normalize_sizes(n, ratios)

    y = df[target]
    nunique = int(y.nunique(dropna=False))
    categorical_like = (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
        or nunique <= min(50, max(2, int(0.1 * n)))
    )

    if stratify is None:
        do_stratify = categorical_like
    else:
        do_stratify = bool(stratify)
        if do_stratify and not categorical_like:
            raise ValueError(
                f"stratify=True but target '{target}' looks continuous (nunique={nunique}); "
                "set stratify=False for regression/continuous targets."
            )

    if not do_stratify:
        indices = np.arange(n, dtype=int)
        rng.shuffle(indices)
        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size :]
    else:
        train_parts = []
        val_parts = []
        test_parts = []

        for label, idxs in y.groupby(y, dropna=False).groups.items():
            idxs = np.asarray(list(idxs), dtype=int)
            rng.shuffle(idxs)

            k = len(idxs)
            if k < 3:
                raise ValueError(
                    f"Cannot stratify into 3 splits: class {label!r} has only {k} samples."
                )

            t, v, te = _validate_and_normalize_sizes(k, ratios)
            train_parts.append(idxs[:t])
            val_parts.append(idxs[t : t + v])
            test_parts.append(idxs[t + v :])

        train_idx = np.concatenate(train_parts).astype(int, copy=False)
        val_idx = np.concatenate(val_parts).astype(int, copy=False)
        test_idx = np.concatenate(test_parts).astype(int, copy=False)

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        if len(np.unique(all_idx)) != n:
            raise RuntimeError("Internal error: stratified split produced overlap or missing indices.")

    metadata = {
        "strategy": "random",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "stratified": bool(do_stratify),
        "target_nunique": nunique,
    }

    if do_stratify:
        metadata["target_counts_overall"] = {k: int(v) for k, v in y.value_counts(dropna=False).to_dict().items()}
        metadata["target_counts_train"] = {k: int(v) for k, v in y.iloc[train_idx].value_counts(dropna=False).to_dict().items()}
        metadata["target_counts_val"] = {k: int(v) for k, v in y.iloc[val_idx].value_counts(dropna=False).to_dict().items()}
        metadata["target_counts_test"] = {k: int(v) for k, v in y.iloc[test_idx].value_counts(dropna=False).to_dict().items()}

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, metadata=metadata)


def _group_split(df, groups, ratios, seed):
    """Split by group — no group appears in more than one split."""
    unique_groups = np.asarray(df[groups].unique())
    n_groups = len(unique_groups)

    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups, got {n_groups}")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    train_n, val_n, test_n = _validate_and_normalize_sizes(n_groups, ratios)

    train_groups = set(shuffled[:train_n].tolist())
    val_groups = set(shuffled[train_n : train_n + val_n].tolist())
    test_groups = set(shuffled[train_n + val_n :].tolist())

    group_col = df[groups]
    train_idx = np.where(group_col.isin(train_groups))[0].astype(int)
    val_idx = np.where(group_col.isin(val_groups))[0].astype(int)
    test_idx = np.where(group_col.isin(test_groups))[0].astype(int)

    metadata = {
        "strategy": "group",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": len(df),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "n_groups": n_groups,
        "groups_per_split": {"train": train_n, "val": val_n, "test": test_n},
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, metadata=metadata)


def _time_split(df, time_col, ratios, seed):
    """Chronological split — train is earliest, test is latest."""
    n = len(df)
    time_series = pd.to_datetime(df[time_col])
    sorted_idx = np.argsort(time_series.values, kind="stable").astype(int)

    train_n, val_n, test_n = _validate_and_normalize_sizes(n, ratios)

    train_idx = sorted_idx[:train_n]
    val_idx = sorted_idx[train_n : train_n + val_n]
    test_idx = sorted_idx[train_n + val_n :]

    train_times = time_series.iloc[train_idx]
    val_times = time_series.iloc[val_idx]
    test_times = time_series.iloc[test_idx]

    metadata = {
        "strategy": "time",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "time_range": {
            "train_min": train_times.min().isoformat(),
            "train_max": train_times.max().isoformat(),
            "val_min": val_times.min().isoformat(),
            "val_max": val_times.max().isoformat(),
            "test_min": test_times.min().isoformat(),
            "test_max": test_times.max().isoformat(),
        },
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, metadata=metadata)


def _group_time_split(df, groups, time_col, ratios, seed):
    """Split groups by their latest timestamp — prevents entity + time leakage."""
    time_series = pd.to_datetime(df[time_col])
    max_times = time_series.groupby(df[groups]).max()

    group_order = pd.DataFrame({"group": max_times.index, "max_time": max_times.values})
    group_order = group_order.sort_values(["max_time", "group"], kind="mergesort")
    sorted_groups = group_order["group"].values

    n_groups = len(sorted_groups)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups, got {n_groups}")

    train_n, val_n, test_n = _validate_and_normalize_sizes(n_groups, ratios)

    train_groups = set(sorted_groups[:train_n].tolist())
    val_groups = set(sorted_groups[train_n : train_n + val_n].tolist())
    test_groups = set(sorted_groups[train_n + val_n :].tolist())

    group_col = df[groups]
    train_idx = np.where(group_col.isin(train_groups))[0].astype(int)
    val_idx = np.where(group_col.isin(val_groups))[0].astype(int)
    test_idx = np.where(group_col.isin(test_groups))[0].astype(int)

    train_times = time_series.iloc[train_idx]
    val_times = time_series.iloc[val_idx]
    test_times = time_series.iloc[test_idx]

    metadata = {
        "strategy": "group_time",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": len(df),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "n_groups": n_groups,
        "groups_per_split": {"train": train_n, "val": val_n, "test": test_n},
        "time_range": {
            "train_min": train_times.min().isoformat(),
            "train_max": train_times.max().isoformat(),
            "val_min": val_times.min().isoformat(),
            "val_max": val_times.max().isoformat(),
            "test_min": test_times.min().isoformat(),
            "test_max": test_times.max().isoformat(),
        },
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, metadata=metadata)


def split(
    df,
    target: str,
    groups=None,
    time_col=None,
    strategy: str = "random",
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
    stratify: bool | None = None,
) -> SplitResult:
    # --- Input validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if target not in df.columns:
        raise KeyError(f"target column '{target}' not found in dataframe")

    if not isinstance(ratios, (tuple, list)) or len(ratios) != 3:
        raise ValueError("ratios must have length 3 (train, val, test)")

    if any(r <= 0 for r in ratios):
        raise ValueError("all ratios must be > 0")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    n = len(df)
    if n < 3:
        raise ValueError("dataset must have at least 3 rows")

    # --- Strategy dispatch ---
    if strategy == "random":
        return _random_split(df, target, ratios, seed, stratify)

    elif strategy == "group":
        if stratify is True:
            raise NotImplementedError("Stratified group splitting not yet supported")
        if groups is None:
            raise ValueError("groups parameter is required when strategy='group'")
        if groups not in df.columns:
            raise ValueError(f"groups column '{groups}' not found in dataframe")
        return _group_split(df, groups, ratios, seed)

    elif strategy == "time":
        if stratify is True:
            raise NotImplementedError("Stratified time splitting not yet supported")
        if time_col is None:
            raise ValueError("time_col parameter is required when strategy='time'")
        if time_col not in df.columns:
            raise ValueError(f"time_col column '{time_col}' not found in dataframe")
        return _time_split(df, time_col, ratios, seed)

    elif strategy == "group_time":
        if stratify is True:
            raise NotImplementedError("Stratified group_time splitting not yet supported")
        if groups is None or time_col is None:
            raise ValueError("Both groups and time_col are required when strategy='group_time'")
        if groups not in df.columns:
            raise ValueError(f"groups column '{groups}' not found in dataframe")
        if time_col not in df.columns:
            raise ValueError(f"time_col column '{time_col}' not found in dataframe")
        return _group_time_split(df, groups, time_col, ratios, seed)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from: random, group, time, group_time")