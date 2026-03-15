from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .types import SplitResult
from ._meta import build_metadata


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


def _check_min_samples(y, split_idx_map, min_samples, split_names=("train", "val", "test")):
    """Check that every class has at least min_samples in every split.

    split_idx_map is a dict like {"train": array, "val": array, "test": array}.
    Raises ValueError with a diagnostic message if any class is too sparse.
    """
    if min_samples is None or min_samples <= 0:
        return

    problems = []
    for name in split_names:
        idx = split_idx_map[name]
        if len(idx) == 0:
            continue
        counts = y.iloc[idx].value_counts(dropna=False)
        for label, count in counts.items():
            if count < min_samples:
                problems.append(
                    f"class {label!r} has only {count} sample(s) in {name} "
                    f"(need at least {min_samples})"
                )

    # also check for classes completely missing from a split
    all_classes = set(y.unique())
    for name in split_names:
        idx = split_idx_map[name]
        if len(idx) == 0:
            continue
        present = set(y.iloc[idx].unique())
        missing = all_classes - present
        for label in missing:
            problems.append(
                f"class {label!r} has 0 samples in {name} "
                f"(need at least {min_samples})"
            )

    if problems:
        detail = "; ".join(problems[:5])
        if len(problems) > 5:
            detail += f" ... and {len(problems) - 5} more"
        raise ValueError(
            f"min_samples_per_class={min_samples} violated: {detail}"
        )


def _random_split(df, target, ratios, seed, stratify, min_samples_per_class):
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

    _check_min_samples(y, {"train": train_idx, "val": val_idx, "test": test_idx}, min_samples_per_class)

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


def _group_split(df, groups, ratios, seed, balance_by="groups"):
    """Split by group, no group appears in more than one split."""
    unique_groups = np.asarray(df[groups].unique())
    n_groups = len(unique_groups)

    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups, got {n_groups}")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    if balance_by == "rows":
        train_idx, val_idx, test_idx, groups_per_split = _group_split_balance_rows(
            df, groups, shuffled, ratios, rng
        )
    else:
        train_n, val_n, test_n = _validate_and_normalize_sizes(n_groups, ratios)

        train_groups = set(shuffled[:train_n].tolist())
        val_groups = set(shuffled[train_n : train_n + val_n].tolist())
        test_groups = set(shuffled[train_n + val_n :].tolist())

        group_col = df[groups]
        train_idx = np.where(group_col.isin(train_groups))[0].astype(int)
        val_idx = np.where(group_col.isin(val_groups))[0].astype(int)
        test_idx = np.where(group_col.isin(test_groups))[0].astype(int)
        groups_per_split = {"train": train_n, "val": val_n, "test": test_n}

    n = len(df)
    metadata = {
        "strategy": "group",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "balance_by": balance_by,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "achieved_ratios": {
            "train": round(len(train_idx) / n, 4),
            "val": round(len(val_idx) / n, 4),
            "test": round(len(test_idx) / n, 4),
        },
        "n_groups": n_groups,
        "groups_per_split": groups_per_split,
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, metadata=metadata)


def _group_split_balance_rows(df, groups, shuffled_groups, ratios, rng):
    """Assign groups to splits so row counts approximate the requested ratios."""
    n = len(df)
    group_col = df[groups]
    group_sizes = group_col.value_counts()

    ordered = sorted(shuffled_groups, key=lambda g: -group_sizes[g])

    targets = {
        "train": ratios[0] * n,
        "val": ratios[1] * n,
        "test": ratios[2] * n,
    }
    current = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, list] = {"train": [], "val": [], "test": []}

    for g in ordered:
        sz = group_sizes[g]
        best = max(targets, key=lambda s: targets[s] - current[s])
        assignments[best].append(g)
        current[best] += sz

    for s in ["train", "val", "test"]:
        if not assignments[s]:
            donor = max(assignments, key=lambda k: len(assignments[k]))
            if len(assignments[donor]) >= 2:
                moved = assignments[donor].pop()
                assignments[s].append(moved)

    train_idx = np.where(group_col.isin(set(assignments["train"])))[0].astype(int)
    val_idx = np.where(group_col.isin(set(assignments["val"])))[0].astype(int)
    test_idx = np.where(group_col.isin(set(assignments["test"])))[0].astype(int)

    groups_per_split = {
        "train": len(assignments["train"]),
        "val": len(assignments["val"]),
        "test": len(assignments["test"]),
    }
    return train_idx, val_idx, test_idx, groups_per_split


def _stratified_group_split(df, groups, target, ratios, seed, min_samples_per_class=None):
    """Split by group with approximate class-balance preservation."""
    unique_groups = np.asarray(df[groups].unique())
    n_groups = len(unique_groups)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups, got {n_groups}")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    y = df[target]
    labels = y.unique()
    n = len(df)
    group_col = df[groups]

    group_label_counts: dict = {}
    for g in shuffled:
        mask = group_col == g
        counts = y[mask].value_counts(dropna=False)
        group_label_counts[g] = {lab: int(counts.get(lab, 0)) for lab in labels}

    overall_dist = y.value_counts(dropna=False, normalize=True).to_dict()

    assignments: dict[str, list] = {"train": [], "val": [], "test": []}
    split_label_counts: dict[str, dict] = {
        s: {lab: 0 for lab in labels} for s in ["train", "val", "test"]
    }
    split_totals = {"train": 0, "val": 0, "test": 0}
    target_sizes = {"train": ratios[0] * n, "val": ratios[1] * n, "test": ratios[2] * n}

    for g in shuffled:
        g_counts = group_label_counts[g]
        g_total = sum(g_counts.values())

        best_split = None
        best_score = float("inf")

        for s in ["train", "val", "test"]:
            new_total = split_totals[s] + g_total
            if new_total == 0:
                score = 0.0
            else:
                size_dev = abs(new_total - target_sizes[s]) / max(target_sizes[s], 1)
                dist_dev = 0.0
                for lab in labels:
                    new_count = split_label_counts[s][lab] + g_counts.get(lab, 0)
                    actual_frac = new_count / new_total
                    target_frac = overall_dist.get(lab, 0)
                    dist_dev += abs(actual_frac - target_frac)
                score = size_dev + dist_dev

            if score < best_score:
                best_score = score
                best_split = s

        assignments[best_split].append(g)
        split_totals[best_split] += g_total
        for lab in labels:
            split_label_counts[best_split][lab] += g_counts.get(lab, 0)

    for s in ["train", "val", "test"]:
        if not assignments[s]:
            donor = max(assignments, key=lambda k: len(assignments[k]))
            if len(assignments[donor]) >= 2:
                moved = assignments[donor].pop()
                assignments[s].append(moved)

    train_idx = np.where(group_col.isin(set(assignments["train"])))[0].astype(int)
    val_idx = np.where(group_col.isin(set(assignments["val"])))[0].astype(int)
    test_idx = np.where(group_col.isin(set(assignments["test"])))[0].astype(int)

    _check_min_samples(y, {"train": train_idx, "val": val_idx, "test": test_idx}, min_samples_per_class)

    metadata = {
        "strategy": "group",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "stratified": True,
        "balance_by": "stratified_groups",
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "achieved_ratios": {
            "train": round(len(train_idx) / n, 4),
            "val": round(len(val_idx) / n, 4),
            "test": round(len(test_idx) / n, 4),
        },
        "n_groups": n_groups,
        "groups_per_split": {
            "train": len(assignments["train"]),
            "val": len(assignments["val"]),
            "test": len(assignments["test"]),
        },
        "target_counts_train": {str(k): int(v) for k, v in y.iloc[train_idx].value_counts(dropna=False).to_dict().items()},
        "target_counts_val": {str(k): int(v) for k, v in y.iloc[val_idx].value_counts(dropna=False).to_dict().items()},
        "target_counts_test": {str(k): int(v) for k, v in y.iloc[test_idx].value_counts(dropna=False).to_dict().items()},
    }

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, metadata=metadata)


def _time_split(df, time_col, ratios, seed, gap=0, embargo=0):
    """Chronological split with optional gap/embargo."""
    effective_gap = max(gap, embargo)
    n = len(df)
    time_series = pd.to_datetime(df[time_col])
    sorted_idx = np.argsort(time_series.values, kind="stable").astype(int)

    total_gap = 2 * effective_gap
    usable = n - total_gap
    if usable < 3:
        raise ValueError(
            f"After applying gap={effective_gap}, only {usable} rows remain "
            f"(need at least 3). Reduce gap or add more data."
        )

    train_n, val_n, test_n = _validate_and_normalize_sizes(usable, ratios)

    train_idx = sorted_idx[:train_n]
    val_start = train_n + effective_gap
    val_idx = sorted_idx[val_start : val_start + val_n]
    test_start = val_start + val_n + effective_gap
    test_idx = sorted_idx[test_start : test_start + test_n]

    train_times = time_series.iloc[train_idx]
    val_times = time_series.iloc[val_idx]
    test_times = time_series.iloc[test_idx]

    metadata = {
        "strategy": "time",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "gap": effective_gap,
        "n_dropped_gap": total_gap,
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


def _group_time_split(df, groups, time_col, ratios, seed, gap=0, embargo=0):
    """Split groups by their latest timestamp with optional gap/embargo."""
    effective_gap = max(gap, embargo)
    time_series = pd.to_datetime(df[time_col])
    max_times = time_series.groupby(df[groups]).max()

    group_order = pd.DataFrame({"group": max_times.index, "max_time": max_times.values})
    group_order = group_order.sort_values(["max_time", "group"], kind="mergesort")
    sorted_groups = group_order["group"].values

    n_groups = len(sorted_groups)
    if n_groups < 3:
        raise ValueError(f"Need at least 3 unique groups, got {n_groups}")

    total_gap = 2 * effective_gap
    usable = n_groups - total_gap
    if usable < 3:
        raise ValueError(
            f"After applying gap={effective_gap}, only {usable} groups remain "
            f"(need at least 3). Reduce gap or add more groups."
        )

    train_n, val_n, test_n = _validate_and_normalize_sizes(usable, ratios)

    train_groups = set(sorted_groups[:train_n].tolist())
    val_start = train_n + effective_gap
    val_groups = set(sorted_groups[val_start : val_start + val_n].tolist())
    test_start = val_start + val_n + effective_gap
    test_groups = set(sorted_groups[test_start : test_start + test_n].tolist())

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
        "gap": effective_gap,
        "n_dropped_gap_groups": total_gap,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "n_groups": n_groups,
        "groups_per_split": {"train": len(train_groups), "val": len(val_groups), "test": len(test_groups)},
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
    balance_by: str = "groups",
    gap: int = 0,
    embargo: int = 0,
    min_samples_per_class: Optional[int] = None,
) -> SplitResult:
    """Split a DataFrame into train/val/test sets.

    Parameters
    ----------
    df : DataFrame
    target : target column name
    groups : group column (required for group/group_time strategies)
    time_col : time column (required for time/group_time strategies)
    strategy : "random", "group", "time", or "group_time"
    ratios : (train, val, test) ratios summing to 1
    seed : random seed
    stratify : whether to preserve class balance
    balance_by : "groups" or "rows" for group strategy
    gap : rows/groups to drop between splits (embargo/purge)
    embargo : alias for gap
    min_samples_per_class : minimum samples of each class required in every
        split. Only checked for stratified splits. Raises ValueError if violated.
    """
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

    if not isinstance(gap, int) or gap < 0:
        raise ValueError("gap must be a non-negative integer")

    if not isinstance(embargo, int) or embargo < 0:
        raise ValueError("embargo must be a non-negative integer")

    n = len(df)
    if n < 3:
        raise ValueError("dataset must have at least 3 rows")

    if balance_by not in ("groups", "rows"):
        raise ValueError(f"balance_by must be 'groups' or 'rows', got {balance_by!r}")

    # Build reproducibility params dict
    params = {
        "target": target, "groups": groups, "time_col": time_col,
        "strategy": strategy, "ratios": tuple(ratios), "seed": seed,
        "stratify": stratify, "balance_by": balance_by,
        "gap": gap, "embargo": embargo,
        "min_samples_per_class": min_samples_per_class,
    }

    if strategy == "random":
        result = _random_split(df, target, ratios, seed, stratify, min_samples_per_class)

    elif strategy == "group":
        if groups is None:
            raise ValueError("groups parameter is required when strategy='group'")
        if groups not in df.columns:
            raise ValueError(f"groups column '{groups}' not found in dataframe")
        if stratify is True:
            result = _stratified_group_split(df, groups, target, ratios, seed, min_samples_per_class)
        else:
            result = _group_split(df, groups, ratios, seed, balance_by=balance_by)

    elif strategy == "time":
        if stratify is True:
            raise NotImplementedError("Stratified time splitting not yet supported")
        if time_col is None:
            raise ValueError("time_col parameter is required when strategy='time'")
        if time_col not in df.columns:
            raise ValueError(f"time_col column '{time_col}' not found in dataframe")
        result = _time_split(df, time_col, ratios, seed, gap=gap, embargo=embargo)

    elif strategy == "group_time":
        if stratify is True:
            raise NotImplementedError("Stratified group_time splitting not yet supported")
        if groups is None or time_col is None:
            raise ValueError("Both groups and time_col are required when strategy='group_time'")
        if groups not in df.columns:
            raise ValueError(f"groups column '{groups}' not found in dataframe")
        if time_col not in df.columns:
            raise ValueError(f"time_col column '{time_col}' not found in dataframe")
        result = _group_time_split(df, groups, time_col, ratios, seed, gap=gap, embargo=embargo)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from: random, group, time, group_time")

    # Enrich metadata with reproducibility info
    repro = build_metadata(df, params)
    result.metadata["reproducibility"] = repro

    return result


def repeated_split(
    df,
    target: str,
    n_repeats: int = 10,
    groups=None,
    time_col=None,
    strategy: str = "random",
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
    stratify: bool | None = None,
    balance_by: str = "groups",
    gap: int = 0,
    embargo: int = 0,
    min_samples_per_class: Optional[int] = None,
) -> Dict[str, Any]:
    """Run multiple holdout splits with different seeds and summarize stability.

    Returns a dict with:
        results: list of SplitResult objects
        summary: variance and consistency statistics across repeats
    """
    if not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")

    results: List[SplitResult] = []
    for i in range(n_repeats):
        r = split(
            df, target=target, groups=groups, time_col=time_col,
            strategy=strategy, ratios=ratios, seed=seed + i,
            stratify=stratify, balance_by=balance_by,
            gap=gap, embargo=embargo,
            min_samples_per_class=min_samples_per_class,
        )
        results.append(r)

    # Compute stability summary
    train_sizes = [len(r.train_idx) for r in results]
    val_sizes = [len(r.val_idx) for r in results]
    test_sizes = [len(r.test_idx) for r in results]

    summary: Dict[str, Any] = {
        "n_repeats": n_repeats,
        "strategy": strategy,
        "seed_range": [seed, seed + n_repeats - 1],
        "train_size": {"mean": float(np.mean(train_sizes)), "std": float(np.std(train_sizes))},
        "val_size": {"mean": float(np.mean(val_sizes)), "std": float(np.std(val_sizes))},
        "test_size": {"mean": float(np.mean(test_sizes)), "std": float(np.std(test_sizes))},
    }

    # If stratified, check class balance consistency
    y = df[target]
    categorical_like = (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
        or int(y.nunique(dropna=False)) <= min(50, max(2, int(0.1 * len(df))))
    )
    if categorical_like:
        train_prevalences = []
        for r in results:
            counts = y.iloc[r.train_idx].value_counts(dropna=False, normalize=True)
            train_prevalences.append(counts.to_dict())

        if train_prevalences:
            all_labels = sorted(set(k for d in train_prevalences for k in d), key=str)
            balance_stats = {}
            for label in all_labels:
                vals = [d.get(label, 0.0) for d in train_prevalences]
                balance_stats[str(label)] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                }
            summary["train_class_balance"] = balance_stats

    return {"results": results, "summary": summary}
