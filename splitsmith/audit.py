"""Leakage auditing for splitsmith."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .types import CVResult, Finding, LeakageReport, SplitResult

_MAX_EVIDENCE_ROWS = 5


# Safe hashing helpers

def _safe_serialize(val: Any) -> str:
    """Convert a single cell value to a deterministic string for hashing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "__NaN__"
    if isinstance(val, np.ndarray):
        return f"ndarray:{val.tobytes().hex()}"
    if isinstance(val, (list, tuple)):
        try:
            return json.dumps(val, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(val)
    if isinstance(val, dict):
        try:
            return json.dumps(val, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def _has_unhashable_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
    cols = columns if columns is not None else df.columns.tolist()
    for col in cols:
        try:
            pd.util.hash_pandas_object(df[[col]], index=False)
        except (TypeError, ValueError):
            return True
    return False


def _hash_rows(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    unhashable_policy: str = "serialize",
) -> pd.Series:
    """Per-row hashes, handling unhashable columns gracefully."""
    sub = df[columns] if columns is not None else df

    try:
        return pd.util.hash_pandas_object(sub, index=False)
    except (TypeError, ValueError):
        pass

    if unhashable_policy == "error":
        raise TypeError(
            "DataFrame contains unhashable column types. "
            "Use ignore_columns or duplicate_subset to restrict checked columns, "
            "or set unhashable_policy='serialize' or 'skip'."
        )

    if unhashable_policy == "skip":
        safe_cols = []
        for col in sub.columns:
            try:
                pd.util.hash_pandas_object(sub[[col]], index=False)
                safe_cols.append(col)
            except (TypeError, ValueError):
                pass
        if not safe_cols:
            return pd.Series(range(len(sub)), index=sub.index)
        return pd.util.hash_pandas_object(sub[safe_cols], index=False)

    hashes = []
    for idx in range(len(sub)):
        row_str = "|".join(_safe_serialize(sub.iloc[idx, c]) for c in range(sub.shape[1]))
        h = int(hashlib.sha256(row_str.encode("utf-8")).hexdigest()[:16], 16)
        hashes.append(h)
    return pd.Series(hashes, index=sub.index)


def _resolve_dup_columns(
    df: pd.DataFrame,
    ignore_columns: Optional[Sequence[str]],
    duplicate_subset: Optional[Sequence[str]],
) -> Optional[List[str]]:
    if duplicate_subset is not None:
        cols = list(duplicate_subset)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"duplicate_subset columns not found in DataFrame: {missing}")
        return cols
    if ignore_columns is not None:
        ignore = set(ignore_columns)
        missing = [c for c in ignore if c not in df.columns]
        if missing:
            raise ValueError(f"ignore_columns not found in DataFrame: {missing}")
        return [c for c in df.columns if c not in ignore]
    return None


def _compute_psi(train_values: np.ndarray, test_values: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two numeric arrays."""
    eps = 1e-6
    # Adapt bin count to smallest sample to avoid noisy estimates
    effective_bins = min(n_bins, max(3, min(len(train_values), len(test_values)) // 5))
    combined = np.concatenate([train_values, test_values])
    edges = np.histogram_bin_edges(combined, bins=effective_bins)
    train_hist, _ = np.histogram(train_values, bins=edges)
    test_hist, _ = np.histogram(test_values, bins=edges)
    train_pct = train_hist / max(len(train_values), 1) + eps
    test_pct = test_hist / max(len(test_values), 1) + eps
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def _normalize_groups(
    groups: Optional[Union[str, Sequence[str]]], df: pd.DataFrame
) -> List[str]:
    if groups is None:
        return []
    if isinstance(groups, str):
        if groups not in df.columns:
            raise ValueError(f"groups column '{groups}' not found in DataFrame")
        return [groups]
    result = list(groups)
    for g in result:
        if g not in df.columns:
            raise ValueError(f"groups column '{g}' not found in DataFrame")
    return result


# Public API

def audit(
    df: pd.DataFrame,
    split_result: SplitResult,
    target: str,
    *,
    groups: Optional[Union[str, Sequence[str]]] = None,
    time_col: Optional[str] = None,
    ignore_columns: Optional[Sequence[str]] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
    unhashable_policy: str = "serialize",
    near_duplicate_columns: Optional[Sequence[str]] = None,
    near_duplicate_tolerance: float = 0.0,
    check_drift: bool = False,
    drift_columns: Optional[Sequence[str]] = None,
    drift_threshold: float = 0.15,
    check_feature_leakage: bool = False,
    feature_leakage_threshold: float = 0.95,
) -> LeakageReport:
    """Audit a split for leakage and integrity issues.

    Parameters
    ----------
    df : DataFrame
    split_result : SplitResult from split()
    target : target column name
    groups : group column(s) for leakage check. String for single-level,
        list of strings for hierarchical checks.
    time_col : time column for time-leakage check
    ignore_columns : columns to exclude from duplicate detection
    duplicate_subset : columns to use for duplicate detection (overrides ignore_columns)
    unhashable_policy : "serialize" | "skip" | "error"
    near_duplicate_columns : columns to check for near-duplicate rows
    near_duplicate_tolerance : absolute tolerance for near-duplicate numeric matching
    check_drift : if True, report target prevalence and covariate drift
    drift_columns : specific columns to measure drift on (None = auto-select numeric)
    drift_threshold : max acceptable target prevalence shift before warning
    check_feature_leakage : if True, flag columns suspiciously correlated with target
    feature_leakage_threshold : correlation threshold above which to flag (0-1)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(split_result, SplitResult):
        raise TypeError("split_result must be a SplitResult")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if time_col is not None and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")
    if unhashable_policy not in ("serialize", "skip", "error"):
        raise ValueError(f"unhashable_policy must be 'serialize', 'skip', or 'error', got {unhashable_policy!r}")

    groups_list = _normalize_groups(groups, df)
    dup_columns = _resolve_dup_columns(df, ignore_columns, duplicate_subset)

    report = LeakageReport()

    _check_overlap(split_result, report)
    _check_duplicates(df, split_result, report, dup_columns, unhashable_policy)

    if near_duplicate_columns is not None:
        _check_near_duplicates(df, split_result, near_duplicate_columns,
                               near_duplicate_tolerance, report)

    if groups_list:
        if len(groups_list) == 1:
            _check_group_leakage(df, split_result, groups_list[0], report)
        else:
            _check_hierarchical_group_leakage(df, split_result, groups_list, report)

    if time_col is not None:
        _check_time_leakage(df, split_result, time_col, report)

    if check_drift:
        _check_label_drift(df, split_result, target, drift_threshold, report)
        _check_feature_drift(df, split_result, target, drift_columns, report)

    if check_feature_leakage:
        _check_feature_leakage(df, split_result, target, feature_leakage_threshold, report)

    return report


def audit_cv(
    df: pd.DataFrame,
    cv_result: CVResult,
    target: str,
    *,
    groups: Optional[Union[str, Sequence[str]]] = None,
    time_col: Optional[str] = None,
    ignore_columns: Optional[Sequence[str]] = None,
    duplicate_subset: Optional[Sequence[str]] = None,
    unhashable_policy: str = "serialize",
    near_duplicate_columns: Optional[Sequence[str]] = None,
    near_duplicate_tolerance: float = 0.0,
    check_drift: bool = False,
    drift_columns: Optional[Sequence[str]] = None,
    drift_threshold: float = 0.15,
    check_feature_leakage: bool = False,
    feature_leakage_threshold: float = 0.95,
) -> List[LeakageReport]:
    """Audit every fold of a CV result for leakage."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(cv_result, CVResult):
        raise TypeError("cv_result must be a CVResult")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")

    reports = []
    for fold in cv_result.folds:
        sr = SplitResult(
            train_idx=fold.train_idx,
            val_idx=fold.val_idx,
            test_idx=np.array([], dtype=int),
        )
        reports.append(audit(
            df, sr, target,
            groups=groups, time_col=time_col,
            ignore_columns=ignore_columns,
            duplicate_subset=duplicate_subset,
            unhashable_policy=unhashable_policy,
            near_duplicate_columns=near_duplicate_columns,
            near_duplicate_tolerance=near_duplicate_tolerance,
            check_drift=check_drift,
            drift_columns=drift_columns,
            drift_threshold=drift_threshold,
            check_feature_leakage=check_feature_leakage,
            feature_leakage_threshold=feature_leakage_threshold,
        ))
    return reports


def audit_cv_summary(reports: List[LeakageReport]) -> Dict[str, Any]:
    """Summarize audit results across all folds."""
    return {
        "n_folds": len(reports),
        "all_ok": all(r.ok for r in reports),
        "per_fold": [r.summary() for r in reports],
        "total_errors": sum(r.summary()["error"] for r in reports),
        "total_warnings": sum(r.summary()["warn"] for r in reports),
    }


# Check implementations

def _check_overlap(sr: SplitResult, report: LeakageReport) -> None:
    splits = {
        "train": set(sr.train_idx.tolist()),
        "val": set(sr.val_idx.tolist()),
        "test": set(sr.test_idx.tolist()),
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    found_overlap = False

    for a, b in pairs:
        overlap = splits[a] & splits[b]
        if overlap:
            found_overlap = True
            examples = sorted(overlap)[:_MAX_EVIDENCE_ROWS]
            report.add(Finding(
                id="index_overlap", severity="error",
                title=f"Index overlap: {a} & {b}",
                details=f"{len(overlap)} index(es) appear in both {a} and {b}.",
                evidence={"count": len(overlap), "examples": examples},
            ))

    if not found_overlap:
        report.add(Finding(
            id="index_overlap", severity="info",
            title="No index overlap",
            details="All split indices are mutually exclusive.",
        ))


def _check_duplicates(
    df: pd.DataFrame, sr: SplitResult, report: LeakageReport,
    columns: Optional[List[str]] = None, unhashable_policy: str = "serialize",
) -> None:
    idx_to_split: Dict[int, str] = {}
    for idx in sr.train_idx.tolist():
        idx_to_split[idx] = "train"
    for idx in sr.val_idx.tolist():
        idx_to_split[idx] = "val"
    for idx in sr.test_idx.tolist():
        idx_to_split[idx] = "test"

    row_hashes = _hash_rows(df, columns=columns, unhashable_policy=unhashable_policy)
    hash_to_indices: Dict[int, List[int]] = {}
    for idx, h in row_hashes.items():
        hash_to_indices.setdefault(int(h), []).append(int(idx))

    dup_groups = [indices for indices in hash_to_indices.values() if len(indices) > 1]

    if not dup_groups:
        report.add(Finding(
            id="duplicate_rows", severity="info",
            title="No duplicate rows",
            details="All rows across all splits are unique.",
        ))
        return

    cross_split: List[List[int]] = []
    within_split: List[List[int]] = []

    for group_indices in dup_groups:
        split_names = {idx_to_split[i] for i in group_indices if i in idx_to_split}
        if len(split_names) > 1:
            cross_split.append(group_indices)
        elif len(split_names) == 1:
            within_split.append(group_indices)

    if cross_split:
        examples = []
        for gi in cross_split[:_MAX_EVIDENCE_ROWS]:
            splits = sorted({idx_to_split[i] for i in gi if i in idx_to_split})
            examples.append({"indices": gi[:_MAX_EVIDENCE_ROWS], "splits": splits})
        report.add(Finding(
            id="duplicate_rows", severity="error",
            title="Cross-split duplicate rows",
            details=f"{len(cross_split)} group(s) of identical rows span multiple splits.",
            evidence={"n_groups": len(cross_split), "n_rows": sum(len(g) for g in cross_split), "examples": examples},
        ))

    if within_split:
        examples = []
        for gi in within_split[:_MAX_EVIDENCE_ROWS]:
            splits = sorted({idx_to_split[i] for i in gi if i in idx_to_split})
            examples.append({"indices": gi[:_MAX_EVIDENCE_ROWS], "splits": splits})
        report.add(Finding(
            id="duplicate_rows", severity="warn",
            title="Within-split duplicate rows",
            details=f"{len(within_split)} group(s) of identical rows found within the same split.",
            evidence={"n_groups": len(within_split), "n_rows": sum(len(g) for g in within_split), "examples": examples},
        ))


def _check_near_duplicates(
    df: pd.DataFrame, sr: SplitResult,
    columns: Sequence[str], tolerance: float, report: LeakageReport,
) -> None:
    """Find rows that are nearly identical on given columns across splits."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"near_duplicate_columns not found in DataFrame: {missing}")

    sub = df[list(columns)]
    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        report.add(Finding(
            id="near_duplicates", severity="info",
            title="No near-duplicate check performed",
            details="No numeric columns found in near_duplicate_columns.",
        ))
        return

    sub_numeric = sub[numeric_cols].copy()

    # Build rounded matrix for tolerance-based bucketing
    if tolerance > 0:
        for col in numeric_cols:
            sub_numeric[col] = np.round(sub_numeric[col].values / tolerance) * tolerance

    idx_to_split: Dict[int, str] = {}
    for idx in sr.train_idx.tolist():
        idx_to_split[idx] = "train"
    for idx in sr.val_idx.tolist():
        idx_to_split[idx] = "val"
    for idx in sr.test_idx.tolist():
        idx_to_split[idx] = "test"

    train_set = set(sr.train_idx.tolist())
    original_numeric = df[numeric_cols]

    row_hashes = _hash_rows(sub_numeric, unhashable_policy="serialize")
    hash_to_indices: Dict[int, List[int]] = {}
    for idx, h in row_hashes.items():
        hash_to_indices.setdefault(int(h), []).append(int(idx))

    near_dup_groups = [indices for indices in hash_to_indices.values() if len(indices) > 1]

    cross_split = []
    for group_indices in near_dup_groups:
        split_names = {idx_to_split[i] for i in group_indices if i in idx_to_split}
        if len(split_names) > 1:
            cross_split.append(group_indices)

    if cross_split:
        examples = []
        for gi in cross_split[:_MAX_EVIDENCE_ROWS]:
            # find a non-train index and its closest train index with distance
            non_train = [i for i in gi if i not in train_set]
            in_train = [i for i in gi if i in train_set]
            if non_train and in_train:
                nt_idx = non_train[0]
                tr_idx = in_train[0]
                nt_vals = original_numeric.iloc[nt_idx].values.astype(float)
                tr_vals = original_numeric.iloc[tr_idx].values.astype(float)
                dist = float(np.sqrt(np.sum((nt_vals - tr_vals) ** 2)))
                examples.append({
                    "index": nt_idx,
                    "closest_train_index": tr_idx,
                    "distance": round(dist, 6),
                    "split": idx_to_split.get(nt_idx, "unknown"),
                })
            else:
                examples.append({"indices": gi[:_MAX_EVIDENCE_ROWS]})

        report.add(Finding(
            id="near_duplicates", severity="warn",
            title="Near-duplicate rows across splits",
            details=(
                f"{len(cross_split)} group(s) of near-duplicate rows "
                f"(tolerance={tolerance}) span multiple splits."
            ),
            evidence={"count": len(cross_split), "tolerance": tolerance,
                       "columns": list(columns), "examples": examples},
        ))
    else:
        report.add(Finding(
            id="near_duplicates", severity="info",
            title="No near-duplicate rows across splits",
            details=f"No near-duplicates found (tolerance={tolerance}) on columns {list(columns)}.",
        ))


def _check_group_leakage(
    df: pd.DataFrame, sr: SplitResult, groups: str, report: LeakageReport,
) -> None:
    train_groups = set(df.iloc[sr.train_idx][groups].unique())
    val_groups = set(df.iloc[sr.val_idx][groups].unique()) if len(sr.val_idx) > 0 else set()
    test_groups = set(df.iloc[sr.test_idx][groups].unique()) if len(sr.test_idx) > 0 else set()

    pairs = [
        ("train", "val", train_groups, val_groups),
        ("train", "test", train_groups, test_groups),
        ("val", "test", val_groups, test_groups),
    ]
    found = False
    for a_name, b_name, a_set, b_set in pairs:
        shared = a_set & b_set
        if shared:
            found = True
            examples = sorted(shared, key=str)[:_MAX_EVIDENCE_ROWS]
            report.add(Finding(
                id="group_leakage", severity="error",
                title=f"Group leakage: {a_name} & {b_name}",
                details=f"{len(shared)} group(s) appear in both {a_name} and {b_name}.",
                evidence={"count": len(shared), "examples": examples},
            ))
    if not found:
        report.add(Finding(
            id="group_leakage", severity="info",
            title=f"No group leakage on '{groups}'",
            details="Each group value appears in exactly one split.",
        ))


def _check_hierarchical_group_leakage(
    df: pd.DataFrame, sr: SplitResult, groups_list: List[str], report: LeakageReport,
) -> None:
    """Check leakage at each level of a group hierarchy independently."""
    any_leakage = False

    for level_col in groups_list:
        train_vals = set(df.iloc[sr.train_idx][level_col].unique())
        val_vals = set(df.iloc[sr.val_idx][level_col].unique()) if len(sr.val_idx) > 0 else set()
        test_vals = set(df.iloc[sr.test_idx][level_col].unique()) if len(sr.test_idx) > 0 else set()

        pairs = [
            ("train", "val", train_vals, val_vals),
            ("train", "test", train_vals, test_vals),
            ("val", "test", val_vals, test_vals),
        ]

        for a_name, b_name, a_set, b_set in pairs:
            shared = a_set & b_set
            if shared:
                any_leakage = True
                examples = sorted(shared, key=str)[:_MAX_EVIDENCE_ROWS]
                report.add(Finding(
                    id="hierarchical_leakage", severity="error",
                    title=f"Hierarchical leakage on '{level_col}': {a_name} & {b_name}",
                    details=f"{len(shared)} value(s) of '{level_col}' appear in both {a_name} and {b_name}.",
                    evidence={"level": level_col, "count": len(shared), "examples": examples},
                ))

    if not any_leakage:
        report.add(Finding(
            id="hierarchical_leakage", severity="info",
            title="No hierarchical group leakage",
            details=f"All levels {groups_list} are clean across splits.",
        ))


def _check_time_leakage(
    df: pd.DataFrame, sr: SplitResult, time_col: str, report: LeakageReport,
) -> None:
    ts = pd.to_datetime(df[time_col])
    train_ts = ts.iloc[sr.train_idx]
    val_ts = ts.iloc[sr.val_idx]
    test_ts = ts.iloc[sr.test_idx]

    train_max = train_ts.max()
    val_min = val_ts.min()
    val_max = val_ts.max()
    test_min = test_ts.min()
    issues_found = False

    if train_max > val_min:
        n_v = int((train_ts > val_min).sum())
        report.add(Finding(
            id="time_leakage", severity="error",
            title="Time leakage: train > val",
            details=f"Train max ({train_max.isoformat()}) > val min ({val_min.isoformat()}).",
            evidence={"pair": "train > val", "train_max": train_max.isoformat(),
                       "val_min": val_min.isoformat(), "n_violating_rows": n_v},
        ))
        issues_found = True

    if train_max > test_min:
        n_v = int((train_ts > test_min).sum())
        report.add(Finding(
            id="time_leakage", severity="error",
            title="Time leakage: train > test",
            details=f"Train max ({train_max.isoformat()}) > test min ({test_min.isoformat()}).",
            evidence={"pair": "train > test", "train_max": train_max.isoformat(),
                       "test_min": test_min.isoformat(), "n_violating_rows": n_v},
        ))
        issues_found = True

    if val_max > test_min:
        n_v = int((val_ts > test_min).sum())
        report.add(Finding(
            id="time_leakage", severity="warn",
            title="Time leakage: val > test",
            details=f"Val max ({val_max.isoformat()}) > test min ({test_min.isoformat()}).",
            evidence={"pair": "val > test", "val_max": val_max.isoformat(),
                       "test_min": test_min.isoformat(), "n_violating_rows": n_v},
        ))
        issues_found = True

    if not issues_found:
        report.add(Finding(
            id="time_leakage", severity="info",
            title=f"No time leakage on '{time_col}'",
            details="Temporal ordering is clean: train < val < test.",
        ))


def _check_label_drift(
    df: pd.DataFrame, sr: SplitResult, target: str,
    threshold: float, report: LeakageReport,
) -> None:
    """Report target prevalence per split and warn if shifted beyond threshold."""
    y = df[target]
    prevalence = {}
    for name, idx in [("train", sr.train_idx), ("val", sr.val_idx), ("test", sr.test_idx)]:
        if len(idx) == 0:
            continue
        counts = y.iloc[idx].value_counts(dropna=False, normalize=True)
        prevalence[name] = {str(k): round(float(v), 4) for k, v in counts.items()}

    severity = "info"
    details = "Target prevalence is consistent across splits."

    if "train" in prevalence:
        max_shift = 0.0
        for split_name in ["val", "test"]:
            if split_name not in prevalence:
                continue
            for label in prevalence["train"]:
                train_p = prevalence["train"].get(label, 0)
                other_p = prevalence[split_name].get(label, 0)
                max_shift = max(max_shift, abs(train_p - other_p))

        if max_shift > threshold:
            severity = "warn"
            details = f"Target prevalence shifted by up to {max_shift:.2%} across splits."

    report.add(Finding(
        id="label_drift", severity=severity,
        title="Target prevalence by split",
        details=details,
        evidence={"prevalence": prevalence},
    ))


def _check_feature_drift(
    df: pd.DataFrame, sr: SplitResult, target: str,
    drift_columns: Optional[Sequence[str]], report: LeakageReport,
) -> None:
    """PSI-based covariate drift on numeric columns."""
    if drift_columns is not None:
        cols_to_check = list(drift_columns)
        missing = [c for c in cols_to_check if c not in df.columns]
        if missing:
            raise ValueError(f"drift_columns not found in DataFrame: {missing}")
    else:
        cols_to_check = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target
        ]

    if not cols_to_check or len(sr.train_idx) == 0:
        return

    psi_results: Dict[str, Dict[str, float]] = {}
    high_drift_cols = []

    for col in cols_to_check:
        train_vals = df[col].iloc[sr.train_idx].dropna().values.astype(float)
        if len(train_vals) < 5:
            continue
        for split_name, idx in [("val", sr.val_idx), ("test", sr.test_idx)]:
            if len(idx) == 0:
                continue
            other_vals = df[col].iloc[idx].dropna().values.astype(float)
            if len(other_vals) < 5:
                continue
            psi = _compute_psi(train_vals, other_vals)
            key = f"train_vs_{split_name}"
            if col not in psi_results:
                psi_results[col] = {}
            psi_results[col][key] = round(psi, 4)
            if psi > 0.25:
                high_drift_cols.append(f"{col} ({key}: PSI={psi:.3f})")

    if psi_results:
        if high_drift_cols:
            report.add(Finding(
                id="feature_drift", severity="warn",
                title="Significant covariate drift detected",
                details=f"{len(high_drift_cols)} column(s) show PSI > 0.25.",
                evidence={"psi": psi_results},
            ))
        else:
            report.add(Finding(
                id="feature_drift", severity="info",
                title="No significant covariate drift",
                details="All checked columns have PSI <= 0.25.",
                evidence={"psi": psi_results},
            ))


def _check_feature_leakage(
    df: pd.DataFrame, sr: SplitResult, target: str,
    threshold: float, report: LeakageReport,
) -> None:
    """Flag columns suspiciously correlated with the target."""
    y = df[target]
    train_idx = sr.train_idx

    if len(train_idx) < 10:
        return

    y_train = y.iloc[train_idx]
    suspicious = []

    # Numeric columns: Pearson correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == target:
            continue
        x_train = df[col].iloc[train_idx]
        if x_train.std() == 0 or y_train.std() == 0:
            continue
        try:
            valid = x_train.notna() & y_train.notna()
            if valid.sum() < 10:
                continue
            corr = float(np.corrcoef(
                x_train[valid].values.astype(float),
                y_train[valid].values.astype(float)
            )[0, 1])
            if abs(corr) > threshold:
                suspicious.append({
                    "column": col,
                    "reason": "high_correlation",
                    "correlation": round(corr, 4),
                })
        except (ValueError, TypeError):
            continue

    # Categorical columns: check if they perfectly separate the target
    # Only check low-cardinality categoricals (not IDs)
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in cat_cols:
        if col == target:
            continue
        col_train = df[col].iloc[train_idx]
        nunique = col_train.nunique(dropna=False)
        n = len(col_train)
        # skip high-cardinality columns (likely IDs)
        if nunique > 0.5 * n:
            continue
        try:
            # check if each category value maps to a single target value
            mapping = df[[col, target]].iloc[train_idx].groupby(col)[target].nunique()
            if (mapping == 1).all() and len(mapping) > 1:
                suspicious.append({
                    "column": col,
                    "reason": "perfect_categorical_separation",
                    "n_categories": int(nunique),
                })
        except (ValueError, TypeError):
            continue

    if suspicious:
        report.add(Finding(
            id="feature_leakage", severity="warn",
            title="Potential feature leakage detected",
            details=f"{len(suspicious)} column(s) flagged as potential leakage sources.",
            evidence={"flagged": suspicious[:10]},
        ))
    else:
        report.add(Finding(
            id="feature_leakage", severity="info",
            title="No feature leakage detected",
            details="No columns show suspiciously high target correlation or perfect separation.",
        ))
