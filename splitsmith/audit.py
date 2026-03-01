"""Leakage auditing for splitsmith.

Checks a completed split for common data leakage patterns and integrity issues.
Returns a structured LeakageReport with actionable findings and evidence.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple, Dict

import numpy as np
import pandas as pd

from .types import Finding, LeakageReport, SplitResult

_MAX_EVIDENCE_ROWS = 5  # cap example indices/rows shown in evidence


def audit(
    df: pd.DataFrame,
    split_result: SplitResult,
    target: str,
    *,
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
) -> LeakageReport:
    """Audit a split for leakage and integrity issues.

    Parameters
    ----------
    df : pd.DataFrame – the original dataset that was split.
    split_result : SplitResult – output of splitsmith.split().
    target : str – target column name.
    groups : str, optional – column whose values should not appear in more
        than one split (entity / group leakage check).
    time_col : str, optional – datetime-like column; checks that train
        timestamps don't exceed the earliest val/test timestamp.

    Returns
    -------
    LeakageReport
    """
    # --- Input validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(split_result, SplitResult):
        raise TypeError("split_result must be a SplitResult")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if groups is not None and groups not in df.columns:
        raise ValueError(f"groups column '{groups}' not found in DataFrame")
    if time_col is not None and time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in DataFrame")

    report = LeakageReport()

    # 1. Index overlap check
    _check_overlap(split_result, report)

    # 2. Duplicate rows check
    _check_duplicates(df, split_result, report)

    # 3. Group leakage check (only if groups column provided)
    if groups is not None:
        _check_group_leakage(df, split_result, groups, report)

    # 4. Time leakage check (only if time_col provided)
    if time_col is not None:
        _check_time_leakage(df, split_result, time_col, report)

    return report


# ---------------------------------------------------------------------------
# Check 1: Index overlap
# ---------------------------------------------------------------------------

def _check_overlap(sr: SplitResult, report: LeakageReport) -> None:
    """Every index must appear in exactly one split."""
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
                id="index_overlap",
                severity="error",
                title=f"Index overlap: {a} ∩ {b}",
                details=f"{len(overlap)} index(es) appear in both {a} and {b}.",
                evidence={"count": len(overlap), "examples": examples},
            ))

    if not found_overlap:
        report.add(Finding(
            id="index_overlap",
            severity="info",
            title="No index overlap",
            details="All split indices are mutually exclusive.",
        ))


# ---------------------------------------------------------------------------
# Check 2: Duplicate rows
# ---------------------------------------------------------------------------

def _check_duplicates(
    df: pd.DataFrame, sr: SplitResult, report: LeakageReport
) -> None:
    """Find exact duplicate rows within and across splits."""
    # Map each index to its split name
    idx_to_split: Dict[int, str] = {}
    for idx in sr.train_idx.tolist():
        idx_to_split[idx] = "train"
    for idx in sr.val_idx.tolist():
        idx_to_split[idx] = "val"
    for idx in sr.test_idx.tolist():
        idx_to_split[idx] = "test"

    # Hash each row for fast comparison
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    hash_to_indices: Dict[int, List[int]] = {}
    for idx, h in row_hashes.items():
        hash_to_indices.setdefault(int(h), []).append(int(idx))

    # Find groups of duplicate rows
    dup_groups = [indices for indices in hash_to_indices.values() if len(indices) > 1]

    if not dup_groups:
        report.add(Finding(
            id="duplicate_rows",
            severity="info",
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
            id="duplicate_rows",
            severity="error",
            title="Cross-split duplicate rows",
            details=(
                f"{len(cross_split)} group(s) of identical rows span multiple "
                f"splits ({sum(len(g) for g in cross_split)} rows total). "
                f"Identical rows in different splits cause direct data leakage."
            ),
            evidence={
                "n_groups": len(cross_split),
                "n_rows": sum(len(g) for g in cross_split),
                "examples": examples,
            },
        ))

    if within_split:
        examples = []
        for gi in within_split[:_MAX_EVIDENCE_ROWS]:
            splits = sorted({idx_to_split[i] for i in gi if i in idx_to_split})
            examples.append({"indices": gi[:_MAX_EVIDENCE_ROWS], "splits": splits})
        report.add(Finding(
            id="duplicate_rows",
            severity="warn",
            title="Within-split duplicate rows",
            details=(
                f"{len(within_split)} group(s) of identical rows found within "
                f"the same split ({sum(len(g) for g in within_split)} rows total). "
                f"Usually harmless but may indicate data quality issues."
            ),
            evidence={
                "n_groups": len(within_split),
                "n_rows": sum(len(g) for g in within_split),
                "examples": examples,
            },
        ))


# ---------------------------------------------------------------------------
# Check 3: Group / entity leakage
# ---------------------------------------------------------------------------

def _check_group_leakage(
    df: pd.DataFrame, sr: SplitResult, groups: str, report: LeakageReport
) -> None:
    """Check that no group value appears in more than one split."""
    train_groups = set(df.iloc[sr.train_idx][groups].unique())
    val_groups = set(df.iloc[sr.val_idx][groups].unique())
    test_groups = set(df.iloc[sr.test_idx][groups].unique())

    pairs = [
        ("train", "val", train_groups, val_groups),
        ("train", "test", train_groups, test_groups),
        ("val", "test", val_groups, test_groups),
    ]
    found_leakage = False

    for a_name, b_name, a_set, b_set in pairs:
        shared = a_set & b_set
        if shared:
            found_leakage = True
            examples = sorted(shared, key=str)[:_MAX_EVIDENCE_ROWS]
            report.add(Finding(
                id="group_leakage",
                severity="error",
                title=f"Group leakage: {a_name} ∩ {b_name}",
                details=(
                    f"{len(shared)} group(s) from column '{groups}' appear in "
                    f"both {a_name} and {b_name}."
                ),
                evidence={"count": len(shared), "examples": examples},
            ))

    if not found_leakage:
        report.add(Finding(
            id="group_leakage",
            severity="info",
            title=f"No group leakage on '{groups}'",
            details="Each group value appears in exactly one split.",
        ))


# ---------------------------------------------------------------------------
# Check 4: Time leakage
# ---------------------------------------------------------------------------

def _check_time_leakage(
    df: pd.DataFrame, sr: SplitResult, time_col: str, report: LeakageReport
) -> None:
    """Check that train timestamps don't exceed val/test boundaries."""
    ts = pd.to_datetime(df[time_col])

    train_ts = ts.iloc[sr.train_idx]
    val_ts = ts.iloc[sr.val_idx]
    test_ts = ts.iloc[sr.test_idx]

    train_max = train_ts.max()
    val_min = val_ts.min()
    val_max = val_ts.max()
    test_min = test_ts.min()

    issues_found = False

    # Train leaking into val
    if train_max > val_min:
        n_violating = int((train_ts > val_min).sum())
        report.add(Finding(
            id="time_leakage",
            severity="error",
            title="Time leakage: train → val",
            details=(
                f"Train max ({train_max.isoformat()}) > val min "
                f"({val_min.isoformat()}). {n_violating} train row(s) "
                f"have timestamps after the val period starts."
            ),
            evidence={
                "pair": "train → val",
                "train_max": train_max.isoformat(),
                "val_min": val_min.isoformat(),
                "n_violating_rows": n_violating,
            },
        ))
        issues_found = True

    # Train leaking into test
    if train_max > test_min:
        n_violating = int((train_ts > test_min).sum())
        report.add(Finding(
            id="time_leakage",
            severity="error",
            title="Time leakage: train → test",
            details=(
                f"Train max ({train_max.isoformat()}) > test min "
                f"({test_min.isoformat()}). {n_violating} train row(s) "
                f"have timestamps after the test period starts."
            ),
            evidence={
                "pair": "train → test",
                "train_max": train_max.isoformat(),
                "test_min": test_min.isoformat(),
                "n_violating_rows": n_violating,
            },
        ))
        issues_found = True

    # Val leaking into test (less severe)
    if val_max > test_min:
        n_violating = int((val_ts > test_min).sum())
        report.add(Finding(
            id="time_leakage",
            severity="warn",
            title="Time leakage: val → test",
            details=(
                f"Val max ({val_max.isoformat()}) > test min "
                f"({test_min.isoformat()}). {n_violating} val row(s) "
                f"have timestamps after the test period starts."
            ),
            evidence={
                "pair": "val → test",
                "val_max": val_max.isoformat(),
                "test_min": test_min.isoformat(),
                "n_violating_rows": n_violating,
            },
        ))
        issues_found = True

    if not issues_found:
        report.add(Finding(
            id="time_leakage",
            severity="info",
            title=f"No time leakage on '{time_col}'",
            details="Temporal ordering is clean: train < val < test.",
        ))
