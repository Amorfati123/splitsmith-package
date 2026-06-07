"""Pre-split feasibility analysis.

Analyzes whether a requested splitting configuration is achievable before
actually splitting, and reports exactly what constraints cannot be met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeasibilityIssue:
    """A single feasibility problem or warning."""
    severity: str  # "error" (impossible) | "warn" (risky but possible)
    category: str  # "data_size", "groups", "stratification", "time", "class_balance"
    message: str
    suggestion: str = ""


@dataclass
class FeasibilityReport:
    """Result of pre-split feasibility analysis."""
    feasible: bool
    issues: List[FeasibilityIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "feasible" if self.feasible else "infeasible"
        n_err = sum(1 for i in self.issues if i.severity == "error")
        n_warn = sum(1 for i in self.issues if i.severity == "warn")
        return f"FeasibilityReport({status}, {n_err} error(s), {n_warn} warning(s))"


def check_feasibility(
    df: pd.DataFrame,
    target: str,
    *,
    strategy: str = "random",
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    ratios: tuple = (0.7, 0.15, 0.15),
    stratify: bool = False,
    min_samples_per_class: Optional[int] = None,
    gap: int = 0,
) -> FeasibilityReport:
    """Analyze whether a splitting configuration is achievable.

    Checks data size, group counts, class distributions, temporal ordering,
    and other constraints before splitting. Reports what will work, what won't,
    and what's risky.

    Parameters
    ----------
    df : DataFrame to be split
    target : target column name
    strategy : "random", "group", "time", or "group_time"
    groups : group column name
    time_col : time column name
    ratios : target split ratios
    stratify : whether stratification will be requested
    min_samples_per_class : minimum class count per split
    gap : embargo gap size
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")

    issues: List[FeasibilityIssue] = []
    n = len(df)
    y = df[target]
    class_counts = y.value_counts(dropna=False)

    data_summary: Dict[str, Any] = {
        "n_rows": n,
        "n_classes": len(class_counts),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "strategy": strategy,
    }

    # basic data size
    if n < 3:
        issues.append(FeasibilityIssue(
            severity="error", category="data_size",
            message=f"Dataset has only {n} rows, need at least 3.",
            suggestion="Add more data.",
        ))

    # target split sizes
    if n >= 3:
        expected = {
            "train": int(np.floor(ratios[0] * n)),
            "val": int(np.floor(ratios[1] * n)),
            "test": n - int(np.floor(ratios[0] * n)) - int(np.floor(ratios[1] * n)),
        }
        for name, size in expected.items():
            if size < 1:
                issues.append(FeasibilityIssue(
                    severity="error", category="data_size",
                    message=f"Ratio {ratios} with {n} rows gives {name}={size} (need >= 1).",
                    suggestion="Adjust ratios or add more data.",
                ))
        data_summary["expected_sizes"] = expected

    # group checks
    if strategy in ("group", "group_time"):
        if groups is None:
            issues.append(FeasibilityIssue(
                severity="error", category="groups",
                message=f"strategy='{strategy}' requires a groups column.",
                suggestion="Pass groups='column_name'.",
            ))
        elif groups not in df.columns:
            issues.append(FeasibilityIssue(
                severity="error", category="groups",
                message=f"groups column '{groups}' not found in DataFrame.",
            ))
        else:
            unique_groups = df[groups].unique()
            n_groups = len(unique_groups)
            data_summary["n_groups"] = n_groups

            group_sizes = df[groups].value_counts()
            data_summary["group_size_range"] = {
                "min": int(group_sizes.min()),
                "max": int(group_sizes.max()),
                "median": float(group_sizes.median()),
            }

            usable_groups = n_groups - 2 * gap if strategy == "group_time" else n_groups
            if usable_groups < 3:
                issues.append(FeasibilityIssue(
                    severity="error", category="groups",
                    message=(
                        f"Only {n_groups} groups (usable after gap={gap}: {usable_groups}), "
                        f"need at least 3."
                    ),
                    suggestion="Reduce gap or add more groups.",
                ))

            # check group size variance
            cv = group_sizes.std() / group_sizes.mean() if group_sizes.mean() > 0 else 0
            if cv > 1.0:
                issues.append(FeasibilityIssue(
                    severity="warn", category="groups",
                    message=(
                        f"Group sizes vary a lot (CV={cv:.2f}, range {int(group_sizes.min())}"
                        f"-{int(group_sizes.max())}). Row ratios may drift from targets."
                    ),
                    suggestion="Consider balance_by='rows' for better row-level balance.",
                ))

    # time checks
    if strategy in ("time", "group_time"):
        if time_col is None:
            issues.append(FeasibilityIssue(
                severity="error", category="time",
                message=f"strategy='{strategy}' requires a time_col.",
                suggestion="Pass time_col='column_name'.",
            ))
        elif time_col not in df.columns:
            issues.append(FeasibilityIssue(
                severity="error", category="time",
                message=f"time_col '{time_col}' not found in DataFrame.",
            ))
        else:
            try:
                ts = pd.to_datetime(df[time_col])
                data_summary["time_range"] = {
                    "min": ts.min().isoformat(),
                    "max": ts.max().isoformat(),
                }
                n_unique_times = ts.nunique()
                data_summary["n_unique_timestamps"] = n_unique_times

                if strategy == "time":
                    usable = n - 2 * gap
                    if usable < 3:
                        issues.append(FeasibilityIssue(
                            severity="error", category="time",
                            message=f"After gap={gap}, only {usable} rows usable (need >= 3).",
                            suggestion="Reduce gap or add more data.",
                        ))
            except Exception:
                issues.append(FeasibilityIssue(
                    severity="error", category="time",
                    message=f"Column '{time_col}' cannot be parsed as datetime.",
                    suggestion="Ensure the column contains valid date/time values.",
                ))

    # stratification feasibility
    if stratify:
        for label, count in class_counts.items():
            if count < 3:
                issues.append(FeasibilityIssue(
                    severity="error", category="stratification",
                    message=(
                        f"Class {label!r} has only {count} sample(s), "
                        f"cannot stratify into 3 splits."
                    ),
                    suggestion="Merge rare classes or use stratify=False.",
                ))

        # check if stratified group split is feasible
        if strategy == "group" and groups is not None and groups in df.columns:
            for label, count in class_counts.items():
                groups_with_label = df[df[target] == label][groups].nunique()
                if groups_with_label < 3:
                    issues.append(FeasibilityIssue(
                        severity="warn", category="stratification",
                        message=(
                            f"Class {label!r} appears in only {groups_with_label} group(s). "
                            f"Stratified group split may concentrate this class."
                        ),
                        suggestion="Consider whether stratification is appropriate here.",
                    ))

    # min_samples_per_class feasibility
    if min_samples_per_class is not None and min_samples_per_class > 0:
        required_total = min_samples_per_class * 3  # at least this many across 3 splits
        for label, count in class_counts.items():
            if count < required_total:
                issues.append(FeasibilityIssue(
                    severity="warn" if count >= min_samples_per_class else "error",
                    category="class_balance",
                    message=(
                        f"Class {label!r} has {count} samples but "
                        f"min_samples_per_class={min_samples_per_class} requires "
                        f"at least {required_total} total (3 splits x {min_samples_per_class})."
                    ),
                    suggestion="Lower min_samples_per_class or add more data for this class.",
                ))

    feasible = not any(i.severity == "error" for i in issues)
    return FeasibilityReport(feasible=feasible, issues=issues, summary=data_summary)
