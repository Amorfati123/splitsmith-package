"""One-call diagnostic report card for splitsmith.

Runs feasibility analysis, splits the data, audits the result, computes
the leakage score, and returns a comprehensive structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .types import SplitResult, LeakageReport
from .split import split
from .audit import audit
from .score import compute_score, LeakageScore
from .feasibility import check_feasibility, FeasibilityReport


@dataclass
class DiagnosticReport:
    """Comprehensive split quality assessment.

    Produced by diagnose(). Contains everything a user needs to evaluate
    whether their split is suitable for training and evaluation.
    """
    feasibility: FeasibilityReport
    split_result: Optional[SplitResult]
    audit_report: Optional[LeakageReport]
    leakage_score: Optional[LeakageScore]
    recommendations: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "feasible" if self.feasibility.feasible else "infeasible"
        grade = self.leakage_score.grade if self.leakage_score else "N/A"
        score = self.leakage_score.score if self.leakage_score else 0.0
        return f"DiagnosticReport({status}, score={score:.3f}, grade='{grade}')"


def diagnose(
    df: pd.DataFrame,
    target: str,
    *,
    strategy: str = "random",
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    ratios: tuple = (0.7, 0.15, 0.15),
    seed: int = 42,
    stratify: Optional[bool] = None,
    balance_by: str = "groups",
    gap: int = 0,
    embargo: int = 0,
    min_samples_per_class: Optional[int] = None,
    check_drift: bool = True,
    check_feature_leakage: bool = True,
    near_duplicate_columns: Optional[Sequence[str]] = None,
    near_duplicate_tolerance: float = 0.0,
    embeddings: Optional[np.ndarray] = None,
    similarity_threshold: float = 0.9,
) -> DiagnosticReport:
    """Run a complete diagnostic: feasibility, split, audit, score.

    This is the capstone function that ties every splitsmith capability
    into a single call. Returns a DiagnosticReport with feasibility analysis,
    the split result, a full audit report, a leakage score, and actionable
    recommendations.

    Parameters
    ----------
    df : DataFrame
    target : target column name
    strategy, groups, time_col, ratios, seed, stratify, balance_by,
        gap, embargo, min_samples_per_class : passed to split()
    check_drift, check_feature_leakage : passed to audit()
    near_duplicate_columns, near_duplicate_tolerance : passed to audit()
    embeddings : optional embedding matrix for similarity leakage check
    similarity_threshold : cosine similarity threshold for embedding check
    """
    recommendations = []

    # step 1: feasibility
    feasibility = check_feasibility(
        df, target, strategy=strategy, groups=groups, time_col=time_col,
        ratios=ratios, stratify=bool(stratify) if stratify is not None else False,
        min_samples_per_class=min_samples_per_class, gap=gap,
    )

    if not feasibility.feasible:
        for issue in feasibility.issues:
            if issue.severity == "error" and issue.suggestion:
                recommendations.append(issue.suggestion)
        return DiagnosticReport(
            feasibility=feasibility,
            split_result=None,
            audit_report=None,
            leakage_score=None,
            recommendations=recommendations,
            summary={"status": "infeasible", "reason": "pre-split checks failed"},
        )

    # collect feasibility warnings as recommendations
    for issue in feasibility.issues:
        if issue.severity == "warn" and issue.suggestion:
            recommendations.append(issue.suggestion)

    # step 2: split
    split_result = split(
        df, target=target, strategy=strategy, groups=groups,
        time_col=time_col, ratios=ratios, seed=seed, stratify=stratify,
        balance_by=balance_by, gap=gap, embargo=embargo,
        min_samples_per_class=min_samples_per_class,
    )

    # step 3: audit
    groups_arg = groups
    if groups is not None and strategy in ("group", "group_time"):
        groups_arg = groups

    audit_report = audit(
        df, split_result, target,
        groups=groups_arg,
        time_col=time_col if strategy in ("time", "group_time") else None,
        near_duplicate_columns=near_duplicate_columns,
        near_duplicate_tolerance=near_duplicate_tolerance,
        check_drift=check_drift,
        check_feature_leakage=check_feature_leakage,
    )

    # step 3b: similarity check if embeddings provided
    if embeddings is not None:
        from .similarity import check_similarity_leakage
        sim_findings = check_similarity_leakage(
            split_result, embeddings, threshold=similarity_threshold,
        )
        for f in sim_findings:
            audit_report.add(f)

    # step 4: score
    leakage_score = compute_score(audit_report)

    # step 5: generate recommendations from audit
    for finding in audit_report.findings:
        if finding.severity == "error":
            if finding.id == "index_overlap":
                recommendations.append("Split has index overlap. This is a serious bug.")
            elif finding.id == "duplicate_rows":
                recommendations.append(
                    "Identical rows appear in different splits. "
                    "Consider deduplication before splitting."
                )
            elif finding.id == "group_leakage":
                recommendations.append(
                    "Same group appears in multiple splits. "
                    "Use strategy='group' to enforce group exclusivity."
                )
            elif finding.id == "time_leakage":
                recommendations.append(
                    "Temporal ordering is violated. Use strategy='time' "
                    "or add gap/embargo to separate splits."
                )
        elif finding.severity == "warn":
            if finding.id == "similarity_leakage":
                count = finding.evidence.get("count", 0)
                recommendations.append(
                    f"{count} val/test samples are similar to train in embedding space. "
                    f"Consider similarity_split() to prevent this."
                )
            elif finding.id == "feature_leakage":
                recommendations.append(
                    "Some features have suspiciously high correlation with the target. "
                    "Review flagged columns for potential data leakage."
                )
            elif finding.id == "label_drift":
                recommendations.append(
                    "Label prevalence differs between splits. "
                    "Consider stratified splitting."
                )

    # deduplicate recommendations
    seen = set()
    unique_recs = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    summary = {
        "status": "complete",
        "n_rows": len(df),
        "strategy": strategy,
        "leakage_score": leakage_score.score,
        "grade": leakage_score.grade,
        "audit_ok": audit_report.ok,
        "n_findings": len(audit_report.findings),
        "n_recommendations": len(unique_recs),
    }

    return DiagnosticReport(
        feasibility=feasibility,
        split_result=split_result,
        audit_report=audit_report,
        leakage_score=leakage_score,
        recommendations=unique_recs,
        summary=summary,
    )
