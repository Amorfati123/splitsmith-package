"""Leakage scoring for splitsmith.

Computes a composite 0-to-1 "split hygiene score" from audit findings,
giving a more nuanced CI signal than binary pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import LeakageReport


@dataclass
class LeakageScore:
    """Quantitative leakage risk assessment from audit findings.

    Attributes
    ----------
    score : float
        Overall split hygiene score from 0.0 (severe leakage) to 1.0 (clean).
    components : dict
        Individual scores per check category.
    grade : str
        Letter grade: A (>= 0.9), B (>= 0.75), C (>= 0.5), D (>= 0.25), F (< 0.25).
    details : list of str
        Human-readable explanations of deductions.
    """
    score: float
    components: Dict[str, float] = field(default_factory=dict)
    grade: str = "A"
    details: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if score >= 0.5 (passing)."""
        return self.score >= 0.5

    def __repr__(self) -> str:
        return f"LeakageScore(score={self.score:.3f}, grade='{self.grade}')"


def _grade_from_score(score: float) -> str:
    if score >= 0.9:
        return "A"
    elif score >= 0.75:
        return "B"
    elif score >= 0.5:
        return "C"
    elif score >= 0.25:
        return "D"
    return "F"


def compute_score(report: LeakageReport) -> LeakageScore:
    """Compute a composite leakage score from an audit report.

    Each finding category contributes to the score. Errors cause large
    deductions, warnings cause smaller deductions, info findings are neutral.

    Scoring logic:
    - Start at 1.0 (perfect)
    - index_overlap error: -0.3 per occurrence
    - duplicate_rows error (cross-split): -0.2 per occurrence
    - duplicate_rows warn (within-split): -0.05 per occurrence
    - group_leakage error: -0.25 per occurrence
    - hierarchical_leakage error: -0.25 per occurrence
    - time_leakage error: -0.25 per occurrence
    - time_leakage warn: -0.1 per occurrence
    - near_duplicates warn: -0.1 per occurrence
    - label_drift warn: -0.05 per occurrence
    - feature_drift warn: -0.05 per occurrence
    - feature_leakage warn: -0.15 per occurrence
    """
    score = 1.0
    details = []
    components = {
        "overlap": 1.0,
        "duplicates": 1.0,
        "group_leakage": 1.0,
        "time_leakage": 1.0,
        "near_duplicates": 1.0,
        "drift": 1.0,
        "feature_leakage": 1.0,
    }

    penalties = {
        ("index_overlap", "error"): ("overlap", 0.3),
        ("duplicate_rows", "error"): ("duplicates", 0.2),
        ("duplicate_rows", "warn"): ("duplicates", 0.05),
        ("group_leakage", "error"): ("group_leakage", 0.25),
        ("hierarchical_leakage", "error"): ("group_leakage", 0.25),
        ("time_leakage", "error"): ("time_leakage", 0.25),
        ("time_leakage", "warn"): ("time_leakage", 0.1),
        ("near_duplicates", "warn"): ("near_duplicates", 0.1),
        ("label_drift", "warn"): ("drift", 0.05),
        ("feature_drift", "warn"): ("drift", 0.05),
        ("feature_leakage", "warn"): ("feature_leakage", 0.15),
    }

    for finding in report.findings:
        key = (finding.id, finding.severity)
        if key in penalties:
            component, amount = penalties[key]
            components[component] = max(0.0, components[component] - amount)
            score = max(0.0, score - amount)
            details.append(
                f"-{amount:.2f}: {finding.title} ({finding.severity})"
            )

    score = max(0.0, min(1.0, score))

    # round components
    components = {k: round(v, 3) for k, v in components.items()}

    return LeakageScore(
        score=round(score, 3),
        components=components,
        grade=_grade_from_score(score),
        details=details,
    )
