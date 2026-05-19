"""Tests for leakage scoring."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import audit, SplitResult
from splitsmith.score import compute_score, LeakageScore


def _make_result(train, val, test):
    return SplitResult(
        train_idx=np.array(train, dtype=int),
        val_idx=np.array(val, dtype=int),
        test_idx=np.array(test, dtype=int),
    )


class TestLeakageScore:
    def test_clean_split_scores_high(self):
        df = pd.DataFrame({
            "feature": range(60),
            "target": [0, 1] * 30,
        })
        sr = _make_result(list(range(42)), list(range(42, 51)), list(range(51, 60)))
        report = audit(df, sr, "target")
        score = compute_score(report)
        assert score.score >= 0.9
        assert score.grade == "A"
        assert score.ok is True

    def test_overlapping_split_scores_low(self):
        # manually create a bad split with overlap
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(
            id="index_overlap", severity="error",
            title="Index overlap: train and val",
            details="5 overlapping indices",
        ))
        score = compute_score(report)
        assert score.score < 1.0
        assert score.components["overlap"] < 1.0

    def test_group_leakage_deduction(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(
            id="group_leakage", severity="error",
            title="Group leakage",
            details="Groups leaked",
        ))
        score = compute_score(report)
        assert score.components["group_leakage"] < 1.0
        assert score.score < 1.0

    def test_multiple_errors_accumulate(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(id="index_overlap", severity="error", title="a", details="b"))
        report.add(Finding(id="group_leakage", severity="error", title="c", details="d"))
        report.add(Finding(id="time_leakage", severity="error", title="e", details="f"))
        score = compute_score(report)
        assert score.score < 0.5
        assert score.grade in ("D", "F")

    def test_warnings_are_mild(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(id="near_duplicates", severity="warn", title="a", details="b"))
        score = compute_score(report)
        assert score.score >= 0.8

    def test_info_findings_dont_affect_score(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(id="index_overlap", severity="info", title="clean", details="ok"))
        report.add(Finding(id="duplicate_rows", severity="info", title="clean", details="ok"))
        score = compute_score(report)
        assert score.score == 1.0
        assert score.grade == "A"

    def test_score_never_below_zero(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        for _ in range(20):
            report.add(Finding(id="index_overlap", severity="error", title="a", details="b"))
        score = compute_score(report)
        assert score.score >= 0.0

    def test_score_never_above_one(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        score = compute_score(report)
        assert score.score <= 1.0

    def test_details_list_populated(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(id="index_overlap", severity="error", title="overlap", details="bad"))
        score = compute_score(report)
        assert len(score.details) >= 1
        assert "overlap" in score.details[0].lower()

    def test_grade_boundaries(self):
        from splitsmith.score import _grade_from_score
        assert _grade_from_score(1.0) == "A"
        assert _grade_from_score(0.9) == "A"
        assert _grade_from_score(0.89) == "B"
        assert _grade_from_score(0.75) == "B"
        assert _grade_from_score(0.74) == "C"
        assert _grade_from_score(0.5) == "C"
        assert _grade_from_score(0.49) == "D"
        assert _grade_from_score(0.25) == "D"
        assert _grade_from_score(0.24) == "F"
        assert _grade_from_score(0.0) == "F"

    def test_ok_property(self):
        from splitsmith.score import _grade_from_score
        ls_pass = LeakageScore(score=0.8, grade="B")
        ls_fail = LeakageScore(score=0.3, grade="D")
        assert ls_pass.ok is True
        assert ls_fail.ok is False

    def test_repr(self):
        ls = LeakageScore(score=0.85, grade="B")
        assert "0.850" in repr(ls)
        assert "B" in repr(ls)

    def test_integration_with_full_audit(self):
        """Score works end-to-end with a real audit."""
        df = pd.DataFrame({
            "f1": range(100),
            "group": [f"g{i // 10}" for i in range(100)],
            "target": [0, 1] * 50,
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target", groups="group")
        score = compute_score(report)
        assert isinstance(score, LeakageScore)
        assert 0.0 <= score.score <= 1.0

    def test_feature_leakage_warn_deduction(self):
        from splitsmith.types import LeakageReport, Finding
        report = LeakageReport()
        report.add(Finding(id="feature_leakage", severity="warn",
                           title="leak", details="col correlates"))
        score = compute_score(report)
        assert score.components["feature_leakage"] < 1.0
