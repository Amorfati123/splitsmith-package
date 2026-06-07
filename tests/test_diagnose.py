"""Tests for the diagnostic report card."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import diagnose, DiagnosticReport, SplitResult
from splitsmith.score import LeakageScore
from splitsmith.feasibility import FeasibilityReport


def _simple_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "target": rng.choice([0, 1], size=n),
    })


def _grouped_df(n_groups=10, rows_per=10):
    rows = []
    for i in range(n_groups):
        for j in range(rows_per):
            rows.append({"group_id": f"g{i}", "f1": i * rows_per + j, "target": j % 2})
    return pd.DataFrame(rows)


def _time_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "f1": rng.normal(size=n),
        "target": rng.choice([0, 1], size=n),
    })


class TestDiagnoseBasic:
    def test_returns_diagnostic_report(self):
        df = _simple_df()
        report = diagnose(df, "target")
        assert isinstance(report, DiagnosticReport)

    def test_has_all_components(self):
        df = _simple_df()
        report = diagnose(df, "target")
        assert isinstance(report.feasibility, FeasibilityReport)
        assert isinstance(report.split_result, SplitResult)
        assert report.audit_report is not None
        assert isinstance(report.leakage_score, LeakageScore)

    def test_feasible_data_gets_score(self):
        df = _simple_df()
        report = diagnose(df, "target")
        assert report.leakage_score.score > 0
        assert report.summary["status"] == "complete"

    def test_infeasible_data_stops_early(self):
        df = pd.DataFrame({"target": [0, 1]})
        report = diagnose(df, "target")
        assert report.feasibility.feasible is False
        assert report.split_result is None
        assert report.leakage_score is None
        assert report.summary["status"] == "infeasible"

    def test_summary_has_grade(self):
        df = _simple_df()
        report = diagnose(df, "target")
        assert "grade" in report.summary
        assert report.summary["grade"] in ("A", "B", "C", "D", "F")

    def test_repr(self):
        df = _simple_df()
        report = diagnose(df, "target")
        r = repr(report)
        assert "score=" in r
        assert "grade=" in r


class TestDiagnoseStrategies:
    def test_group_strategy(self):
        df = _grouped_df()
        report = diagnose(df, "target", strategy="group", groups="group_id")
        assert report.feasibility.feasible is True
        assert report.split_result is not None
        assert report.summary["strategy"] == "group"

    def test_time_strategy(self):
        df = _time_df()
        report = diagnose(df, "target", strategy="time", time_col="timestamp")
        assert report.feasibility.feasible is True
        assert report.split_result is not None

    def test_group_strategy_infeasible(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "target": [0, 1, 0, 1]})
        report = diagnose(df, "target", strategy="group", groups="group")
        assert report.feasibility.feasible is False


class TestDiagnoseWithEmbeddings:
    def test_similarity_check_integrated(self):
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({"f1": range(n), "target": rng.choice([0, 1], size=n)})
        emb = rng.normal(size=(n, 8))
        # make row 0 and 35 near-identical
        emb[35] = emb[0] + rng.normal(scale=0.001, size=8)

        report = diagnose(df, "target", embeddings=emb, similarity_threshold=0.99)
        assert report.split_result is not None
        # should have similarity findings in the audit
        sim_findings = [f for f in report.audit_report.findings
                        if f.id == "similarity_leakage"]
        assert len(sim_findings) >= 1

    def test_no_embeddings_no_sim_check(self):
        df = _simple_df()
        report = diagnose(df, "target")
        sim_findings = [f for f in report.audit_report.findings
                        if f.id == "similarity_leakage"]
        assert len(sim_findings) == 0


class TestDiagnoseRecommendations:
    def test_recommendations_are_strings(self):
        df = _simple_df()
        report = diagnose(df, "target")
        for r in report.recommendations:
            assert isinstance(r, str)

    def test_infeasible_gives_suggestions(self):
        df = pd.DataFrame({"target": [0, 1]})
        report = diagnose(df, "target")
        assert len(report.recommendations) >= 1

    def test_no_duplicate_recommendations(self):
        df = _simple_df()
        report = diagnose(df, "target")
        assert len(report.recommendations) == len(set(report.recommendations))
