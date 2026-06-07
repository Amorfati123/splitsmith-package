"""Tests for pre-split feasibility analysis."""

import numpy as np
import pandas as pd
import pytest

from splitsmith.feasibility import check_feasibility, FeasibilityReport


def _simple_df(n=100):
    return pd.DataFrame({
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


def _grouped_df(n_groups=10, rows_per=5):
    rows = []
    for i in range(n_groups):
        for j in range(rows_per):
            rows.append({"group_id": f"g{i}", "feature": i * rows_per + j, "target": j % 2})
    return pd.DataFrame(rows)


class TestBasicFeasibility:
    def test_normal_data_is_feasible(self):
        df = _simple_df()
        report = check_feasibility(df, "target")
        assert report.feasible is True
        assert isinstance(report, FeasibilityReport)

    def test_too_few_rows(self):
        df = pd.DataFrame({"target": [0, 1]})
        report = check_feasibility(df, "target")
        assert report.feasible is False
        assert any(i.category == "data_size" for i in report.issues)

    def test_missing_target_raises(self):
        df = _simple_df()
        with pytest.raises(ValueError, match="target"):
            check_feasibility(df, "nonexistent")

    def test_summary_has_row_count(self):
        df = _simple_df()
        report = check_feasibility(df, "target")
        assert report.summary["n_rows"] == 100

    def test_summary_has_class_counts(self):
        df = _simple_df()
        report = check_feasibility(df, "target")
        assert "class_counts" in report.summary


class TestGroupFeasibility:
    def test_enough_groups(self):
        df = _grouped_df(n_groups=10)
        report = check_feasibility(df, "target", strategy="group", groups="group_id")
        assert report.feasible is True

    def test_too_few_groups(self):
        df = _grouped_df(n_groups=2)
        report = check_feasibility(df, "target", strategy="group", groups="group_id")
        assert report.feasible is False
        assert any(i.category == "groups" for i in report.issues)

    def test_missing_groups_col(self):
        df = _simple_df()
        report = check_feasibility(df, "target", strategy="group", groups="nonexistent")
        assert report.feasible is False

    def test_no_groups_param(self):
        df = _simple_df()
        report = check_feasibility(df, "target", strategy="group")
        assert report.feasible is False

    def test_uneven_groups_warns(self):
        rows = []
        for j in range(100):
            rows.append({"group_id": "big", "feature": j, "target": j % 2})
        for i in range(1, 10):
            rows.append({"group_id": f"g{i}", "feature": 100 + i, "target": i % 2})
        df = pd.DataFrame(rows)
        report = check_feasibility(df, "target", strategy="group", groups="group_id")
        warns = [i for i in report.issues if i.severity == "warn" and i.category == "groups"]
        assert len(warns) >= 1

    def test_gap_reduces_usable_groups(self):
        df = _grouped_df(n_groups=4)
        report = check_feasibility(df, "target", strategy="group_time",
                                   groups="group_id", time_col="feature", gap=2)
        assert report.feasible is False


class TestStratificationFeasibility:
    def test_enough_samples_per_class(self):
        df = _simple_df()
        report = check_feasibility(df, "target", stratify=True)
        assert report.feasible is True

    def test_rare_class_blocks_stratification(self):
        targets = [0] * 50 + [1] * 2
        df = pd.DataFrame({"feature": range(52), "target": targets})
        report = check_feasibility(df, "target", stratify=True)
        assert report.feasible is False
        assert any(i.category == "stratification" for i in report.issues)

    def test_min_samples_infeasible(self):
        targets = [0] * 10 + [1] * 4
        df = pd.DataFrame({"feature": range(14), "target": targets})
        report = check_feasibility(df, "target", min_samples_per_class=5)
        issues = [i for i in report.issues if i.category == "class_balance"]
        assert len(issues) >= 1


class TestTimeFeasibility:
    def test_time_col_missing(self):
        df = _simple_df()
        report = check_feasibility(df, "target", strategy="time")
        assert report.feasible is False

    def test_time_col_not_in_df(self):
        df = _simple_df()
        report = check_feasibility(df, "target", strategy="time", time_col="nonexistent")
        assert report.feasible is False

    def test_gap_too_large(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "target": [0, 1] * 5,
        })
        report = check_feasibility(df, "target", strategy="time",
                                   time_col="timestamp", gap=5)
        assert report.feasible is False

    def test_valid_time_data(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "target": [0, 1] * 50,
        })
        report = check_feasibility(df, "target", strategy="time", time_col="timestamp")
        assert report.feasible is True
        assert "time_range" in report.summary


class TestRepr:
    def test_repr_feasible(self):
        df = _simple_df()
        report = check_feasibility(df, "target")
        assert "feasible" in repr(report)

    def test_repr_infeasible(self):
        df = pd.DataFrame({"target": [0]})
        report = check_feasibility(df, "target")
        assert "infeasible" in repr(report)
