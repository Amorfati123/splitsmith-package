import numpy as np
import pandas as pd
import pytest

from splitsmith import LeakageReport, audit_cv, audit_cv_summary, k_fold


def _simple_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": rng.standard_normal(n),
        "target": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


def _grouped_df(n_groups=20, rows_per_group=5):
    rows = []
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({"group_id": f"g{i}", "feature": i * rows_per_group + j, "target": j % 2})
    return pd.DataFrame(rows)


class TestAuditCV:
    def test_clean_basic(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=5)
        reports = audit_cv(df, cv, "target")
        assert all(r.ok for r in reports)

    def test_clean_group(self):
        df = _grouped_df()
        cv = k_fold(df, "target", k=5, strategy="group", groups="group_id")
        reports = audit_cv(df, cv, "target", groups="group_id")
        assert all(r.ok for r in reports)

    def test_returns_list(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=4)
        reports = audit_cv(df, cv, "target")
        assert isinstance(reports, list)
        assert len(reports) == 4
        assert all(isinstance(r, LeakageReport) for r in reports)

    def test_validation_errors(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=4)
        with pytest.raises(TypeError):
            audit_cv([], cv, "target")
        with pytest.raises(TypeError):
            audit_cv(df, [], "target")
        with pytest.raises(ValueError):
            audit_cv(df, cv, "missing")


class TestAuditCVSummary:
    def test_structure(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=4)
        reports = audit_cv(df, cv, "target")
        s = audit_cv_summary(reports)
        assert set(s.keys()) == {"n_folds", "all_ok", "per_fold", "total_errors", "total_warnings"}

    def test_all_ok_when_clean(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=4)
        s = audit_cv_summary(audit_cv(df, cv, "target"))
        assert s["all_ok"] is True

    def test_total_counts(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=4)
        s = audit_cv_summary(audit_cv(df, cv, "target"))
        assert s["total_errors"] == sum(p["error"] for p in s["per_fold"])
        assert s["total_warnings"] == sum(p["warn"] for p in s["per_fold"])
