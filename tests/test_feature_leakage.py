"""Tests for feature leakage detection in audit."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import SplitResult, audit


def _make_result(train, val, test):
    return SplitResult(
        train_idx=np.array(train, dtype=int),
        val_idx=np.array(val, dtype=int),
        test_idx=np.array(test, dtype=int),
    )


class TestFeatureLeakage:
    def test_perfect_correlation_flagged(self):
        """A column that is a copy of target should be flagged."""
        n = 100
        rng = np.random.default_rng(0)
        targets = rng.choice([0, 1], size=n)
        df = pd.DataFrame({
            "leaked_col": targets.copy(),
            "normal_col": rng.normal(size=n),
            "target": targets,
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target",
                       check_feature_leakage=True,
                       feature_leakage_threshold=0.95)
        warns = [f for f in report.findings if f.id == "feature_leakage" and f.severity == "warn"]
        assert len(warns) == 1
        flagged_cols = [item["column"] for item in warns[0].evidence["flagged"]]
        assert "leaked_col" in flagged_cols

    def test_uncorrelated_columns_clean(self):
        """Random columns should not be flagged."""
        n = 100
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "target": rng.choice([0, 1], size=n),
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target",
                       check_feature_leakage=True,
                       feature_leakage_threshold=0.95)
        infos = [f for f in report.findings if f.id == "feature_leakage" and f.severity == "info"]
        assert len(infos) == 1

    def test_categorical_perfect_separation(self):
        """A categorical column that perfectly predicts target should be flagged."""
        df = pd.DataFrame({
            "group": ["cat"] * 50 + ["dog"] * 50,
            "target": [0] * 50 + [1] * 50,
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target",
                       check_feature_leakage=True,
                       feature_leakage_threshold=0.95)
        warns = [f for f in report.findings if f.id == "feature_leakage" and f.severity == "warn"]
        assert len(warns) == 1
        flagged = warns[0].evidence["flagged"]
        assert any(item["column"] == "group" for item in flagged)

    def test_disabled_by_default(self):
        df = pd.DataFrame({
            "leaked": [0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        leakage_findings = [f for f in report.findings if f.id == "feature_leakage"]
        assert len(leakage_findings) == 0

    def test_threshold_controls_sensitivity(self):
        """Lower threshold flags more columns."""
        n = 100
        rng = np.random.default_rng(0)
        targets = rng.choice([0, 1], size=n)
        noise = rng.normal(size=n) * 0.3
        df = pd.DataFrame({
            "noisy_copy": targets + noise,
            "target": targets,
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report_strict = audit(df, sr, "target",
                              check_feature_leakage=True,
                              feature_leakage_threshold=0.99)
        report_loose = audit(df, sr, "target",
                             check_feature_leakage=True,
                             feature_leakage_threshold=0.5)
        strict_warns = [f for f in report_strict.findings if f.id == "feature_leakage" and f.severity == "warn"]
        loose_warns = [f for f in report_loose.findings if f.id == "feature_leakage" and f.severity == "warn"]
        assert len(loose_warns) >= len(strict_warns)

    def test_high_cardinality_categorical_skipped(self):
        """Unique ID columns should be skipped."""
        n = 100
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "unique_id": [f"id_{i}" for i in range(n)],
            "target": rng.choice([0, 1], size=n),
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target",
                       check_feature_leakage=True,
                       feature_leakage_threshold=0.95)
        infos = [f for f in report.findings if f.id == "feature_leakage" and f.severity == "info"]
        assert len(infos) == 1
