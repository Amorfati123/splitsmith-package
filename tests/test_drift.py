"""Tests for cross-split drift diagnostics."""

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


class TestLabelDrift:
    def test_balanced_split_reports_prevalence(self):
        df = pd.DataFrame({
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target", check_drift=True)
        label_findings = [f for f in report.findings if f.id == "label_drift"]
        assert len(label_findings) >= 1
        # should have prevalence evidence
        assert any("prevalence" in f.evidence for f in label_findings)

    def test_imbalanced_split_warns(self):
        """When train is all 0s and test is all 1s, should warn."""
        df = pd.DataFrame({
            "feature": range(20),
            "target": [0] * 10 + [1] * 10,
        })
        sr = _make_result(list(range(10)), list(range(10, 15)), list(range(15, 20)))
        report = audit(df, sr, "target", check_drift=True, drift_threshold=0.1)
        warns = [f for f in report.findings if f.id == "label_drift" and f.severity == "warn"]
        assert len(warns) >= 1

    def test_no_drift_flag_when_balanced(self):
        rng = np.random.default_rng(42)
        targets = rng.choice([0, 1], size=200, p=[0.5, 0.5])
        df = pd.DataFrame({"feature": range(200), "target": targets})
        sr = _make_result(list(range(140)), list(range(140, 170)), list(range(170, 200)))
        report = audit(df, sr, "target", check_drift=True, drift_threshold=0.3)
        warns = [f for f in report.findings if f.id == "label_drift" and f.severity == "warn"]
        assert len(warns) == 0


class TestFeatureDrift:
    def test_psi_computed_for_numeric_columns(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.normal(0, 1, 100),
            "f2": rng.normal(0, 1, 100),
            "target": [0, 1] * 50,
        })
        sr = _make_result(list(range(70)), list(range(70, 85)), list(range(85, 100)))
        report = audit(df, sr, "target", check_drift=True)
        psi_findings = [f for f in report.findings if f.id == "feature_drift"]
        assert len(psi_findings) >= 1
        assert any("psi" in f.evidence for f in psi_findings)

    def test_high_psi_warns(self):
        """Train and test from very different distributions should flag."""
        df = pd.DataFrame({
            "f1": list(np.zeros(50)) + list(np.ones(50)),
            "target": [0, 1] * 50,
        })
        sr = _make_result(list(range(50)), list(range(50, 75)), list(range(75, 100)))
        report = audit(df, sr, "target", check_drift=True)
        warns = [f for f in report.findings if f.id == "feature_drift" and f.severity == "warn"]
        assert len(warns) >= 1

    def test_custom_drift_columns(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.normal(0, 1, 50),
            "f2": rng.normal(0, 1, 50),
            "target": [0, 1] * 25,
        })
        sr = _make_result(list(range(35)), list(range(35, 42)), list(range(42, 50)))
        report = audit(df, sr, "target", check_drift=True, drift_columns=["f1"])
        psi_findings = [f for f in report.findings if f.id == "feature_drift"]
        if psi_findings:
            psi_data = psi_findings[0].evidence.get("psi", {})
            assert "f1" in psi_data
            assert "f2" not in psi_data

    def test_invalid_drift_column_raises(self):
        df = pd.DataFrame({"f1": range(10), "target": [0, 1] * 5})
        sr = _make_result(list(range(7)), list(range(7, 8)), list(range(8, 10)))
        with pytest.raises(ValueError, match="drift_columns"):
            audit(df, sr, "target", check_drift=True, drift_columns=["bad_col"])

    def test_drift_disabled_by_default(self):
        df = pd.DataFrame({"f1": range(10), "target": [0, 1] * 5})
        sr = _make_result(list(range(7)), list(range(7, 8)), list(range(8, 10)))
        report = audit(df, sr, "target")
        drift_findings = [f for f in report.findings
                          if f.id in ("label_drift", "feature_drift")]
        assert len(drift_findings) == 0
