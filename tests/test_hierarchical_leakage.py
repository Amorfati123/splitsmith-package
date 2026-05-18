"""Tests for hierarchical group leakage detection."""

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


class TestHierarchicalLeakage:
    def test_clean_hierarchy(self):
        """No leakage when both levels are properly separated."""
        df = pd.DataFrame({
            "patient": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "session": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups=["patient", "session"])
        findings = [f for f in report.findings if f.id == "hierarchical_leakage"]
        assert len(findings) == 1
        assert findings[0].severity == "info"

    def test_patient_leakage_detected(self):
        """Same patient in train and val should be flagged."""
        df = pd.DataFrame({
            "patient": ["P1", "P1", "P1", "P2", "P3", "P3"],
            "session": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups=["patient", "session"])
        errors = [f for f in report.findings
                  if f.id == "hierarchical_leakage" and f.severity == "error"]
        assert len(errors) >= 1
        levels = [e.evidence["level"] for e in errors]
        assert "patient" in levels

    def test_session_leakage_only(self):
        """If patients are clean but sessions leak, still detected."""
        df = pd.DataFrame({
            "patient": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "session": ["S1", "S2", "S1", "S4", "S5", "S6"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups=["patient", "session"])
        errors = [f for f in report.findings
                  if f.id == "hierarchical_leakage" and f.severity == "error"]
        levels = [e.evidence["level"] for e in errors]
        assert "session" in levels

    def test_single_group_string_uses_regular_check(self):
        """Single string should use the single-level group leakage check."""
        df = pd.DataFrame({
            "user": ["A", "A", "B", "B", "C", "C"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups="user")
        group_findings = [f for f in report.findings if f.id == "group_leakage"]
        hier_findings = [f for f in report.findings if f.id == "hierarchical_leakage"]
        assert len(group_findings) >= 1
        assert len(hier_findings) == 0

    def test_three_level_hierarchy(self):
        df = pd.DataFrame({
            "hospital": ["H1"] * 4 + ["H2"] * 4,
            "patient": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
            "recording": [f"R{i}" for i in range(8)],
            "target": [0, 1] * 4,
        })
        sr = _make_result([0, 1, 2, 3], [4, 5], [6, 7])
        report = audit(df, sr, "target",
                       groups=["hospital", "patient", "recording"])
        findings = [f for f in report.findings if f.id == "hierarchical_leakage"]
        assert len(findings) >= 1

    def test_invalid_column_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        sr = _make_result([0], [1], [2])
        with pytest.raises(ValueError, match="groups column"):
            audit(df, sr, "target", groups=["a", "nonexistent"])
