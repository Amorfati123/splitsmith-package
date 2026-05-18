"""Tests for near-duplicate detection in audit."""

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


class TestNearDuplicates:
    def test_exact_copies_found(self):
        """Rows identical on checked columns should be caught even at tolerance=0."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 1.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 10.0, 30.0, 40.0, 50.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(
            df, sr, "target",
            near_duplicate_columns=["f1", "f2"],
            near_duplicate_tolerance=0.0,
        )
        findings = [f for f in report.findings if f.id == "near_duplicates"]
        warns = [f for f in findings if f.severity == "warn"]
        assert len(warns) == 1
        assert warns[0].evidence["count"] >= 1

    def test_within_tolerance_found(self):
        """Rows within tolerance should be flagged."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 1.05, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(
            df, sr, "target",
            near_duplicate_columns=["f1"],
            near_duplicate_tolerance=0.1,
        )
        warns = [f for f in report.findings if f.id == "near_duplicates" and f.severity == "warn"]
        assert len(warns) == 1

    def test_beyond_tolerance_clean(self):
        """Rows far apart should not be flagged."""
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(
            df, sr, "target",
            near_duplicate_columns=["f1"],
            near_duplicate_tolerance=0.01,
        )
        infos = [f for f in report.findings if f.id == "near_duplicates" and f.severity == "info"]
        assert len(infos) == 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "target": [0, 1] * 3})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        with pytest.raises(ValueError, match="near_duplicate_columns"):
            audit(df, sr, "target",
                  near_duplicate_columns=["nonexistent"],
                  near_duplicate_tolerance=0.1)

    def test_non_numeric_columns_skipped(self):
        """Non-numeric near_duplicate_columns are skipped gracefully."""
        df = pd.DataFrame({
            "name": ["a", "b", "c", "d", "e", "f"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(
            df, sr, "target",
            near_duplicate_columns=["name"],
            near_duplicate_tolerance=0.1,
        )
        findings = [f for f in report.findings if f.id == "near_duplicates"]
        assert len(findings) == 1
        assert findings[0].severity == "info"

    def test_not_triggered_when_columns_none(self):
        """Near-duplicate check only runs when columns are specified."""
        df = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6], "target": [0, 1] * 3})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        findings = [f for f in report.findings if f.id == "near_duplicates"]
        assert len(findings) == 0

    def test_evidence_has_distance(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 1.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0, 1],
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(
            df, sr, "target",
            near_duplicate_columns=["f1"],
            near_duplicate_tolerance=0.0,
        )
        warns = [f for f in report.findings if f.id == "near_duplicates" and f.severity == "warn"]
        if warns:
            examples = warns[0].evidence.get("examples", [])
            if examples and "distance" in examples[0]:
                assert isinstance(examples[0]["distance"], float)
