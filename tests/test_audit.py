"""Tests for splitsmith.audit()."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import SplitResult, audit, split


def _make_result(train, val, test):
    """Helper to build a SplitResult from plain lists."""
    return SplitResult(
        train_idx=np.array(train, dtype=int),
        val_idx=np.array(val, dtype=int),
        test_idx=np.array(test, dtype=int),
    )


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

class TestAuditValidation:
    def test_bad_df_type(self):
        sr = _make_result([0], [1], [2])
        with pytest.raises(TypeError, match="pandas DataFrame"):
            audit("not a df", sr, "target")

    def test_bad_split_result_type(self):
        df = pd.DataFrame({"target": [0, 1, 2]})
        with pytest.raises(TypeError, match="SplitResult"):
            audit(df, "bad", "target")

    def test_missing_target(self):
        df = pd.DataFrame({"feature": [0, 1, 2]})
        sr = _make_result([0], [1], [2])
        with pytest.raises(ValueError, match="target column"):
            audit(df, sr, "nonexistent")

    def test_missing_groups_col(self):
        df = pd.DataFrame({"target": [0, 1, 2]})
        sr = _make_result([0], [1], [2])
        with pytest.raises(ValueError, match="groups column"):
            audit(df, sr, "target", groups="nonexistent")

    def test_missing_time_col(self):
        df = pd.DataFrame({"target": [0, 1, 2]})
        sr = _make_result([0], [1], [2])
        with pytest.raises(ValueError, match="time_col"):
            audit(df, sr, "target", time_col="nonexistent")


# ---------------------------------------------------------------------------
# 2. Overlap check
# ---------------------------------------------------------------------------

class TestOverlapCheck:
    def test_clean_split_no_overlap(self):
        df = pd.DataFrame({"target": range(7)})
        sr = _make_result([0, 1, 2], [3, 4], [5, 6])
        report = audit(df, sr, "target")
        overlap = [f for f in report.findings if f.id == "index_overlap"]
        assert len(overlap) == 1
        assert overlap[0].severity == "info"

    def test_overlapping_indices_detected(self):
        df = pd.DataFrame({"target": range(7)})
        sr = _make_result([0, 1, 2], [2, 3, 4], [5, 6])
        report = audit(df, sr, "target")
        errors = [f for f in report.findings if f.id == "index_overlap" and f.severity == "error"]
        assert len(errors) == 1
        assert errors[0].evidence["count"] == 1
        assert 2 in errors[0].evidence["examples"]

    def test_multiple_overlap_pairs(self):
        df = pd.DataFrame({"target": range(7)})
        sr = _make_result([0, 1, 2], [2, 3, 4], [4, 5, 6])
        report = audit(df, sr, "target")
        errors = [f for f in report.findings if f.id == "index_overlap" and f.severity == "error"]
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# 3. Duplicate rows check
# ---------------------------------------------------------------------------

class TestDuplicateCheck:
    def test_no_duplicates(self):
        df = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "target": [0, 1, 0, 1, 0, 1]})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        dup = [f for f in report.findings if f.id == "duplicate_rows"]
        assert len(dup) == 1
        assert dup[0].severity == "info"

    def test_cross_split_duplicates(self):
        # Row 0 and row 2 are identical (feature=1, target=0), in different splits
        df = pd.DataFrame({"feature": [1, 2, 1, 3, 4, 5], "target": [0, 1, 0, 1, 0, 1]})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target")
        errors = [f for f in report.findings if f.id == "duplicate_rows" and f.severity == "error"]
        assert len(errors) == 1
        assert "Cross-split" in errors[0].title

    def test_within_split_duplicates(self):
        # Row 0 and row 1 are identical, both in train
        df = pd.DataFrame({"feature": [1, 1, 2, 3, 4, 5], "target": [0, 0, 1, 0, 1, 0]})
        sr = _make_result([0, 1, 2], [3, 4], [5])
        report = audit(df, sr, "target")
        warns = [f for f in report.findings if f.id == "duplicate_rows" and f.severity == "warn"]
        assert len(warns) == 1


# ---------------------------------------------------------------------------
# 4. Group leakage check
# ---------------------------------------------------------------------------

class TestGroupLeakage:
    def test_clean_groups(self):
        df = pd.DataFrame({"user": ["A", "A", "B", "B", "C", "C"], "target": [0, 1] * 3})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups="user")
        grp = [f for f in report.findings if f.id == "group_leakage"]
        assert len(grp) == 1
        assert grp[0].severity == "info"

    def test_group_leakage_detected(self):
        # User A in train (idx 0,1) AND val (idx 3)
        df = pd.DataFrame({"user": ["A", "A", "B", "A", "C", "C"], "target": [0, 1] * 3})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups="user")
        errors = [f for f in report.findings if f.id == "group_leakage" and f.severity == "error"]
        assert len(errors) >= 1
        all_examples = []
        for e in errors:
            all_examples.extend(e.evidence["examples"])
        assert "A" in all_examples

    def test_group_leakage_multiple_pairs(self):
        # A in train+val, B in val+test
        df = pd.DataFrame({"user": ["A", "A", "B", "A", "C", "B"], "target": [0, 1] * 3})
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", groups="user")
        errors = [f for f in report.findings if f.id == "group_leakage" and f.severity == "error"]
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# 5. Time leakage check
# ---------------------------------------------------------------------------

class TestTimeLeakage:
    def test_chronological_split(self):
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=6, freq="D"),
            "target": [0, 1] * 3,
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", time_col="ts")
        time_f = [f for f in report.findings if f.id == "time_leakage"]
        assert len(time_f) == 1
        assert time_f[0].severity == "info"

    def test_train_after_val_error(self):
        # idx 1 has Jan 4 (in train), but val starts at Jan 2 (idx 2)
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-02",
                                   "2024-01-03", "2024-01-05", "2024-01-06"]),
            "target": [0, 1] * 3,
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", time_col="ts")
        errors = [f for f in report.findings if f.id == "time_leakage" and f.severity == "error"]
        assert len(errors) >= 1

    def test_only_val_test_warn(self):
        # Train is clean, but val has Jan 5 which is after test's Jan 4
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                                   "2024-01-05", "2024-01-04", "2024-01-06"]),
            "target": [0, 1] * 3,
        })
        sr = _make_result([0, 1], [2, 3], [4, 5])
        report = audit(df, sr, "target", time_col="ts")
        errors = [f for f in report.findings if f.id == "time_leakage" and f.severity == "error"]
        warns = [f for f in report.findings if f.id == "time_leakage" and f.severity == "warn"]
        assert len(errors) == 0
        assert len(warns) == 1


# ---------------------------------------------------------------------------
# 6. Integration with split()
# ---------------------------------------------------------------------------

class TestAuditIntegration:
    def test_clean_split_ok(self):
        df = pd.DataFrame({
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        sr = split(df, target="target", seed=42)
        report = audit(df, sr, "target")
        assert report.ok is True
        assert report.summary()["error"] == 0

    def test_repr_contains_0_error(self):
        df = pd.DataFrame({
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        sr = split(df, target="target", seed=42)
        report = audit(df, sr, "target")
        assert "0 error" in repr(report)
