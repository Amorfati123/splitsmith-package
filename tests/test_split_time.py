import numpy as np
import pandas as pd
import pytest

from splitsmith import split, audit


class TestTimeSplit:
    def test_chronological_order(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        r = split(df, target="target", strategy="time", time_col="timestamp")
        train_max = df.iloc[r.train_idx]["timestamp"].max()
        val_min = df.iloc[r.val_idx]["timestamp"].min()
        val_max = df.iloc[r.val_idx]["timestamp"].max()
        test_min = df.iloc[r.test_idx]["timestamp"].min()
        assert train_max <= val_min
        assert val_max <= test_min

    def test_all_rows_covered(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        r = split(df, target="target", strategy="time", time_col="timestamp")
        all_idx = set(r.train_idx.tolist()) | set(r.val_idx.tolist()) | set(r.test_idx.tolist())
        assert all_idx == set(range(len(df)))

    def test_deterministic(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        r1 = split(df, target="target", strategy="time", time_col="timestamp")
        r2 = split(df, target="target", strategy="time", time_col="timestamp")
        assert np.array_equal(r1.train_idx, r2.train_idx)

    def test_requires_time_col(self):
        df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
        with pytest.raises(ValueError, match="time_col parameter"):
            split(df, target="target", strategy="time")

    def test_invalid_time_col(self):
        df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
        with pytest.raises(ValueError, match="time_col column"):
            split(df, target="target", strategy="time", time_col="nonexistent")

    def test_metadata(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        r = split(df, target="target", strategy="time", time_col="timestamp")
        assert r.metadata["strategy"] == "time"
        assert "time_range" in r.metadata

    def test_stratify_not_supported(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        with pytest.raises(NotImplementedError):
            split(df, target="target", strategy="time", time_col="timestamp", stratify=True)

    def test_audit_integration(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
            "feature": range(100),
            "target": [0, 1] * 50,
        })
        r = split(df, target="target", strategy="time", time_col="timestamp")
        report = audit(df, r, "target", time_col="timestamp")
        assert report.ok is True
