"""Tests for sklearn-compatible splitter classes."""

import numpy as np
import pandas as pd
import pytest

from splitsmith.compat import (
    SplitsmithKFold,
    SplitsmithStratifiedKFold,
    SplitsmithGroupKFold,
    SplitsmithTimeSeriesSplit,
    SplitsmithGroupTimeSeriesSplit,
)


def _simple_df(n=100):
    return pd.DataFrame({
        "f1": range(n),
        "f2": np.random.default_rng(0).normal(size=n),
        "target": [0, 1] * (n // 2),
    })


def _time_df(n=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


def _group_time_df(n_groups=15, rows_per=5):
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n_groups):
        for j in range(rows_per):
            rows.append({
                "group_id": f"g{i}",
                "timestamp": base + pd.Timedelta(days=i * rows_per + j),
                "feature": i * rows_per + j,
                "target": j % 2,
            })
    return pd.DataFrame(rows)


class TestSplitsmithKFold:
    def test_yields_correct_number_of_folds(self):
        X = _simple_df()
        splitter = SplitsmithKFold(n_splits=5, seed=42)
        folds = list(splitter.split(X))
        assert len(folds) == 5

    def test_get_n_splits(self):
        splitter = SplitsmithKFold(n_splits=3)
        assert splitter.get_n_splits() == 3

    def test_no_overlap_between_train_val(self):
        X = _simple_df()
        splitter = SplitsmithKFold(n_splits=5)
        for train_idx, val_idx in splitter.split(X):
            assert set(train_idx).isdisjoint(set(val_idx))

    def test_all_indices_covered(self):
        X = _simple_df(50)
        splitter = SplitsmithKFold(n_splits=5)
        all_val = set()
        for _, val_idx in splitter.split(X):
            all_val.update(val_idx)
        assert all_val == set(range(50))

    def test_works_with_numpy_array(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        splitter = SplitsmithKFold(n_splits=3)
        folds = list(splitter.split(X))
        assert len(folds) == 3


class TestSplitsmithStratifiedKFold:
    def test_yields_correct_folds(self):
        X = _simple_df()
        y = X["target"].values
        splitter = SplitsmithStratifiedKFold(n_splits=5)
        folds = list(splitter.split(X, y=y))
        assert len(folds) == 5

    def test_requires_y(self):
        X = _simple_df()
        splitter = SplitsmithStratifiedKFold(n_splits=5)
        with pytest.raises(ValueError, match="y is required"):
            list(splitter.split(X))

    def test_preserves_class_balance(self):
        df = _simple_df(100)
        y = df["target"].values
        overall_pct = np.mean(y)
        splitter = SplitsmithStratifiedKFold(n_splits=5)
        for train_idx, val_idx in splitter.split(df, y=y):
            val_pct = np.mean(y[val_idx])
            assert abs(val_pct - overall_pct) < 0.15


class TestSplitsmithGroupKFold:
    def test_yields_correct_folds(self):
        df = _group_time_df()
        groups = df["group_id"].values
        splitter = SplitsmithGroupKFold(n_splits=5)
        folds = list(splitter.split(df, groups=groups))
        assert len(folds) == 5

    def test_requires_groups(self):
        df = _simple_df()
        splitter = SplitsmithGroupKFold(n_splits=5)
        with pytest.raises(ValueError, match="groups is required"):
            list(splitter.split(df))

    def test_groups_are_exclusive(self):
        df = _group_time_df()
        groups = df["group_id"].values
        splitter = SplitsmithGroupKFold(n_splits=5)
        for train_idx, val_idx in splitter.split(df, groups=groups):
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            assert train_groups.isdisjoint(val_groups)


class TestSplitsmithTimeSeriesSplit:
    def test_yields_correct_folds(self):
        df = _time_df()
        splitter = SplitsmithTimeSeriesSplit(n_splits=4, time_col="timestamp")
        folds = list(splitter.split(df))
        assert len(folds) == 4

    def test_train_grows_each_fold(self):
        df = _time_df()
        splitter = SplitsmithTimeSeriesSplit(n_splits=4, time_col="timestamp")
        prev_train_size = 0
        for train_idx, val_idx in splitter.split(df):
            assert len(train_idx) >= prev_train_size
            prev_train_size = len(train_idx)

    def test_missing_time_col_raises(self):
        df = _simple_df()
        splitter = SplitsmithTimeSeriesSplit(time_col="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            list(splitter.split(df))

    def test_gap_parameter(self):
        df = _time_df()
        splitter = SplitsmithTimeSeriesSplit(n_splits=3, time_col="timestamp", gap=5)
        for train_idx, val_idx in splitter.split(df):
            train_max = df.iloc[train_idx]["timestamp"].max()
            val_min = df.iloc[val_idx]["timestamp"].min()
            assert train_max < val_min


class TestSplitsmithGroupTimeSeriesSplit:
    def test_yields_correct_folds(self):
        df = _group_time_df()
        splitter = SplitsmithGroupTimeSeriesSplit(
            n_splits=4, time_col="timestamp", groups_col="group_id"
        )
        folds = list(splitter.split(df))
        assert len(folds) == 4

    def test_groups_exclusive_across_folds(self):
        df = _group_time_df()
        splitter = SplitsmithGroupTimeSeriesSplit(
            n_splits=3, time_col="timestamp", groups_col="group_id"
        )
        for train_idx, val_idx in splitter.split(df):
            train_g = set(df.iloc[train_idx]["group_id"])
            val_g = set(df.iloc[val_idx]["group_id"])
            assert train_g.isdisjoint(val_g)

    def test_accepts_groups_argument(self):
        df = _group_time_df()
        groups = df["group_id"].values
        splitter = SplitsmithGroupTimeSeriesSplit(
            n_splits=3, time_col="timestamp", groups_col="nonexistent"
        )
        # should work because groups= argument is passed
        folds = list(splitter.split(df, groups=groups))
        assert len(folds) == 3

    def test_missing_both_raises(self):
        df = _simple_df()
        splitter = SplitsmithGroupTimeSeriesSplit(
            n_splits=3, time_col="timestamp", groups_col="nonexistent"
        )
        with pytest.raises(ValueError):
            list(splitter.split(df))
