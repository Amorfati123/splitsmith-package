"""Tests for repeated_split and repeated_k_fold."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import split, repeated_split, k_fold, repeated_k_fold


def _simple_df(n=200):
    return pd.DataFrame({
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


def _grouped_df(n_groups=12, rows_per_group=10):
    rows = []
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({"group_id": f"g{i}", "feature": i * 10 + j, "target": j % 2})
    return pd.DataFrame(rows)


class TestRepeatedSplit:
    def test_returns_correct_count(self):
        df = _simple_df()
        out = repeated_split(df, target="target", n_repeats=5)
        assert len(out["results"]) == 5
        assert out["summary"]["n_repeats"] == 5

    def test_each_result_is_valid(self):
        df = _simple_df()
        out = repeated_split(df, target="target", n_repeats=3)
        for r in out["results"]:
            all_idx = set(r.train_idx.tolist()) | set(r.val_idx.tolist()) | set(r.test_idx.tolist())
            assert all_idx == set(range(len(df)))

    def test_different_seeds_give_different_splits(self):
        df = _simple_df()
        out = repeated_split(df, target="target", n_repeats=3, seed=42)
        # at least two of the three should differ in train indices
        trains = [set(r.train_idx.tolist()) for r in out["results"]]
        assert trains[0] != trains[1] or trains[1] != trains[2]

    def test_summary_has_size_stats(self):
        df = _simple_df()
        out = repeated_split(df, target="target", n_repeats=5)
        s = out["summary"]
        assert "train_size" in s
        assert "mean" in s["train_size"]
        assert "std" in s["train_size"]
        assert "val_size" in s
        assert "test_size" in s

    def test_summary_has_class_balance(self):
        df = _simple_df()
        out = repeated_split(df, target="target", n_repeats=5)
        s = out["summary"]
        assert "train_class_balance" in s
        assert "0" in s["train_class_balance"] or 0 in s["train_class_balance"]

    def test_works_with_group_strategy(self):
        df = _grouped_df()
        out = repeated_split(df, target="target", strategy="group",
                             groups="group_id", n_repeats=3)
        assert len(out["results"]) == 3
        for r in out["results"]:
            train_g = set(df.iloc[r.train_idx]["group_id"])
            val_g = set(df.iloc[r.val_idx]["group_id"])
            assert train_g.isdisjoint(val_g)

    def test_invalid_n_repeats(self):
        df = _simple_df()
        with pytest.raises(ValueError, match="positive integer"):
            repeated_split(df, target="target", n_repeats=0)

    def test_min_samples_propagated(self):
        df = _simple_df(30)
        # with a very high threshold this should fail
        with pytest.raises(ValueError, match="min_samples_per_class"):
            repeated_split(df, target="target", n_repeats=2,
                           min_samples_per_class=100)

    def test_seed_range_in_summary(self):
        df = _simple_df()
        out = repeated_split(df, target="target", n_repeats=5, seed=10)
        assert out["summary"]["seed_range"] == [10, 14]


class TestRepeatedKFold:
    def test_returns_correct_count(self):
        df = _simple_df()
        out = repeated_k_fold(df, target="target", n_repeats=3, k=5)
        assert len(out["results"]) == 3
        assert out["summary"]["n_repeats"] == 3

    def test_each_cv_has_k_folds(self):
        df = _simple_df()
        out = repeated_k_fold(df, target="target", n_repeats=3, k=4)
        for cv in out["results"]:
            assert cv.k == 4

    def test_different_seeds_give_different_folds(self):
        df = _simple_df()
        out = repeated_k_fold(df, target="target", n_repeats=3, k=3, seed=42)
        fold0_trains = [set(cv.folds[0].train_idx.tolist()) for cv in out["results"]]
        assert fold0_trains[0] != fold0_trains[1]

    def test_summary_has_fold_stats(self):
        df = _simple_df()
        out = repeated_k_fold(df, target="target", n_repeats=3, k=5)
        s = out["summary"]
        assert "train_fold_size" in s
        assert "val_fold_size" in s
        assert "mean" in s["train_fold_size"]
        assert "std" in s["train_fold_size"]

    def test_invalid_n_repeats(self):
        df = _simple_df()
        with pytest.raises(ValueError, match="positive integer"):
            repeated_k_fold(df, target="target", n_repeats=-1)

    def test_works_with_stratified(self):
        df = _simple_df()
        out = repeated_k_fold(df, target="target", n_repeats=2,
                              k=3, strategy="stratified")
        assert len(out["results"]) == 2

    def test_seed_range_in_summary(self):
        df = _simple_df()
        out = repeated_k_fold(df, target="target", n_repeats=4, seed=7)
        assert out["summary"]["seed_range"] == [7, 10]
