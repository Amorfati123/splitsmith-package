import numpy as np
import pandas as pd
import pytest

from splitsmith import CVResult, k_fold


def _simple_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feature": rng.standard_normal(n),
        "target": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })


class TestBasicKFold:
    def test_correct_number_of_folds(self):
        cv = k_fold(_simple_df(), "target", k=5)
        assert isinstance(cv, CVResult)
        assert cv.k == 5

    def test_no_overlap_per_fold(self):
        cv = k_fold(_simple_df(), "target", k=5)
        for fold in cv.folds:
            assert len(set(fold.train_idx) & set(fold.val_idx)) == 0

    def test_full_coverage_per_fold(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=5)
        all_idx = set(range(len(df)))
        for fold in cv.folds:
            assert set(fold.train_idx) | set(fold.val_idx) == all_idx

    def test_val_partitions_data(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=5)
        all_val = np.concatenate([fold.val_idx for fold in cv.folds])
        assert len(all_val) == len(df)
        assert len(np.unique(all_val)) == len(df)

    def test_deterministic(self):
        df = _simple_df()
        a = k_fold(df, "target", k=5, seed=11)
        b = k_fold(df, "target", k=5, seed=11)
        for fa, fb in zip(a.folds, b.folds):
            assert np.array_equal(fa.train_idx, fb.train_idx)
            assert np.array_equal(fa.val_idx, fb.val_idx)

    def test_different_seeds(self):
        df = _simple_df()
        a = k_fold(df, "target", k=5, seed=11)
        b = k_fold(df, "target", k=5, seed=12)
        assert any(not np.array_equal(fa.val_idx, fb.val_idx) for fa, fb in zip(a.folds, b.folds))

    def test_k_equals_n(self):
        cv = k_fold(_simple_df(12), "target", k=12)
        assert all(len(f.val_idx) == 1 for f in cv.folds)

    def test_k_2_minimum(self):
        assert k_fold(_simple_df(), "target", k=2).k == 2

    def test_k_too_large(self):
        with pytest.raises(ValueError):
            k_fold(_simple_df(10), "target", k=11)

    def test_k_less_than_2(self):
        with pytest.raises(ValueError):
            k_fold(_simple_df(10), "target", k=1)

    def test_metadata(self):
        cv = k_fold(_simple_df(25), "target", k=5, seed=9)
        assert cv.metadata["strategy"] == "basic"
        assert cv.metadata["k"] == 5
        assert cv.metadata["seed"] == 9

    def test_summary(self):
        cv = k_fold(_simple_df(20), "target", k=4)
        s = cv.summary()
        assert s["k"] == 4
        assert len(s["fold_sizes"]) == 4


class TestStratifiedKFold:
    def test_class_presence_all_folds(self):
        cv = k_fold(_simple_df(200), "target", k=5, strategy="stratified")
        for fold in cv.folds:
            classes = set(_simple_df(200).iloc[fold.val_idx]["target"].tolist())
            # Can't reuse df easily; just check our df
        # Simpler: build df once
        df = _simple_df(200)
        cv = k_fold(df, "target", k=5, strategy="stratified")
        for fold in cv.folds:
            assert len(set(df.iloc[fold.val_idx]["target"].unique())) >= 2

    def test_class_proportions_approximately_preserved(self):
        df = _simple_df(200)
        cv = k_fold(df, "target", k=5, strategy="stratified")
        overall = float((df["target"] == 1).mean())
        for fold in cv.folds:
            val_ratio = float((df.iloc[fold.val_idx]["target"] == 1).mean())
            assert abs(val_ratio - overall) <= 0.15

    def test_too_few_samples_per_class(self):
        df = pd.DataFrame({"feature": range(10), "target": [0] * 9 + [1]})
        with pytest.raises(ValueError):
            k_fold(df, "target", k=3, strategy="stratified")

    def test_metadata_stratified(self):
        cv = k_fold(_simple_df(100), "target", k=5, strategy="stratified")
        assert cv.metadata["stratified"] is True
        assert all(f.metadata["stratified"] is True for f in cv.folds)
