import numpy as np
import pandas as pd
import pytest

from splitsmith import audit_cv, k_fold


def _time_df(n=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


def _group_time_df(n_groups=20, rows_per_group=5):
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({
                "group_id": f"g{i}",
                "timestamp": base + pd.Timedelta(days=i * rows_per_group + j),
                "feature": i * rows_per_group + j,
                "target": j % 2,
            })
    return pd.DataFrame(rows)


class TestTimeKFold:
    def test_forward_chaining_order(self):
        df = _time_df(60)
        cv = k_fold(df, "target", k=5, strategy="time", time_col="timestamp")
        for fold in cv.folds:
            train_max = df.iloc[fold.train_idx]["timestamp"].max()
            val_min = df.iloc[fold.val_idx]["timestamp"].min()
            assert train_max <= val_min

    def test_train_grows(self):
        df = _time_df(60)
        cv = k_fold(df, "target", k=5, strategy="time", time_col="timestamp")
        sizes = [len(f.train_idx) for f in cv.folds]
        assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))

    def test_correct_k(self):
        assert k_fold(_time_df(60), "target", k=5, strategy="time", time_col="timestamp").k == 5

    def test_all_data_used(self):
        df = _time_df(60)
        cv = k_fold(df, "target", k=5, strategy="time", time_col="timestamp")
        used = set()
        for fold in cv.folds:
            used.update(fold.train_idx.tolist())
            used.update(fold.val_idx.tolist())
        assert used == set(range(len(df)))

    def test_requires_time_col(self):
        with pytest.raises(ValueError):
            k_fold(_time_df(60), "target", k=5, strategy="time")

    def test_metadata_time_ranges(self):
        cv = k_fold(_time_df(60), "target", k=5, strategy="time", time_col="timestamp")
        for fold in cv.folds:
            assert "train_time_range" in fold.metadata
            assert "val_time_range" in fold.metadata

    def test_audit_cv_integration(self):
        df = _time_df(60)
        cv = k_fold(df, "target", k=5, strategy="time", time_col="timestamp")
        reports = audit_cv(df, cv, "target", time_col="timestamp")
        assert len(reports) == 5
        assert all(r.ok for r in reports)


class TestGroupTimeKFold:
    def test_no_group_overlap(self):
        df = _group_time_df()
        cv = k_fold(df, "target", k=5, strategy="group_time", groups="group_id", time_col="timestamp")
        for fold in cv.folds:
            train_g = set(df.iloc[fold.train_idx]["group_id"].unique())
            val_g = set(df.iloc[fold.val_idx]["group_id"].unique())
            assert len(train_g & val_g) == 0

    def test_forward_chaining_group_order(self):
        df = _group_time_df()
        group_max = df.groupby("group_id")["timestamp"].max()
        cv = k_fold(df, "target", k=5, strategy="group_time", groups="group_id", time_col="timestamp")
        for fold in cv.folds:
            train_groups = df.iloc[fold.train_idx]["group_id"].unique()
            val_groups = df.iloc[fold.val_idx]["group_id"].unique()
            assert group_max.loc[train_groups].max() <= group_max.loc[val_groups].min()

    def test_correct_k(self):
        cv = k_fold(_group_time_df(), "target", k=5, strategy="group_time", groups="group_id", time_col="timestamp")
        assert cv.k == 5

    def test_requires_both_params(self):
        df = _group_time_df()
        with pytest.raises(ValueError):
            k_fold(df, "target", k=5, strategy="group_time", groups="group_id")
        with pytest.raises(ValueError):
            k_fold(df, "target", k=5, strategy="group_time", time_col="timestamp")

    def test_deterministic(self):
        df = _group_time_df()
        a = k_fold(df, "target", k=5, strategy="group_time", groups="group_id", time_col="timestamp")
        b = k_fold(df, "target", k=5, strategy="group_time", groups="group_id", time_col="timestamp")
        for fa, fb in zip(a.folds, b.folds):
            assert np.array_equal(fa.train_idx, fb.train_idx)

    def test_audit_cv_integration(self):
        df = _group_time_df()
        cv = k_fold(df, "target", k=5, strategy="group_time", groups="group_id", time_col="timestamp")
        reports = audit_cv(df, cv, "target", groups="group_id", time_col="timestamp")
        assert len(reports) == 5
        assert all(r.ok for r in reports)
