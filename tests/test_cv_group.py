import numpy as np
import pandas as pd
import pytest

from splitsmith import audit_cv, k_fold


def _grouped_df(n_groups=20, rows_per_group=5):
    rows = []
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({"group_id": f"g{i}", "feature": i * rows_per_group + j, "target": j % 2})
    return pd.DataFrame(rows)


class TestGroupKFold:
    def test_no_group_overlap(self):
        df = _grouped_df()
        cv = k_fold(df, "target", k=5, strategy="group", groups="group_id")
        for fold in cv.folds:
            train_g = set(df.iloc[fold.train_idx]["group_id"].unique())
            val_g = set(df.iloc[fold.val_idx]["group_id"].unique())
            assert len(train_g & val_g) == 0

    def test_correct_k(self):
        assert k_fold(_grouped_df(), "target", k=5, strategy="group", groups="group_id").k == 5

    def test_all_rows_covered(self):
        df = _grouped_df()
        cv = k_fold(df, "target", k=5, strategy="group", groups="group_id")
        for fold in cv.folds:
            assert set(fold.train_idx) | set(fold.val_idx) == set(range(len(df)))

    def test_groups_partition(self):
        df = _grouped_df()
        cv = k_fold(df, "target", k=5, strategy="group", groups="group_id")
        all_val_groups = []
        for fold in cv.folds:
            all_val_groups.extend(df.iloc[fold.val_idx]["group_id"].unique().tolist())
        assert len(all_val_groups) == len(set(all_val_groups))
        assert set(all_val_groups) == set(df["group_id"].unique())

    def test_requires_groups(self):
        with pytest.raises(ValueError):
            k_fold(_grouped_df(), "target", k=5, strategy="group")

    def test_fewer_groups_than_k(self):
        df = _grouped_df(n_groups=4, rows_per_group=2)
        with pytest.raises(ValueError):
            k_fold(df, "target", k=5, strategy="group", groups="group_id")

    def test_deterministic(self):
        df = _grouped_df()
        a = k_fold(df, "target", k=5, strategy="group", groups="group_id", seed=1)
        b = k_fold(df, "target", k=5, strategy="group", groups="group_id", seed=1)
        for fa, fb in zip(a.folds, b.folds):
            assert np.array_equal(fa.train_idx, fb.train_idx)

    def test_metadata(self):
        cv = k_fold(_grouped_df(), "target", k=5, strategy="group", groups="group_id")
        assert cv.metadata["n_groups"] == 20
        for fold in cv.folds:
            assert "n_groups_val" in fold.metadata

    def test_audit_cv_integration(self):
        df = _grouped_df()
        cv = k_fold(df, "target", k=5, strategy="group", groups="group_id")
        reports = audit_cv(df, cv, "target", groups="group_id")
        assert len(reports) == 5
        assert all(r.ok for r in reports)
