import numpy as np
import pandas as pd
import pytest

from splitsmith import split, audit


def _grouped_df(n_groups=10, rows_per_group=5):
    rows = []
    for i in range(n_groups):
        for j in range(rows_per_group):
            rows.append({
                "group_id": f"g{i}",
                "feature": i * rows_per_group + j,
                "target": j % 2,
            })
    return pd.DataFrame(rows)


class TestGroupSplit:
    def test_no_group_overlap(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id")
        train_g = set(df.iloc[r.train_idx]["group_id"].unique())
        val_g = set(df.iloc[r.val_idx]["group_id"].unique())
        test_g = set(df.iloc[r.test_idx]["group_id"].unique())
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)

    def test_all_rows_covered(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id")
        all_idx = set(r.train_idx.tolist()) | set(r.val_idx.tolist()) | set(r.test_idx.tolist())
        assert all_idx == set(range(len(df)))

    def test_deterministic(self):
        df = _grouped_df()
        r1 = split(df, target="target", strategy="group", groups="group_id", seed=7)
        r2 = split(df, target="target", strategy="group", groups="group_id", seed=7)
        assert np.array_equal(r1.train_idx, r2.train_idx)
        assert np.array_equal(r1.val_idx, r2.val_idx)
        assert np.array_equal(r1.test_idx, r2.test_idx)

    def test_requires_groups_param(self):
        df = _grouped_df()
        with pytest.raises(ValueError, match="groups parameter"):
            split(df, target="target", strategy="group")

    def test_invalid_groups_col(self):
        df = _grouped_df()
        with pytest.raises(ValueError, match="groups column"):
            split(df, target="target", strategy="group", groups="nonexistent")

    def test_min_3_groups(self):
        df = pd.DataFrame({"group_id": ["A", "A", "B", "B"], "target": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="at least 3"):
            split(df, target="target", strategy="group", groups="group_id")

    def test_metadata(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id")
        assert r.metadata["strategy"] == "group"
        assert r.metadata["n_groups"] == 10
        assert "groups_per_split" in r.metadata

    def test_stratify_not_supported(self):
        df = _grouped_df()
        with pytest.raises(NotImplementedError):
            split(df, target="target", strategy="group", groups="group_id", stratify=True)

    def test_audit_integration(self):
        df = _grouped_df()
        r = split(df, target="target", strategy="group", groups="group_id")
        report = audit(df, r, "target", groups="group_id")
        assert report.ok is True
