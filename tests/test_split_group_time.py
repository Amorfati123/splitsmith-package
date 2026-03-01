import numpy as np
import pandas as pd
import pytest

from splitsmith import split, audit


def _group_time_df(n_groups=10, rows_per_group=5):
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


class TestGroupTimeSplit:
    def test_no_group_overlap(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        train_g = set(df.iloc[r.train_idx]["group_id"].unique())
        val_g = set(df.iloc[r.val_idx]["group_id"].unique())
        test_g = set(df.iloc[r.test_idx]["group_id"].unique())
        assert train_g.isdisjoint(val_g)
        assert train_g.isdisjoint(test_g)
        assert val_g.isdisjoint(test_g)

    def test_chronological_group_order(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        group_max = df.groupby("group_id")["timestamp"].max()
        train_groups = df.iloc[r.train_idx]["group_id"].unique()
        val_groups = df.iloc[r.val_idx]["group_id"].unique()
        test_groups = df.iloc[r.test_idx]["group_id"].unique()
        assert group_max.loc[train_groups].max() <= group_max.loc[val_groups].min()
        assert group_max.loc[val_groups].max() <= group_max.loc[test_groups].min()

    def test_all_rows_covered(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        all_idx = set(r.train_idx.tolist()) | set(r.val_idx.tolist()) | set(r.test_idx.tolist())
        assert all_idx == set(range(len(df)))

    def test_requires_both_params(self):
        df = _group_time_df()
        with pytest.raises(ValueError, match="Both groups and time_col"):
            split(df, target="target", strategy="group_time", groups="group_id")
        with pytest.raises(ValueError, match="Both groups and time_col"):
            split(df, target="target", strategy="group_time", time_col="timestamp")

    def test_deterministic(self):
        df = _group_time_df()
        r1 = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        r2 = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        assert np.array_equal(r1.train_idx, r2.train_idx)

    def test_metadata(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        assert r.metadata["strategy"] == "group_time"
        assert r.metadata["n_groups"] == 10
        assert "time_range" in r.metadata
        assert "groups_per_split" in r.metadata

    def test_stratify_not_supported(self):
        df = _group_time_df()
        with pytest.raises(NotImplementedError):
            split(df, target="target", strategy="group_time",
                  groups="group_id", time_col="timestamp", stratify=True)

    def test_audit_integration(self):
        df = _group_time_df()
        r = split(df, target="target", strategy="group_time", groups="group_id", time_col="timestamp")
        report = audit(df, r, "target", groups="group_id", time_col="timestamp")
        assert report.ok is True
