"""Tests for reproducibility metadata in split and CV results."""

import numpy as np
import pandas as pd

from splitsmith import split, k_fold
from splitsmith._version import __version__


def _simple_df(n=50):
    return pd.DataFrame({
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


class TestSplitReproducibility:
    def test_metadata_has_reproducibility_block(self):
        df = _simple_df()
        r = split(df, target="target")
        assert "reproducibility" in r.metadata

    def test_has_version_info(self):
        df = _simple_df()
        r = split(df, target="target")
        repro = r.metadata["reproducibility"]
        assert repro["splitsmith_version"] == __version__
        assert "numpy_version" in repro
        assert "pandas_version" in repro

    def test_has_data_fingerprints(self):
        df = _simple_df()
        r = split(df, target="target")
        repro = r.metadata["reproducibility"]
        assert "data_hash" in repro
        assert "schema_hash" in repro
        assert len(repro["data_hash"]) == 16
        assert len(repro["schema_hash"]) == 16

    def test_has_timestamp(self):
        df = _simple_df()
        r = split(df, target="target")
        repro = r.metadata["reproducibility"]
        assert "timestamp" in repro
        assert "T" in repro["timestamp"]  # ISO format

    def test_has_params(self):
        df = _simple_df()
        r = split(df, target="target", seed=99, ratios=(0.6, 0.2, 0.2))
        params = r.metadata["reproducibility"]["params"]
        assert params["seed"] == 99
        assert params["ratios"] == (0.6, 0.2, 0.2)
        assert params["strategy"] == "random"
        assert params["target"] == "target"

    def test_same_data_same_hash(self):
        df = _simple_df()
        r1 = split(df, target="target", seed=1)
        r2 = split(df, target="target", seed=2)
        assert r1.metadata["reproducibility"]["data_hash"] == r2.metadata["reproducibility"]["data_hash"]

    def test_different_data_different_hash(self):
        df1 = _simple_df(50)
        df2 = _simple_df(60)
        r1 = split(df1, target="target")
        r2 = split(df2, target="target")
        assert r1.metadata["reproducibility"]["data_hash"] != r2.metadata["reproducibility"]["data_hash"]

    def test_has_row_and_col_counts(self):
        df = _simple_df()
        r = split(df, target="target")
        repro = r.metadata["reproducibility"]
        assert repro["n_rows"] == len(df)
        assert repro["n_columns"] == len(df.columns)


class TestCVReproducibility:
    def test_cv_metadata_has_reproducibility(self):
        df = _simple_df()
        cv = k_fold(df, target="target", k=3)
        assert "reproducibility" in cv.metadata

    def test_cv_has_version_info(self):
        df = _simple_df()
        cv = k_fold(df, target="target", k=3)
        repro = cv.metadata["reproducibility"]
        assert repro["splitsmith_version"] == __version__

    def test_cv_has_params(self):
        df = _simple_df()
        cv = k_fold(df, target="target", k=4, strategy="stratified", seed=77)
        params = cv.metadata["reproducibility"]["params"]
        assert params["k"] == 4
        assert params["strategy"] == "stratified"
        assert params["seed"] == 77

    def test_cv_has_data_hash(self):
        df = _simple_df()
        cv = k_fold(df, target="target", k=3)
        repro = cv.metadata["reproducibility"]
        assert "data_hash" in repro
        assert "schema_hash" in repro
