"""Scikit-learn compatible splitter classes.

Each class implements .split(X, y, groups) and .get_n_splits() so it can
be used directly with cross_val_score, GridSearchCV, and similar sklearn
tooling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cv import (
    _basic_folds,
    _stratified_folds,
    _group_folds,
    _time_folds,
    _group_time_folds,
)


def _ensure_dataframe(X, y=None, groups=None, target_name="target", groups_name="groups"):
    """Convert X, y, groups into a DataFrame suitable for splitsmith internals."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(np.asarray(X))

    if y is not None:
        y_arr = np.asarray(y)
        df[target_name] = y_arr

    if groups is not None:
        g_arr = np.asarray(groups)
        df[groups_name] = g_arr

    return df


class SplitsmithKFold:
    """K-fold CV compatible with sklearn's cross-validation API."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        folds = _basic_folds(n, self.n_splits, self.seed)
        for fold in folds:
            yield fold.train_idx, fold.val_idx


class SplitsmithStratifiedKFold:
    """Stratified k-fold CV compatible with sklearn's cross-validation API."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("y is required for stratified splitting")

        df = _ensure_dataframe(X, y=y, target_name="_target_")
        folds = _stratified_folds(df, "_target_", self.n_splits, self.seed)
        for fold in folds:
            yield fold.train_idx, fold.val_idx


class SplitsmithGroupKFold:
    """Group k-fold CV compatible with sklearn's cross-validation API."""

    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required for group splitting")

        df = _ensure_dataframe(X, y=y, groups=groups,
                               target_name="_target_", groups_name="_groups_")
        folds, _ = _group_folds(df, "_groups_", self.n_splits, self.seed)
        for fold in folds:
            yield fold.train_idx, fold.val_idx


class SplitsmithTimeSeriesSplit:
    """Forward-chaining time series CV compatible with sklearn's API.

    Requires X to be a DataFrame with a time column, or X to already be
    sorted chronologically. If time_col is given, it will be used for
    ordering. Otherwise X must have a column named 'timestamp' or be
    pre-sorted.
    """

    def __init__(
        self,
        n_splits: int = 5,
        seed: int = 42,
        time_col: str = "timestamp",
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.seed = seed
        self.time_col = time_col
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        df = _ensure_dataframe(X, y=y, target_name="_target_")
        if self.time_col not in df.columns:
            raise ValueError(
                f"time_col '{self.time_col}' not found. Pass a DataFrame with "
                f"that column or set time_col to the correct name."
            )
        folds, _ = _time_folds(df, self.time_col, self.n_splits, self.seed, self.gap)
        for fold in folds:
            yield fold.train_idx, fold.val_idx


class SplitsmithGroupTimeSeriesSplit:
    """Grouped forward-chaining time series CV compatible with sklearn's API."""

    def __init__(
        self,
        n_splits: int = 5,
        seed: int = 42,
        time_col: str = "timestamp",
        groups_col: str = "group_id",
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.seed = seed
        self.time_col = time_col
        self.groups_col = groups_col
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        df = _ensure_dataframe(X, y=y, target_name="_target_")

        # Prefer explicit groups argument, fall back to groups_col in df
        if groups is not None:
            df["_groups_"] = np.asarray(groups)
            gcol = "_groups_"
        elif self.groups_col in df.columns:
            gcol = self.groups_col
        else:
            raise ValueError(
                f"groups_col '{self.groups_col}' not found and no groups argument given."
            )

        if self.time_col not in df.columns:
            raise ValueError(f"time_col '{self.time_col}' not found in the data.")

        folds, _ = _group_time_folds(
            df, gcol, self.time_col, self.n_splits, self.seed, self.gap
        )
        for fold in folds:
            yield fold.train_idx, fold.val_idx
