"""Tests for min_samples_per_class guardrails."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import split


def _binary_df(n=100):
    return pd.DataFrame({
        "feature": range(n),
        "target": [0, 1] * (n // 2),
    })


def _rare_class_df():
    """DataFrame where class 2 has only 4 samples total."""
    targets = [0] * 50 + [1] * 46 + [2] * 4
    return pd.DataFrame({
        "feature": range(100),
        "target": targets,
    })


def _grouped_rare_df():
    """Grouped data where one class is rare but spread across groups."""
    rows = []
    for i in range(10):
        for j in range(10):
            # put a class-2 sample in several groups so it can appear in each split
            label = 2 if j == 9 else (j % 2)
            rows.append({"group_id": f"g{i}", "feature": i * 10 + j, "target": label})
    return pd.DataFrame(rows)


class TestMinSamplesRandom:
    def test_passes_when_satisfied(self):
        df = _binary_df()
        r = split(df, target="target", min_samples_per_class=5)
        assert len(r.train_idx) > 0

    def test_raises_when_violated(self):
        df = _rare_class_df()
        # class 2 has only 4 samples, so asking for 3 per split is impossible
        # since 4 / 3 splits < 3 per split
        with pytest.raises(ValueError, match="min_samples_per_class"):
            split(df, target="target", min_samples_per_class=3, stratify=True)

    def test_none_means_no_check(self):
        df = _rare_class_df()
        # should not raise even though class 2 might be sparse
        r = split(df, target="target", min_samples_per_class=None)
        assert len(r.train_idx) > 0

    def test_zero_means_no_check(self):
        df = _rare_class_df()
        r = split(df, target="target", min_samples_per_class=0)
        assert len(r.train_idx) > 0

    def test_error_message_is_informative(self):
        df = _rare_class_df()
        with pytest.raises(ValueError) as exc_info:
            split(df, target="target", min_samples_per_class=3, stratify=True)
        msg = str(exc_info.value)
        assert "class" in msg
        assert "sample" in msg


class TestMinSamplesStratifiedGroup:
    def test_passes_when_satisfied(self):
        df = _grouped_rare_df()
        r = split(df, target="target", strategy="group",
                  groups="group_id", stratify=True, min_samples_per_class=1)
        assert len(r.train_idx) > 0

    def test_raises_when_too_strict(self):
        df = _grouped_rare_df()
        with pytest.raises(ValueError, match="min_samples_per_class"):
            split(df, target="target", strategy="group",
                  groups="group_id", stratify=True, min_samples_per_class=50)


class TestMinSamplesDefaultNone:
    def test_default_is_none(self):
        """By default, no min_samples check is done."""
        df = _rare_class_df()
        r = split(df, target="target")
        assert len(r.train_idx) > 0
