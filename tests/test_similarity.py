"""Tests for similarity-aware leakage detection and splitting."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import SplitResult, audit
from splitsmith.similarity import (
    check_similarity_leakage,
    similarity_split,
    cosine_similarity_matrix,
)


def _make_result(train, val, test):
    return SplitResult(
        train_idx=np.array(train, dtype=int),
        val_idx=np.array(val, dtype=int),
        test_idx=np.array(test, dtype=int),
    )


def _make_embeddings_with_duplicates(n=20, d=8):
    """Create embeddings where rows 0 and 10 are near-identical."""
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(n, d))
    emb[10] = emb[0] + rng.normal(scale=0.001, size=d)  # near-copy of row 0
    return emb


class TestCosineSimMatrix:
    def test_shape(self):
        emb = np.random.default_rng(0).normal(size=(10, 5))
        sim = cosine_similarity_matrix(emb)
        assert sim.shape == (10, 10)

    def test_diagonal_is_one(self):
        emb = np.random.default_rng(0).normal(size=(10, 5))
        sim = cosine_similarity_matrix(emb)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_symmetric(self):
        emb = np.random.default_rng(0).normal(size=(10, 5))
        sim = cosine_similarity_matrix(emb)
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)

    def test_identical_rows_have_sim_1(self):
        emb = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        sim = cosine_similarity_matrix(emb)
        assert abs(sim[0, 1] - 1.0) < 1e-6

    def test_orthogonal_rows_have_sim_0(self):
        emb = np.array([[1, 0], [0, 1]], dtype=float)
        sim = cosine_similarity_matrix(emb)
        assert abs(sim[0, 1]) < 1e-6

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            cosine_similarity_matrix(np.array([1, 2, 3]))


class TestCheckSimilarityLeakage:
    def test_detects_similar_pair_across_splits(self):
        emb = _make_embeddings_with_duplicates()
        sr = _make_result(list(range(10)), list(range(10, 15)), list(range(15, 20)))
        findings = check_similarity_leakage(sr, emb, threshold=0.99)
        warns = [f for f in findings if f.severity == "warn"]
        assert len(warns) == 1
        assert warns[0].evidence["count"] >= 1
        # row 10 (val) should be flagged as similar to row 0 (train)
        pairs = warns[0].evidence["top_pairs"]
        flagged_indices = [p["index"] for p in pairs]
        assert 10 in flagged_indices

    def test_clean_when_no_similar_pairs(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(20, 50))  # high-dim, all random
        sr = _make_result(list(range(10)), list(range(10, 15)), list(range(15, 20)))
        findings = check_similarity_leakage(sr, emb, threshold=0.99)
        infos = [f for f in findings if f.severity == "info"]
        assert len(infos) == 1

    def test_threshold_controls_sensitivity(self):
        emb = _make_embeddings_with_duplicates()
        sr = _make_result(list(range(10)), list(range(10, 15)), list(range(15, 20)))
        strict = check_similarity_leakage(sr, emb, threshold=0.9999)
        loose = check_similarity_leakage(sr, emb, threshold=0.5)
        strict_count = sum(f.evidence.get("count", 0) for f in strict if f.severity == "warn")
        loose_count = sum(f.evidence.get("count", 0) for f in loose if f.severity == "warn")
        assert loose_count >= strict_count

    def test_bad_shape_raises(self):
        emb = np.array([1, 2, 3])
        sr = _make_result([0], [1], [2])
        with pytest.raises(ValueError, match="2-D"):
            check_similarity_leakage(sr, emb, threshold=0.9)

    def test_index_out_of_range_raises(self):
        emb = np.random.default_rng(0).normal(size=(5, 3))
        sr = _make_result([0, 1], [2, 3], [10])  # 10 > 5
        with pytest.raises(ValueError, match="exceed"):
            check_similarity_leakage(sr, emb, threshold=0.9)

    def test_evidence_has_cosine_similarity(self):
        emb = _make_embeddings_with_duplicates()
        sr = _make_result(list(range(10)), list(range(10, 15)), list(range(15, 20)))
        findings = check_similarity_leakage(sr, emb, threshold=0.99)
        warns = [f for f in findings if f.severity == "warn"]
        if warns:
            pair = warns[0].evidence["top_pairs"][0]
            assert "cosine_similarity" in pair
            assert pair["cosine_similarity"] >= 0.99


class TestSimilaritySplit:
    def _df_and_embeddings(self, n=60, d=8):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "feature": range(n),
            "target": rng.choice([0, 1], size=n),
        })
        emb = rng.normal(size=(n, d))
        # create some near-duplicate clusters
        for i in [10, 20, 30]:
            emb[i] = emb[0] + rng.normal(scale=0.001, size=d)
        return df, emb

    def test_returns_split_result(self):
        df, emb = self._df_and_embeddings()
        r = similarity_split(df, "target", emb)
        assert isinstance(r, SplitResult)
        assert len(r.train_idx) > 0

    def test_covers_all_rows(self):
        df, emb = self._df_and_embeddings()
        r = similarity_split(df, "target", emb)
        all_idx = set(r.train_idx.tolist()) | set(r.val_idx.tolist()) | set(r.test_idx.tolist())
        assert all_idx == set(range(len(df)))

    def test_no_overlap(self):
        df, emb = self._df_and_embeddings()
        r = similarity_split(df, "target", emb)
        assert set(r.train_idx).isdisjoint(set(r.val_idx))
        assert set(r.train_idx).isdisjoint(set(r.test_idx))

    def test_reduces_violations(self):
        """Similarity split should have fewer violations than random."""
        df, emb = self._df_and_embeddings()
        from splitsmith import split as plain_split
        from splitsmith.similarity import _count_cross_split_violations

        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = emb / norms

        random_r = plain_split(df, "target", seed=42)
        sim_r = similarity_split(df, "target", emb, seed=42,
                                  similarity_threshold=0.99, max_iterations=500)

        random_asgn = np.zeros(len(df), dtype=int)
        random_asgn[random_r.val_idx] = 1
        random_asgn[random_r.test_idx] = 2

        sim_asgn = np.zeros(len(df), dtype=int)
        sim_asgn[sim_r.val_idx] = 1
        sim_asgn[sim_r.test_idx] = 2

        random_v = _count_cross_split_violations(normed, random_asgn, 0.99)
        sim_v = _count_cross_split_violations(normed, sim_asgn, 0.99)
        assert sim_v <= random_v

    def test_deterministic(self):
        df, emb = self._df_and_embeddings()
        r1 = similarity_split(df, "target", emb, seed=42)
        r2 = similarity_split(df, "target", emb, seed=42)
        assert np.array_equal(r1.train_idx, r2.train_idx)

    def test_metadata_has_strategy(self):
        df, emb = self._df_and_embeddings()
        r = similarity_split(df, "target", emb)
        assert r.metadata["strategy"] == "similarity"
        assert "optimization" in r.metadata
        assert "final_violations" in r.metadata["optimization"]

    def test_grouped_mode(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group_id": [f"g{i // 5}" for i in range(30)],
            "feature": range(30),
            "target": rng.choice([0, 1], size=30),
        })
        emb = rng.normal(size=(30, 5))
        r = similarity_split(df, "target", emb, groups="group_id")
        train_g = set(df.iloc[r.train_idx]["group_id"])
        val_g = set(df.iloc[r.val_idx]["group_id"])
        assert train_g.isdisjoint(val_g)
        assert r.metadata["optimization"]["mode"] == "grouped"

    def test_bad_embedding_shape_raises(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1]})
        emb = np.random.default_rng(0).normal(size=(10, 3))  # wrong n
        with pytest.raises(ValueError, match="doesn't match"):
            similarity_split(df, "target", emb)

    def test_missing_target_raises(self):
        df = pd.DataFrame({"f": [1, 2, 3]})
        emb = np.random.default_rng(0).normal(size=(3, 2))
        with pytest.raises(ValueError, match="target"):
            similarity_split(df, "nonexistent", emb)
