"""Similarity-aware leakage detection and splitting.

Detects and prevents cross-split leakage in embedding space. While exact
and near-duplicate checks catch identical or numerically close rows,
similarity-aware checking catches semantically similar items: images of
the same subject, audio from the same speaker, paraphrases of the same
text. These are invisible to other tools but obvious in embedding space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .types import Finding, LeakageReport, SplitResult
from ._meta import build_metadata

_MAX_EVIDENCE_PAIRS = 10


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for an embedding matrix.

    Parameters
    ----------
    embeddings : ndarray of shape (n_samples, n_dims)

    Returns
    -------
    ndarray of shape (n_samples, n_samples) with values in [-1, 1]
    """
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = emb / norms
    return normalized @ normalized.T


def check_similarity_leakage(
    split_result: SplitResult,
    embeddings: np.ndarray,
    threshold: float = 0.9,
) -> List[Finding]:
    """Check whether highly similar items ended up in different splits.

    For each val/test sample, finds the most similar train sample using
    cosine similarity. Pairs exceeding the threshold are flagged.

    Parameters
    ----------
    split_result : SplitResult
    embeddings : ndarray of shape (n_samples, n_dims)
    threshold : cosine similarity above which a pair is flagged

    Returns
    -------
    List of Finding objects (can be added to a LeakageReport)
    """
    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")

    train_idx = split_result.train_idx
    val_idx = split_result.val_idx
    test_idx = split_result.test_idx

    max_idx = max(
        train_idx.max() if len(train_idx) else 0,
        val_idx.max() if len(val_idx) else 0,
        test_idx.max() if len(test_idx) else 0,
    )
    if max_idx >= len(emb):
        raise ValueError(
            f"split indices (max={max_idx}) exceed embedding rows ({len(emb)})"
        )

    # normalize once
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = emb / norms

    train_normed = normed[train_idx]

    flagged_pairs = []

    for check_name, check_idx in [("val", val_idx), ("test", test_idx)]:
        if len(check_idx) == 0 or len(train_idx) == 0:
            continue
        check_normed = normed[check_idx]

        # cosine similarities: (n_check, n_train)
        sims = check_normed @ train_normed.T
        max_sims = sims.max(axis=1)
        argmax_sims = sims.argmax(axis=1)

        for i in range(len(check_idx)):
            if max_sims[i] >= threshold:
                flagged_pairs.append({
                    "index": int(check_idx[i]),
                    "split": check_name,
                    "closest_train_index": int(train_idx[argmax_sims[i]]),
                    "cosine_similarity": round(float(max_sims[i]), 4),
                })

    findings = []

    if flagged_pairs:
        flagged_pairs.sort(key=lambda p: -p["cosine_similarity"])
        findings.append(Finding(
            id="similarity_leakage",
            severity="warn",
            title=f"Cross-split similarity leakage ({len(flagged_pairs)} pairs above {threshold})",
            details=(
                f"{len(flagged_pairs)} val/test sample(s) have cosine similarity "
                f">= {threshold} with a train sample. These may represent the same "
                f"underlying entity in embedding space."
            ),
            evidence={
                "count": len(flagged_pairs),
                "threshold": threshold,
                "top_pairs": flagged_pairs[:_MAX_EVIDENCE_PAIRS],
                "max_similarity": flagged_pairs[0]["cosine_similarity"],
            },
        ))
    else:
        findings.append(Finding(
            id="similarity_leakage",
            severity="info",
            title="No cross-split similarity leakage",
            details=(
                f"No val/test samples have cosine similarity >= {threshold} "
                f"with any train sample."
            ),
        ))

    return findings


def similarity_split(
    df: pd.DataFrame,
    target: str,
    embeddings: np.ndarray,
    *,
    groups: Optional[str] = None,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    similarity_threshold: float = 0.9,
    max_iterations: int = 1000,
    patience: int = 50,
) -> SplitResult:
    """Split data while minimizing cross-split similarity leakage.

    Uses iterative refinement: starts with a random feasible assignment,
    then swaps samples (or groups) between splits to reduce the number
    of cross-split pairs exceeding the similarity threshold.

    Parameters
    ----------
    df : DataFrame
    target : target column name
    embeddings : ndarray of shape (n_samples, n_dims)
    groups : if given, swap entire groups (preserves group exclusivity)
    ratios : target (train, val, test) proportions
    seed : random seed
    similarity_threshold : cosine similarity above which a pair is penalized
    max_iterations : max swap iterations
    patience : stop if no improvement for this many iterations
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame")
    if groups is not None and groups not in df.columns:
        raise ValueError(f"groups column '{groups}' not found in DataFrame")

    emb = np.asarray(embeddings, dtype=float)
    n = len(df)
    if emb.ndim != 2 or emb.shape[0] != n:
        raise ValueError(
            f"embeddings shape {emb.shape} doesn't match DataFrame length {n}"
        )
    if n < 3:
        raise ValueError("need at least 3 rows")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")

    # precompute normalized embeddings
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = emb / norms

    rng = np.random.default_rng(seed)
    y = df[target]

    if groups is not None:
        result, opt_meta = _similarity_split_grouped(
            df, y, groups, normed, ratios, similarity_threshold,
            rng, max_iterations, patience,
        )
    else:
        result, opt_meta = _similarity_split_ungrouped(
            n, y, normed, ratios, similarity_threshold,
            rng, max_iterations, patience,
        )

    params = {
        "target": target, "groups": groups, "ratios": tuple(ratios),
        "seed": seed, "similarity_threshold": similarity_threshold,
        "max_iterations": max_iterations, "patience": patience,
        "embedding_dims": emb.shape[1],
    }
    result.metadata = {
        "strategy": "similarity",
        "ratios": tuple(ratios),
        "seed": seed,
        "n_rows": n,
        "similarity_threshold": similarity_threshold,
        "split_sizes": {
            "train": len(result.train_idx),
            "val": len(result.val_idx),
            "test": len(result.test_idx),
        },
        "optimization": opt_meta,
        "reproducibility": build_metadata(df, params),
    }
    return result


def _count_cross_split_violations(
    normed: np.ndarray, assignments: np.ndarray, threshold: float,
    train_id: int = 0,
) -> int:
    """Count val/test samples that have a train neighbor above threshold."""
    train_mask = assignments == train_id
    if train_mask.sum() == 0:
        return 0

    train_normed = normed[train_mask]
    count = 0
    for split_id in [1, 2]:
        check_mask = assignments == split_id
        if check_mask.sum() == 0:
            continue
        check_normed = normed[check_mask]
        sims = check_normed @ train_normed.T
        count += int((sims.max(axis=1) >= threshold).sum())
    return count


def _similarity_split_ungrouped(n, y, normed, ratios, threshold,
                                 rng, max_iter, patience):
    """Row-level similarity-aware splitting."""
    assignments = np.zeros(n, dtype=int)
    shuffled = np.arange(n)
    rng.shuffle(shuffled)
    train_n = int(ratios[0] * n)
    val_n = int(ratios[1] * n)
    assignments[shuffled[:train_n]] = 0
    assignments[shuffled[train_n:train_n + val_n]] = 1
    assignments[shuffled[train_n + val_n:]] = 2

    best_violations = _count_cross_split_violations(normed, assignments, threshold)
    best_assignments = assignments.copy()
    no_improve = 0
    iters_used = 0

    for iteration in range(max_iter):
        i = rng.integers(n)
        old = assignments[i]
        new = rng.choice([s for s in range(3) if s != old])
        assignments[i] = new

        v = _count_cross_split_violations(normed, assignments, threshold)
        if v < best_violations:
            best_violations = v
            best_assignments = assignments.copy()
            no_improve = 0
        else:
            assignments[i] = old
            no_improve += 1

        iters_used = iteration + 1
        if no_improve >= patience:
            break
        if best_violations == 0:
            break

    assignments = best_assignments
    train_idx = np.where(assignments == 0)[0].astype(int)
    val_idx = np.where(assignments == 1)[0].astype(int)
    test_idx = np.where(assignments == 2)[0].astype(int)

    opt_meta = {
        "iterations": iters_used,
        "final_violations": best_violations,
        "converged": no_improve >= patience or best_violations == 0,
        "mode": "ungrouped",
    }
    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx), opt_meta


def _similarity_split_grouped(df, y, groups, normed, ratios, threshold,
                               rng, max_iter, patience):
    """Group-level similarity-aware splitting."""
    group_col = df[groups]
    unique_groups = np.array(group_col.unique().tolist())
    n_groups = len(unique_groups)
    n = len(df)

    if n_groups < 3:
        raise ValueError(f"Need at least 3 groups, got {n_groups}")

    group_to_rows = {}
    for g in unique_groups:
        group_to_rows[g] = np.where(group_col == g)[0]

    # initial assignment by ratios
    shuffled_g = unique_groups.copy()
    rng.shuffle(shuffled_g)
    group_assignments = {}
    current = {0: 0, 1: 0, 2: 0}
    targets = {0: ratios[0] * n, 1: ratios[1] * n, 2: ratios[2] * n}
    for g in shuffled_g:
        sz = len(group_to_rows[g])
        best = max(range(3), key=lambda s: targets[s] - current[s])
        group_assignments[g] = best
        current[best] += sz

    assignments = np.zeros(n, dtype=int)
    for g, sid in group_assignments.items():
        assignments[group_to_rows[g]] = sid

    best_violations = _count_cross_split_violations(normed, assignments, threshold)
    best_ga = group_assignments.copy()
    no_improve = 0
    iters_used = 0

    for iteration in range(max_iter):
        g = rng.choice(unique_groups)
        old = group_assignments[g]
        new = rng.choice([s for s in range(3) if s != old])
        group_assignments[g] = new
        assignments[group_to_rows[g]] = new

        v = _count_cross_split_violations(normed, assignments, threshold)
        if v < best_violations:
            best_violations = v
            best_ga = group_assignments.copy()
            no_improve = 0
        else:
            group_assignments[g] = old
            assignments[group_to_rows[g]] = old
            no_improve += 1

        iters_used = iteration + 1
        if no_improve >= patience:
            break
        if best_violations == 0:
            break

    for g, sid in best_ga.items():
        assignments[group_to_rows[g]] = sid

    train_idx = np.where(assignments == 0)[0].astype(int)
    val_idx = np.where(assignments == 1)[0].astype(int)
    test_idx = np.where(assignments == 2)[0].astype(int)

    opt_meta = {
        "iterations": iters_used,
        "final_violations": best_violations,
        "converged": no_improve >= patience or best_violations == 0,
        "mode": "grouped",
        "n_groups": n_groups,
    }
    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx), opt_meta
