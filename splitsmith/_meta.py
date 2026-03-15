"""Reproducibility metadata for splitsmith results."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

from ._version import __version__


def _df_content_hash(df: pd.DataFrame) -> str:
    """SHA-256 hash of the DataFrame contents (first 16 hex chars)."""
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=True)
        combined = hashlib.sha256(row_hashes.values.tobytes())
        return combined.hexdigest()[:16]
    except (TypeError, ValueError):
        # fallback for unhashable columns: hash the shape + column names
        sig = f"{df.shape}|{'|'.join(df.columns)}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]


def _schema_hash(df: pd.DataFrame) -> str:
    """Hash of column names and dtypes, independent of data."""
    parts = [f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes)]
    sig = "|".join(parts)
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def build_metadata(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the reproducibility block that gets merged into result metadata.

    Returns a dict with environment info, data fingerprints, and the full
    parameter set used to produce the split or CV result.
    """
    return {
        "splitsmith_version": __version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_hash": _df_content_hash(df),
        "schema_hash": _schema_hash(df),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "params": params,
    }
