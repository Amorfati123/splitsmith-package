"""JSON export for splitsmith results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .types import CVResult, LeakageReport, SplitResult
from .audit import audit_cv_summary


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def _write_if_path(content: str, path: Optional[str]) -> str:
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(content)
    return content


def split_to_json(split_result: SplitResult, path: Optional[str] = None) -> str:
    """Serialize a SplitResult to JSON."""
    if not isinstance(split_result, SplitResult):
        raise TypeError("split_result must be a SplitResult")
    data = _make_serializable({
        "train_idx": split_result.train_idx,
        "val_idx": split_result.val_idx,
        "test_idx": split_result.test_idx,
        "metadata": split_result.metadata,
    })
    return _write_if_path(json.dumps(data, indent=2), path)


def report_to_json(report: LeakageReport, path: Optional[str] = None) -> str:
    """Serialize a LeakageReport to JSON."""
    if not isinstance(report, LeakageReport):
        raise TypeError("report must be a LeakageReport")
    findings = []
    for f in report.findings:
        findings.append(_make_serializable({
            "id": f.id,
            "severity": f.severity,
            "title": f.title,
            "details": f.details,
            "evidence": f.evidence,
        }))
    data = {
        "ok": report.ok,
        "summary": report.summary(),
        "findings": findings,
    }
    return _write_if_path(json.dumps(data, indent=2), path)


def cv_to_json(cv_result: CVResult, path: Optional[str] = None) -> str:
    """Serialize a CVResult to JSON."""
    if not isinstance(cv_result, CVResult):
        raise TypeError("cv_result must be a CVResult")
    folds = []
    for f in cv_result.folds:
        folds.append(_make_serializable({
            "fold": f.fold,
            "train_idx": f.train_idx,
            "val_idx": f.val_idx,
            "metadata": f.metadata,
        }))
    data = _make_serializable({
        "k": cv_result.k,
        "metadata": cv_result.metadata,
        "folds": folds,
    })
    return _write_if_path(json.dumps(data, indent=2), path)


def audit_cv_to_json(reports: List[LeakageReport], path: Optional[str] = None) -> str:
    """Serialize CV audit reports to JSON."""
    if not isinstance(reports, list):
        raise TypeError("reports must be a list of LeakageReport")
    summary = audit_cv_summary(reports)
    per_fold = []
    for i, r in enumerate(reports):
        findings = []
        for f in r.findings:
            findings.append(_make_serializable({
                "id": f.id, "severity": f.severity, "title": f.title,
                "details": f.details, "evidence": f.evidence,
            }))
        per_fold.append({
            "fold": i,
            "ok": r.ok,
            "summary": r.summary(),
            "findings": findings,
        })
    data = {
        "n_folds": summary["n_folds"],
        "all_ok": summary["all_ok"],
        "total_errors": summary["total_errors"],
        "total_warnings": summary["total_warnings"],
        "per_fold": per_fold,
    }
    return _write_if_path(json.dumps(data, indent=2), path)
