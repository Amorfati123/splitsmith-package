from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "train_size": len(self.train_idx),
            "val_size": len(self.val_idx),
            "test_size": len(self.test_idx),
            "metadata": self.metadata,
        }


@dataclass
class Finding:
    id: str
    severity: str  # "info" | "warn" | "error"
    title: str
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageReport:
    findings: List[Finding] = field(default_factory=list)

    def add(self, finding: Finding):
        self.findings.append(finding)

    def summary(self) -> Dict[str, int]:
        counts = {"info": 0, "warn": 0, "error": 0}
        for f in self.findings:
            if f.severity in counts:
                counts[f.severity] += 1
        return counts

    @property
    def ok(self) -> bool:
        return all(f.severity != "error" for f in self.findings)

    def __repr__(self) -> str:
        counts = self.summary()
        return f"LeakageReport({counts['error']} error, {counts['warn']} warn, {counts['info']} info)"


@dataclass
class FoldResult:
    """Result of a single CV fold."""
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CVResult:
    """Result of k-fold cross-validation."""
    folds: List[FoldResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.folds)

    def summary(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "fold_sizes": [
                {"train": len(f.train_idx), "val": len(f.val_idx)}
                for f in self.folds
            ],
            **self.metadata,
        }