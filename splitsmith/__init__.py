from ._version import __version__
from .types import SplitResult, LeakageReport, Finding, FoldResult, CVResult
from .split import split, repeated_split
from .cv import k_fold, repeated_k_fold
from .audit import audit, audit_cv, audit_cv_summary
from .optimize import optimize_split
from .score import compute_score, LeakageScore
from .similarity import (
    check_similarity_leakage,
    similarity_split,
    cosine_similarity_matrix,
)
from .feasibility import check_feasibility, FeasibilityReport
from .diagnose import diagnose, DiagnosticReport
from .export import split_to_json, report_to_json, cv_to_json, audit_cv_to_json
from .report import report_to_html, audit_cv_to_html
from .compat import (
    SplitsmithKFold,
    SplitsmithStratifiedKFold,
    SplitsmithGroupKFold,
    SplitsmithTimeSeriesSplit,
    SplitsmithGroupTimeSeriesSplit,
)
