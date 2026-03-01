"""CLI entry point: python -m splitsmith."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .split import split
from .cv import k_fold
from .audit import audit, audit_cv
from .export import report_to_json, audit_cv_to_json
from .report import report_to_html, audit_cv_to_html


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitsmith",
        description="ML dataset splitting and leakage detection.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- audit subcommand ---
    p_audit = sub.add_parser("audit", help="Split data and audit for leakage")
    p_audit.add_argument("data", help="Path to CSV file")
    p_audit.add_argument("--target", required=True, help="Target column name")
    p_audit.add_argument("--strategy", default="random", choices=["random", "group", "time", "group_time"])
    p_audit.add_argument("--groups", default=None, help="Groups column name")
    p_audit.add_argument("--time-col", default=None, help="Time column name")
    p_audit.add_argument("--seed", type=int, default=42)
    p_audit.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p_audit.add_argument("--output", default=None, help="Save JSON report to path")
    p_audit.add_argument("--html", default=None, help="Save HTML report to path")

    # --- audit-cv subcommand ---
    p_cv = sub.add_parser("audit-cv", help="K-fold CV and audit each fold")
    p_cv.add_argument("data", help="Path to CSV file")
    p_cv.add_argument("--target", required=True, help="Target column name")
    p_cv.add_argument("--k", type=int, default=5, help="Number of folds")
    p_cv.add_argument("--strategy", default="basic", choices=["basic", "stratified", "group", "time", "group_time"])
    p_cv.add_argument("--groups", default=None, help="Groups column name")
    p_cv.add_argument("--time-col", default=None, help="Time column name")
    p_cv.add_argument("--seed", type=int, default=42)
    p_cv.add_argument("--output", default=None, help="Save JSON report to path")
    p_cv.add_argument("--html", default=None, help="Save HTML report to path")

    return parser


def main(argv=None) -> int:
    """Run CLI and return exit code: 0=pass, 1=leakage found, 2=runtime error."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error reading {args.data}: {e}", file=sys.stderr)
        return 2

    if args.command == "audit":
        return _run_audit(df, args)
    elif args.command == "audit-cv":
        return _run_audit_cv(df, args)
    else:
        parser.print_help()
        return 2


def _run_audit(df, args) -> int:
    try:
        sr = split(
            df, target=args.target, strategy=args.strategy,
            groups=args.groups, time_col=args.time_col,
            seed=args.seed, ratios=tuple(args.ratios),
        )
        report = audit(df, sr, args.target, groups=args.groups, time_col=args.time_col)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    _print_summary(report)

    if args.output:
        report_to_json(report, path=args.output)
        print(f"JSON saved to {args.output}")
    if args.html:
        report_to_html(report, path=args.html)
        print(f"HTML saved to {args.html}")

    return 0 if report.ok else 1


def _run_audit_cv(df, args) -> int:
    try:
        cv = k_fold(
            df, target=args.target, k=args.k, strategy=args.strategy,
            groups=args.groups, time_col=args.time_col, seed=args.seed,
        )
        reports = audit_cv(df, cv, args.target, groups=args.groups, time_col=args.time_col)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    all_ok = all(r.ok for r in reports)
    for i, r in enumerate(reports):
        counts = r.summary()
        status = "PASS" if r.ok else "FAIL"
        print(f"  Fold {i}: {status} ({counts['error']} error, {counts['warn']} warn, {counts['info']} info)")

    status = "PASS" if all_ok else "FAIL"
    print(f"\nOverall: {status}")

    if args.output:
        audit_cv_to_json(reports, path=args.output)
        print(f"JSON saved to {args.output}")
    if args.html:
        audit_cv_to_html(reports, path=args.html)
        print(f"HTML saved to {args.html}")

    return 0 if all_ok else 1


def _print_summary(report):
    counts = report.summary()
    status = "PASS" if report.ok else "FAIL"
    print(f"\nAudit: {status}")
    print(f"  Errors:   {counts['error']}")
    print(f"  Warnings: {counts['warn']}")
    print(f"  Info:     {counts['info']}")
    for f in sorted(report.findings, key=lambda x: {"error": 0, "warn": 1, "info": 2}.get(x.severity, 3)):
        icon = {"error": "[ERROR]", "warn": "[WARN]", "info": "[INFO]"}.get(f.severity, "[?]")
        print(f"  {icon} {f.title}: {f.details}")


if __name__ == "__main__":
    sys.exit(main())
