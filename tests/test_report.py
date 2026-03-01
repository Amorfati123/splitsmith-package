"""Tests for splitsmith.report."""

import numpy as np
import pandas as pd
import pytest

from splitsmith import (
    split, audit, k_fold, audit_cv,
    report_to_html, audit_cv_to_html,
)


def _simple_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"feature": rng.standard_normal(n), "target": rng.choice([0, 1], n)})


class TestReportToHtml:
    def test_returns_string(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        html = report_to_html(report)
        assert isinstance(html, str)

    def test_contains_pass(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        html = report_to_html(report)
        assert "PASS" in html

    def test_contains_title(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        html = report_to_html(report, title="My Custom Title")
        assert "My Custom Title" in html

    def test_contains_findings(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        html = report_to_html(report)
        assert "No index overlap" in html

    def test_write_to_file(self, tmp_path):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        out = tmp_path / "report.html"
        returned = report_to_html(report, path=str(out))
        with open(out, "r", encoding="utf-8", newline="") as f:
            written = f.read()
        assert returned == written

    def test_type_error(self):
        with pytest.raises(TypeError):
            report_to_html("not a report")

    def test_self_contained(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        html = report_to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<style>" in html


class TestAuditCvToHtml:
    def test_returns_string(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        html = audit_cv_to_html(reports)
        assert isinstance(html, str)

    def test_contains_fold_sections(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        html = audit_cv_to_html(reports)
        assert "Fold 0" in html
        assert "Fold 1" in html
        assert "Fold 2" in html

    def test_collapsible_details(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        html = audit_cv_to_html(reports)
        assert "<details" in html
        assert "<summary" in html

    def test_write_to_file(self, tmp_path):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        out = tmp_path / "cv_report.html"
        returned = audit_cv_to_html(reports, path=str(out))
        with open(out, "r", encoding="utf-8", newline="") as f:
            written = f.read()
        assert returned == written

    def test_type_error(self):
        with pytest.raises(TypeError):
            audit_cv_to_html("not a list")
