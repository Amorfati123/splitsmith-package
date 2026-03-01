"""Tests for splitsmith.export."""

import json
import os
import numpy as np
import pandas as pd
import pytest

from splitsmith import (
    SplitResult, LeakageReport, CVResult,
    split, audit, k_fold, audit_cv,
    split_to_json, report_to_json, cv_to_json, audit_cv_to_json,
)


def _simple_df(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"feature": rng.standard_normal(n), "target": rng.choice([0, 1], n)})


class TestSplitToJson:
    def test_returns_string(self):
        sr = split(_simple_df(), "target")
        result = split_to_json(sr)
        assert isinstance(result, str)

    def test_valid_json(self):
        sr = split(_simple_df(), "target")
        data = json.loads(split_to_json(sr))
        assert "train_idx" in data
        assert "val_idx" in data
        assert "test_idx" in data
        assert "metadata" in data

    def test_write_to_file(self, tmp_path):
        sr = split(_simple_df(), "target")
        out = tmp_path / "split.json"
        result = split_to_json(sr, path=str(out))
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["train_idx"]) == len(sr.train_idx)

    def test_type_error(self):
        with pytest.raises(TypeError):
            split_to_json("not a split")

    def test_numpy_types_serialized(self):
        sr = split(_simple_df(), "target")
        text = split_to_json(sr)
        # Should not raise - all numpy types converted
        json.loads(text)


class TestReportToJson:
    def test_returns_string(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        result = report_to_json(report)
        assert isinstance(result, str)

    def test_valid_json_structure(self):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        data = json.loads(report_to_json(report))
        assert "ok" in data
        assert "summary" in data
        assert "findings" in data

    def test_write_to_file(self, tmp_path):
        df = _simple_df()
        report = audit(df, split(df, "target"), "target")
        out = tmp_path / "report.json"
        report_to_json(report, path=str(out))
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["ok"] is True

    def test_type_error(self):
        with pytest.raises(TypeError):
            report_to_json("not a report")


class TestCvToJson:
    def test_returns_string(self):
        cv = k_fold(_simple_df(), "target", k=3)
        result = cv_to_json(cv)
        assert isinstance(result, str)

    def test_valid_json_structure(self):
        cv = k_fold(_simple_df(), "target", k=3)
        data = json.loads(cv_to_json(cv))
        assert data["k"] == 3
        assert len(data["folds"]) == 3

    def test_write_to_file(self, tmp_path):
        cv = k_fold(_simple_df(), "target", k=3)
        out = tmp_path / "cv.json"
        cv_to_json(cv, path=str(out))
        assert out.exists()

    def test_type_error(self):
        with pytest.raises(TypeError):
            cv_to_json("not a cv")


class TestAuditCvToJson:
    def test_returns_string(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        result = audit_cv_to_json(reports)
        assert isinstance(result, str)

    def test_valid_json_structure(self):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        data = json.loads(audit_cv_to_json(reports))
        assert "all_ok" in data
        assert "per_fold" in data
        assert len(data["per_fold"]) == 3

    def test_write_to_file(self, tmp_path):
        df = _simple_df()
        cv = k_fold(df, "target", k=3)
        reports = audit_cv(df, cv, "target")
        out = tmp_path / "audit_cv.json"
        audit_cv_to_json(reports, path=str(out))
        assert out.exists()

    def test_type_error(self):
        with pytest.raises(TypeError):
            audit_cv_to_json("not a list")
