"""Tests for splitsmith CLI (python -m splitsmith)."""

import json
import subprocess
import sys

import pandas as pd
import pytest


def _write_csv(tmp_path, name="data.csv", n=100):
    import numpy as np
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"feature": rng.standard_normal(n), "target": rng.choice([0, 1], n)})
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _write_grouped_csv(tmp_path, name="grouped.csv"):
    import numpy as np
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(10):
        for j in range(5):
            rows.append({
                "group_id": f"g{i}",
                "timestamp": (base + pd.Timedelta(days=i * 5 + j)).isoformat(),
                "feature": i * 5 + j,
                "target": j % 2,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "splitsmith"] + args,
        capture_output=True, text=True,
    )


class TestAuditCLI:
    def test_basic_pass(self, tmp_path):
        csv = _write_csv(tmp_path)
        result = _run(["audit", csv, "--target", "target"])
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_json_output(self, tmp_path):
        csv = _write_csv(tmp_path)
        out = str(tmp_path / "report.json")
        result = _run(["audit", csv, "--target", "target", "--output", out])
        assert result.returncode == 0
        data = json.loads(open(out).read())
        assert data["ok"] is True

    def test_html_output(self, tmp_path):
        csv = _write_csv(tmp_path)
        out = str(tmp_path / "report.html")
        result = _run(["audit", csv, "--target", "target", "--html", out])
        assert result.returncode == 0
        assert "<!DOCTYPE html>" in open(out).read()

    def test_missing_file(self, tmp_path):
        result = _run(["audit", str(tmp_path / "nope.csv"), "--target", "target"])
        assert result.returncode == 2

    def test_missing_target(self, tmp_path):
        csv = _write_csv(tmp_path)
        result = _run(["audit", csv, "--target", "nonexistent"])
        assert result.returncode == 2

    def test_group_strategy(self, tmp_path):
        csv = _write_grouped_csv(tmp_path)
        result = _run(["audit", csv, "--target", "target", "--strategy", "group", "--groups", "group_id"])
        assert result.returncode == 0

    def test_time_strategy(self, tmp_path):
        csv = _write_grouped_csv(tmp_path)
        result = _run(["audit", csv, "--target", "target", "--strategy", "time", "--time-col", "timestamp"])
        assert result.returncode == 0


class TestAuditCvCLI:
    def test_basic_pass(self, tmp_path):
        csv = _write_csv(tmp_path)
        result = _run(["audit-cv", csv, "--target", "target", "--k", "3"])
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_json_output(self, tmp_path):
        csv = _write_csv(tmp_path)
        out = str(tmp_path / "cv_report.json")
        result = _run(["audit-cv", csv, "--target", "target", "--k", "3", "--output", out])
        assert result.returncode == 0
        data = json.loads(open(out).read())
        assert data["all_ok"] is True

    def test_html_output(self, tmp_path):
        csv = _write_csv(tmp_path)
        out = str(tmp_path / "cv_report.html")
        result = _run(["audit-cv", csv, "--target", "target", "--k", "3", "--html", out])
        assert result.returncode == 0
        assert "<!DOCTYPE html>" in open(out).read()


class TestNoCommand:
    def test_no_args_returns_2(self):
        result = _run([])
        assert result.returncode == 2
