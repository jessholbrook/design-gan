"""CLI smoke tests via Typer's runner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from design_gan.cli import app


def test_help_shows_commands():
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in (
        "run",
        "benchmark-evaluator",
        "calibrate-evaluator",
        "capture-evaluator-case",
        "audit-evaluator-corpus",
        "list-incumbents",
        "list-runs",
        "demo",
        "viewer",
        "export",
    ):
        assert cmd in r.output


def test_run_help_shows_product_evaluation_controls():
    r = CliRunner().invoke(app, ["run", "--help"])
    assert r.exit_code == 0
    assert "--domain" in r.output
    assert "--evaluation-trials" in r.output
    assert "--promotion-alpha" in r.output


def test_evaluator_commands_expose_confidence_level():
    runner = CliRunner()
    for command in ("benchmark-evaluator", "calibrate-evaluator"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--confidence" in result.output


def test_run_rejects_unknown_product_domain():
    r = CliRunner().invoke(app, ["run", "a site", "--domain", "checkout"])
    assert r.exit_code == 2
    assert "unknown product domain" in r.output


def test_list_runs_empty(tmp_path: Path):
    runner = CliRunner()
    r = runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert "No runs" in r.output


def test_list_incumbents_empty(tmp_path: Path):
    r = CliRunner().invoke(app, ["list-incumbents", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert "No incumbents" in r.output


def test_capture_evaluator_case_rejects_unknown_label(tmp_path: Path):
    r = CliRunner().invoke(
        app,
        [
            "capture-evaluator-case",
            "1",
            "1",
            "--task-id",
            "task",
            "--case-id",
            "captured-case",
            "--label",
            "maybe",
            "--reviewer",
            "operator-1",
            "--rationale",
            "This label has a concrete operator rationale.",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 2
    assert "pass or fail" in r.output


def test_audit_evaluator_corpus_fails_closed_and_writes_report(tmp_path: Path):
    report_path = tmp_path / "readiness.json"
    r = CliRunner().invoke(
        app,
        [
            "audit-evaluator-corpus",
            "--runs-dir",
            str(tmp_path),
            "--json-out",
            str(report_path),
        ],
    )

    assert r.exit_code == 1
    assert "BLOCKED" in r.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ready"] is False
    assert report["requirements"]["minimum_cases"] == 24


def test_demo_seeds_and_then_list_runs_shows_it(tmp_path: Path):
    runner = CliRunner()
    r = runner.invoke(app, ["demo", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "Seeded demo run" in r.output

    r2 = runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path)])
    assert r2.exit_code == 0
    assert "DEMO" in r2.output


def test_export_writes_best_iteration_html(tmp_path: Path):
    runner = CliRunner()
    r = runner.invoke(app, ["demo", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output

    out = tmp_path / "best.html"
    r2 = runner.invoke(app, ["export", "1", "--runs-dir", str(tmp_path), "--out", str(out)])
    assert r2.exit_code == 0, r2.output
    assert "Exported" in r2.output
    assert out.is_file()
    assert "<html" in out.read_text(encoding="utf-8").lower()


def test_export_unknown_run_fails(tmp_path: Path):
    runner = CliRunner()
    # Initialize an empty db via list-runs, then export a missing run.
    runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path)])
    r = runner.invoke(app, ["export", "42", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 1
    assert "not found" in r.output
