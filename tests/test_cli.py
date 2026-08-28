"""CLI smoke tests via Typer's runner."""

from __future__ import annotations

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


def test_run_rejects_unknown_product_domain():
    r = CliRunner().invoke(app, ["run", "a site", "--domain", "checkout"])
    assert r.exit_code == 2
    assert "unknown product domain" in r.output


def test_list_runs_empty(tmp_path: Path):
    runner = CliRunner()
    r = runner.invoke(app, ["list-runs", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0
    assert "No runs" in r.output


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
