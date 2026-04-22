"""CLI smoke tests via Typer's runner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from design_gan.cli import app


def test_help_shows_commands():
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("run", "list-runs", "demo", "viewer"):
        assert cmd in r.output


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
