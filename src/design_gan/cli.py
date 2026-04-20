"""Typer-based CLI for running the loop and launching the viewer."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import orchestrator, storage

app = typer.Typer(add_completion=False, help="Autoresearch-style design evolution loop.")
console = Console()


def _default_runs_dir() -> Path:
    return Path(os.environ.get("DESIGN_GAN_RUNS_DIR", "./runs"))


def _default_model() -> str:
    return os.environ.get("DESIGN_GAN_MODEL", "claude-sonnet-4-6")


@app.command()
def run(
    brief: str = typer.Argument(..., help="Describe the site the generator should build."),
    max_iters: int = typer.Option(15, help="Maximum generate/critique iterations."),
    patience: int = typer.Option(3, help="Stop after this many iters without improvement."),
    tolerance: float = typer.Option(1.0, help="Min composite-score gain to count as progress."),
    model: str = typer.Option(None, help="Override the Claude model ID."),
    runs_dir: Path = typer.Option(None, help="Where to store per-iteration artifacts."),
) -> None:
    """Run one evolution loop for BRIEF until the score plateaus."""
    load_dotenv()
    runs_dir = runs_dir or _default_runs_dir()
    cfg = orchestrator.LoopConfig(
        brief=brief,
        runs_dir=runs_dir,
        db_path=runs_dir / "design-gan.sqlite",
        model=model or _default_model(),
        max_iters=max_iters,
        patience=patience,
        tolerance=tolerance,
    )
    result = orchestrator.run_loop_sync(cfg, console=console)
    console.rule("[bold green]Done")
    console.print(
        f"run_id={result.run_id}  best_iter={result.best_iter}  "
        f"best_score={result.best_score:.1f}  iters={result.iterations}  "
        f"status={result.status}"
    )


@app.command()
def list_runs(
    runs_dir: Path = typer.Option(None, help="Runs directory containing the sqlite db."),
) -> None:
    """List prior runs stored in the sqlite db."""
    load_dotenv()
    runs_dir = runs_dir or _default_runs_dir()
    store = storage.Storage(runs_dir / "design-gan.sqlite")
    rows = store.list_runs()
    if not rows:
        console.print("[yellow]No runs yet.[/yellow]")
        return
    table = Table(title="Runs")
    for col in ("id", "brief", "model", "best_iter", "best_score", "status"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["id"]),
            (r["brief"] or "")[:60],
            r["model"],
            str(r["best_iter"]) if r["best_iter"] is not None else "-",
            f"{r['best_score']:.1f}" if r["best_score"] is not None else "-",
            r["status"],
        )
    console.print(table)


@app.command()
def demo(
    runs_dir: Path = typer.Option(None, help="Where to write demo artifacts."),
) -> None:
    """Seed a fake run so the viewer has something to show (no API key needed)."""
    from . import demo as demo_mod

    runs_dir = runs_dir or _default_runs_dir()
    run_id = demo_mod.seed_demo(runs_dir)
    console.print(f"[green]Seeded demo run #{run_id}[/green] in {runs_dir}")


@app.command()
def viewer(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    runs_dir: Path = typer.Option(None, help="Runs directory containing the sqlite db."),
) -> None:
    """Launch the FastAPI viewer to browse iterations."""
    import uvicorn

    load_dotenv()
    runs_dir = runs_dir or _default_runs_dir()
    os.environ["DESIGN_GAN_RUNS_DIR"] = str(runs_dir)
    uvicorn.run("design_gan.viewer:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
