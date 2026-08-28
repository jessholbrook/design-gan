"""Typer-based CLI for running the loop and launching the viewer."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import evaluator_benchmark, orchestrator, product_domains, storage

app = typer.Typer(add_completion=False, help="Autoresearch-style design evolution loop.")
console = Console()


def _default_runs_dir() -> Path:
    return Path(os.environ.get("DESIGN_GAN_RUNS_DIR", "./runs"))


def _default_model() -> str:
    return os.environ.get("DESIGN_GAN_MODEL", "claude-sonnet-4-6")


def _load_env() -> None:
    load_dotenv(override=True)
    # The Agent SDK routes through Claude Code OAuth (Max plan) when no API key
    # is set. An empty string still counts as "set" for some clients, so clear it.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ.pop("ANTHROPIC_API_KEY", None)


@app.command()
def run(
    brief: str = typer.Argument(..., help="Describe the site the generator should build."),
    max_iters: int = typer.Option(15, help="Maximum generate/critique iterations."),
    patience: int = typer.Option(3, help="Stop after this many iters without improvement."),
    tolerance: float = typer.Option(1.0, help="Min primary-score gain to count as progress."),
    model: str = typer.Option(None, help="Override the Claude model ID."),
    runs_dir: Path = typer.Option(None, help="Where to store per-iteration artifacts."),
    domain: str = typer.Option(
        "landing-page",
        help="Frozen product domain: landing-page, lead-generation, or storefront.",
    ),
    evaluation_trials: int = typer.Option(
        6, min=1, max=50, help="Repeated browser trials per frozen task."
    ),
    promotion_alpha: float = typer.Option(
        0.05, min=0.0001, max=1.0, help="One-sided promotion significance threshold."
    ),
) -> None:
    """Run one evolution loop for BRIEF until the score plateaus."""
    _load_env()
    runs_dir = runs_dir or _default_runs_dir()
    try:
        product_domains.get_domain(domain)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--domain") from exc
    cfg = orchestrator.LoopConfig(
        brief=brief,
        runs_dir=runs_dir,
        db_path=runs_dir / "design-gan.sqlite",
        model=model or _default_model(),
        max_iters=max_iters,
        patience=patience,
        tolerance=tolerance,
        design_domain=domain,
        evaluation_trials=evaluation_trials,
        promotion_alpha=promotion_alpha,
    )
    result = orchestrator.run_loop_sync(cfg, console=console)
    console.rule("[bold green]Done")
    best_score_txt = f"{result.best_score:.1f}" if result.best_score is not None else "—"
    best_iter_txt = str(result.best_iter) if result.best_iter else "—"
    console.print(
        f"run_id={result.run_id}  best_iter={best_iter_txt}  "
        f"best_score={best_score_txt}  iters={result.iterations}  "
        f"status={result.status}  "
        f"holdout={('pass' if result.holdout_passed else 'fail') if result.holdout_passed is not None else 'n/a'}"
    )


@app.command("benchmark-evaluator")
def benchmark_evaluator(
    json_out: Path = typer.Option(None, help="Optional path for the machine-readable report."),
) -> None:
    """Run the labeled Chromium validity corpus against the semantic actor."""
    report = asyncio.run(evaluator_benchmark.run_benchmark())
    table = Table(title=f"Evaluator benchmark · {report.actor}")
    for column in ("case", "domain", "expected", "actual", "result"):
        table.add_column(column)
    for result in report.results:
        table.add_row(
            result.id,
            result.domain,
            "pass" if result.expected_pass else "fail",
            "pass" if result.actual_pass else "fail",
            "correct" if result.correct else "MISS",
        )
    console.print(table)
    console.print(
        f"accuracy={report.accuracy:.1%} ({report.correct}/{report.total}) "
        f"corpus=v{report.corpus_version}"
    )
    if json_out is not None:
        report.write(json_out)
        console.print(f"report={json_out}")
    if report.correct != report.total:
        raise typer.Exit(1)


@app.command()
def list_runs(
    runs_dir: Path = typer.Option(None, help="Runs directory containing the sqlite db."),
) -> None:
    """List prior runs stored in the sqlite db."""
    _load_env()
    runs_dir = runs_dir or _default_runs_dir()
    store = storage.Storage(runs_dir / "design-gan.sqlite")
    rows = store.list_runs()
    if not rows:
        console.print("[yellow]No runs yet.[/yellow]")
        return
    table = Table(title="Runs")
    for col in ("id", "brief", "model", "best_iter", "best_score", "holdout", "status"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["id"]),
            (r["brief"] or "")[:60],
            r["model"],
            str(r["best_iter"]) if r["best_iter"] is not None else "-",
            f"{r['best_score']:.1f}" if r["best_score"] is not None else "-",
            (
                "pass"
                if r.get("holdout_passed") is True
                else ("fail" if r.get("holdout_passed") is False else "-")
            ),
            r["status"],
        )
    console.print(table)


@app.command()
def converse(
    goal: str = typer.Argument(..., help="The user's goal for the conversation."),
    max_iters: int = typer.Option(8, help="Maximum generate/converse/critique iterations."),
    max_turns: int = typer.Option(5, help="Max assistant turns per conversation."),
    patience: int = typer.Option(3, help="Stop after this many iters without improvement."),
    tolerance: float = typer.Option(1.0, help="Min composite-score gain to count as progress."),
    model: str = typer.Option(None, help="Override the Claude model ID."),
    runs_dir: Path = typer.Option(None, help="Where to store per-iteration artifacts."),
) -> None:
    """Evolve an assistant's system prompt over a short conversation toward GOAL."""
    _load_env()
    runs_dir = runs_dir or _default_runs_dir()
    cfg = orchestrator.LoopConfig(
        brief=goal,
        runs_dir=runs_dir,
        db_path=runs_dir / "design-gan.sqlite",
        model=model or _default_model(),
        max_iters=max_iters,
        patience=patience,
        tolerance=tolerance,
        max_conversation_turns=max_turns,
    )
    result = orchestrator.run_conversation_loop_sync(cfg, console=console)
    console.rule("[bold green]Done")
    best_score_txt = f"{result.best_score:.1f}" if result.best_score is not None else "—"
    best_iter_txt = str(result.best_iter) if result.best_iter else "—"
    console.print(
        f"run_id={result.run_id}  best_iter={best_iter_txt}  "
        f"best_score={best_score_txt}  iters={result.iterations}  "
        f"status={result.status}"
    )


@app.command()
def export(
    run_id: int = typer.Argument(..., help="Run whose best iteration to export."),
    out: Path = typer.Option(
        None,
        help="Output path. Defaults to run_<id>_best.html (design) or .txt (conversation).",
    ),
    runs_dir: Path = typer.Option(None, help="Runs directory containing the sqlite db."),
) -> None:
    """Write the best iteration's artifact (site HTML or system prompt) to a file."""
    _load_env()
    runs_dir = runs_dir or _default_runs_dir()
    store = storage.Storage(runs_dir / "design-gan.sqlite")
    run = store.get_run(run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found.[/red]")
        raise typer.Exit(1)
    iters = store.iterations_for_run(run_id)
    if not iters:
        console.print(f"[red]Run {run_id} has no completed iterations.[/red]")
        raise typer.Exit(1)

    best_iter = run.get("best_iter")
    rec = next((x for x in iters if x["iter"] == best_iter), None)
    if rec is None:
        # No recorded best (e.g. still running) — fall back to the top composite.
        rec = max(iters, key=lambda x: x["composite_score"])

    kind = run.get("kind") or "design"
    if kind == "design" and run.get("holdout_passed") is False:
        console.print(
            "[yellow]Warning: this run failed its final untouched holdout audit.[/yellow]"
        )
    suffix = ".html" if kind == "design" else ".txt"
    out = out or Path(f"run_{run_id:04d}_best{suffix}")
    out.write_text(rec["html"], encoding="utf-8")
    console.print(
        f"[green]Exported[/green] run {run_id} iter {rec['iter']} "
        f"(primary score {rec['composite_score']:.1f}) -> {out}"
    )


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

    _load_env()
    runs_dir = runs_dir or _default_runs_dir()
    os.environ["DESIGN_GAN_RUNS_DIR"] = str(runs_dir)
    uvicorn.run("design_gan.viewer:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
