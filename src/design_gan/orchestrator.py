"""Main loop: generate -> render -> critique -> score; stop when score plateaus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import anthropic
from rich.console import Console

from . import critic, generator, renderer, scorer, storage


@dataclass
class LoopConfig:
    brief: str
    runs_dir: Path
    db_path: Path
    model: str = "claude-sonnet-4-6"
    max_iters: int = 15
    patience: int = 3  # stop after N iters without improvement > tolerance
    tolerance: float = 1.0  # point improvement below this counts as no progress
    viewport: tuple[int, int] = (1280, 800)


@dataclass
class LoopResult:
    run_id: int
    best_iter: int
    best_score: float
    iterations: int
    status: str  # "converged" | "exhausted"


async def run_loop(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    console = console or Console()
    client = anthropic.Anthropic()
    store = storage.Storage(cfg.db_path)
    if run_id is None:
        run_id = store.create_run(cfg.brief, cfg.model)
    run_dir = cfg.runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_iter = 0
    stale = 0
    prev_html: str | None = None
    prev_feedback: str | None = None
    prev_suggestions: list[str] | None = None

    status = "exhausted"
    i = 0

    for i in range(1, cfg.max_iters + 1):
        console.rule(f"[bold cyan]Run {run_id} iter {i}/{cfg.max_iters}")

        console.print("[dim]generating...[/dim]")
        html = generator.generate(
            client,
            cfg.model,
            generator.GenerationRequest(
                brief=cfg.brief,
                prior_html=prev_html,
                critic_feedback=prev_feedback,
                suggestions=prev_suggestions,
            ),
        )

        iter_dir = run_dir / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "site.html").write_text(html, encoding="utf-8")

        console.print("[dim]rendering...[/dim]")
        render = await renderer.render(html, viewport=cfg.viewport)
        renderer.write_artifacts(render, iter_dir)

        console.print("[dim]critiquing...[/dim]")
        sus = critic.critique(
            client,
            cfg.model,
            screenshot_png=render.screenshot_png,
            dom_html=render.dom_html,
            axe_violations=render.axe_violations,
            brief=cfg.brief,
        )

        result = scorer.score(list(sus.sus), render.axe_violations)
        store.save_iteration(
            storage.IterationRecord(
                run_id=run_id,
                iter=i,
                html=html,
                sus_score=result.sus,
                axe_penalty=result.axe_penalty,
                composite_score=result.composite,
                sus_answers=list(sus.sus),
                feedback=sus.feedback,
                suggestions=sus.suggestions,
                artifacts_dir=str(iter_dir),
            )
        )

        console.print(
            f"[bold]score[/bold]: SUS={result.sus:.1f}  "
            f"a11y_penalty={result.axe_penalty:.1f}  "
            f"[green]composite={result.composite:.1f}[/green]"
        )
        console.print(f"[dim]feedback:[/dim] {sus.feedback}")

        if result.composite > best_score + cfg.tolerance:
            best_score = result.composite
            best_iter = i
            stale = 0
        else:
            stale += 1

        if stale >= cfg.patience:
            status = "converged"
            console.print(
                f"[yellow]No improvement over {cfg.patience} iters — stopping.[/yellow]"
            )
            break

        prev_html = html
        prev_feedback = sus.feedback
        prev_suggestions = sus.suggestions

    store.finish_run(run_id, best_iter, best_score, status)
    return LoopResult(
        run_id=run_id,
        best_iter=best_iter,
        best_score=best_score,
        iterations=i,
        status=status,
    )


def run_loop_sync(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    return asyncio.run(run_loop(cfg, console, run_id=run_id))
