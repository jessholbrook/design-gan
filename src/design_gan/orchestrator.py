"""Main loop: generate -> render -> critique -> score; stop when score plateaus."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from . import (
    conversation_generator,
    critic,
    generator,
    renderer,
    scorer,
    storage,
    transcript_renderer,
)

# Phases reported via storage.update_progress so the viewer can display
# which stage of which iteration is in flight.
PHASE_GENERATING = "generating"
PHASE_RENDERING = "rendering"
PHASE_CRITIQUING = "critiquing"
# Conversation runs replace the renderer phase with a multi-turn dialogue.
PHASE_CONVERSING = "conversing"

KIND_DESIGN = "design"
KIND_CONVERSATION = "conversation"


@dataclass
class LoopConfig:
    brief: str  # design runs: the site brief. conversation runs: the user goal.
    runs_dir: Path
    db_path: Path
    model: str = "claude-sonnet-4-6"
    max_iters: int = 15
    patience: int = 3  # stop after N iters without improvement > tolerance
    tolerance: float = 1.0  # point improvement below this counts as no progress
    viewport: tuple[int, int] = (1280, 800)
    # Hard stop when cumulative iteration cost over the last 24h crosses this.
    # Checked before each iteration. None disables the check (local / CLI use).
    daily_budget_usd: float | None = None
    # When set, each iteration runs this list of critics in parallel and
    # aggregates their scores. None = single Usability critic (backward compat).
    critics: list[critic.CriticProfile] | None = None
    # Conversation-run specific: max dialogue turns per iteration (assistant
    # turns). 1-5 feels right; 5 is usually enough to surface resolution
    # without exploding cost.
    max_conversation_turns: int = 5


@dataclass
class LoopResult:
    run_id: int
    best_iter: int
    best_score: float | None
    iterations: int
    status: str  # "converged" | "exhausted" | "errored" | "budget_exhausted"


async def run_loop(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    console = console or Console()
    store = storage.Storage(cfg.db_path)
    if run_id is None:
        run_id = store.create_run(cfg.brief, cfg.model)
    run_dir = cfg.runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_score: float | None = None
    best_iter = 0
    stale = 0
    prev_html: str | None = None
    prev_feedback: str | None = None
    prev_suggestions: list[str] | None = None

    status = "exhausted"
    final_error: str | None = None
    i = 0

    try:
        for i in range(1, cfg.max_iters + 1):
            # Budget gate: consult DB before each iteration so a mid-run cost
            # spike still trips the circuit. A single in-flight iteration can
            # overshoot by at most its own cost.
            if cfg.daily_budget_usd is not None:
                used = store.cost_usd_last_24h()
                if used >= cfg.daily_budget_usd:
                    status = "budget_exhausted"
                    final_error = (
                        f"daily budget exhausted before iter {i}: "
                        f"${used:.2f} used of ${cfg.daily_budget_usd:.2f}"
                    )
                    console.print(f"[red]{final_error}[/red]")
                    break

            console.rule(f"[bold cyan]Run {run_id} iter {i}/{cfg.max_iters}")

            iter_cost_usd = 0.0
            try:
                # --- generate -----------------------------------------------
                store.update_progress(run_id, i, PHASE_GENERATING)
                console.print("[dim]generating...[/dim]")
                html, gen_cost = await generator.generate(
                    cfg.model,
                    generator.GenerationRequest(
                        brief=cfg.brief,
                        prior_html=prev_html,
                        critic_feedback=prev_feedback,
                        suggestions=prev_suggestions,
                    ),
                )
                iter_cost_usd += gen_cost

                iter_dir = run_dir / f"iter_{i:03d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                (iter_dir / "site.html").write_text(html, encoding="utf-8")

                # --- render -------------------------------------------------
                store.update_progress(run_id, i, PHASE_RENDERING)
                console.print("[dim]rendering...[/dim]")
                render = await renderer.render(html, viewport=cfg.viewport)
                artifacts = renderer.write_artifacts(render, iter_dir)

                # --- critique ----------------------------------------------
                store.update_progress(run_id, i, PHASE_CRITIQUING)
                critic_breakdown: list[dict] | None = None
                if cfg.critics:
                    console.print(
                        f"[dim]critiquing (ensemble of {len(cfg.critics)})...[/dim]"
                    )
                    sus, critic_breakdown, crit_cost = await critic.critique_ensemble(
                        cfg.model,
                        cfg.critics,
                        screenshot_path=artifacts["screenshot"].resolve(),
                        dom_html=render.dom_html,
                        axe_violations=render.axe_violations,
                        brief=cfg.brief,
                    )
                else:
                    console.print("[dim]critiquing...[/dim]")
                    sus, crit_cost = await critic.critique(
                        cfg.model,
                        screenshot_path=artifacts["screenshot"].resolve(),
                        dom_html=render.dom_html,
                        axe_violations=render.axe_violations,
                        brief=cfg.brief,
                    )
                iter_cost_usd += crit_cost

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
                        cost_usd=iter_cost_usd,
                        critic_breakdown=critic_breakdown,
                    )
                )
            except Exception as e:
                # A single bad iteration shouldn't kill the whole run. Log it,
                # count it as "no progress", and let the patience rule decide.
                console.print(f"[red]iter {i} failed: {e}[/red]")
                console.print(traceback.format_exc())
                stale += 1
                if stale >= cfg.patience:
                    status = "errored"
                    final_error = f"iter {i}: {e}"
                    break
                continue

            console.print(
                f"[bold]score[/bold]: SUS={result.sus:.1f}  "
                f"a11y_penalty={result.axe_penalty:.1f}  "
                f"[green]composite={result.composite:.1f}[/green]  "
                f"[dim]cost=${iter_cost_usd:.3f}[/dim]"
            )
            console.print(f"[dim]feedback:[/dim] {sus.feedback}")

            if best_score is None or result.composite > best_score + cfg.tolerance:
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
    except Exception as e:
        # Truly unexpected failure — still mark the run so it doesn't hang.
        status = "errored"
        final_error = str(e)
        console.print(f"[red]run errored: {e}[/red]")
        console.print(traceback.format_exc())
    finally:
        # If no iteration ever completed, persist nulls rather than a sentinel
        # 0/0.0 that the UI would otherwise render as a real score.
        final_best_iter = best_iter if best_iter > 0 else None
        store.finish_run(run_id, final_best_iter, best_score, status, error=final_error)

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


async def run_conversation_loop(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    """Autoresearch loop over conversations instead of pixels.

    Structurally parallel to run_loop but swaps:
    - generator.generate -> conversation_generator.generate (evolves a system prompt)
    - renderer.render    -> transcript_renderer.run_conversation
    - critic.critique(*)  -> critic.cus_critique(*)  (CUS items, transcript input)
    - scorer.score(answers, violations) -> scorer.score_from_penalty(answers, penalty)
    """
    console = console or Console()
    store = storage.Storage(cfg.db_path)
    if run_id is None:
        run_id = store.create_run(cfg.brief, cfg.model, kind=KIND_CONVERSATION)
    run_dir = cfg.runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_score: float | None = None
    best_iter = 0
    stale = 0
    prev_prompt: str | None = None
    prev_feedback: str | None = None
    prev_suggestions: list[str] | None = None

    status = "exhausted"
    final_error: str | None = None
    i = 0

    try:
        for i in range(1, cfg.max_iters + 1):
            if cfg.daily_budget_usd is not None:
                used = store.cost_usd_last_24h()
                if used >= cfg.daily_budget_usd:
                    status = "budget_exhausted"
                    final_error = (
                        f"daily budget exhausted before iter {i}: "
                        f"${used:.2f} used of ${cfg.daily_budget_usd:.2f}"
                    )
                    console.print(f"[red]{final_error}[/red]")
                    break

            console.rule(
                f"[bold cyan]Conversation run {run_id} iter {i}/{cfg.max_iters}"
            )

            iter_cost_usd = 0.0
            try:
                # --- generate assistant system prompt -----------------------
                store.update_progress(run_id, i, PHASE_GENERATING)
                console.print("[dim]generating system prompt...[/dim]")
                prompt, gen_cost = await conversation_generator.generate(
                    cfg.model,
                    conversation_generator.ConversationGenerationRequest(
                        goal=cfg.brief,
                        max_turns=cfg.max_conversation_turns,
                        prior_system_prompt=prev_prompt,
                        critic_feedback=prev_feedback,
                        suggestions=prev_suggestions,
                    ),
                )
                iter_cost_usd += gen_cost

                iter_dir = run_dir / f"iter_{i:03d}"
                iter_dir.mkdir(parents=True, exist_ok=True)

                # --- run conversation ---------------------------------------
                store.update_progress(run_id, i, PHASE_CONVERSING)
                console.print("[dim]conversing...[/dim]")
                trans = await transcript_renderer.run_conversation(
                    model=cfg.model,
                    assistant_system_prompt=prompt,
                    goal=cfg.brief,
                    max_turns=cfg.max_conversation_turns,
                )
                iter_cost_usd += trans.total_cost_usd
                artifacts = transcript_renderer.write_transcript_artifacts(
                    trans, iter_dir
                )

                # --- critique -----------------------------------------------
                store.update_progress(run_id, i, PHASE_CRITIQUING)
                critic_breakdown: list[dict] | None = None
                if cfg.critics:
                    console.print(
                        f"[dim]CUS critique (ensemble of {len(cfg.critics)})...[/dim]"
                    )
                    sus, critic_breakdown, crit_cost = await critic.cus_critique_ensemble(
                        cfg.model,
                        cfg.critics,
                        goal=cfg.brief,
                        transcript=trans.transcript,
                        objective_metrics=trans.objective_metrics,
                        assistant_system_prompt=prompt,
                    )
                else:
                    console.print("[dim]CUS critique...[/dim]")
                    sus, crit_cost = await critic.cus_critique(
                        cfg.model,
                        goal=cfg.brief,
                        transcript=trans.transcript,
                        objective_metrics=trans.objective_metrics,
                        assistant_system_prompt=prompt,
                    )
                iter_cost_usd += crit_cost

                result = scorer.score_from_penalty(
                    list(sus.sus),
                    trans.objective_penalty,
                    breakdown={
                        "kind": "conversation",
                        "objective_metrics": trans.objective_metrics,
                        "turns_taken": trans.turns_taken,
                        "satisfied": trans.satisfied,
                    },
                )
                # Persist: html column stores the assistant's system prompt.
                store.save_iteration(
                    storage.IterationRecord(
                        run_id=run_id,
                        iter=i,
                        html=prompt,
                        sus_score=result.sus,
                        axe_penalty=result.axe_penalty,
                        composite_score=result.composite,
                        sus_answers=list(sus.sus),
                        feedback=sus.feedback,
                        suggestions=sus.suggestions,
                        artifacts_dir=str(iter_dir),
                        cost_usd=iter_cost_usd,
                        critic_breakdown=critic_breakdown,
                    )
                )
            except Exception as e:
                console.print(f"[red]iter {i} failed: {e}[/red]")
                console.print(traceback.format_exc())
                stale += 1
                if stale >= cfg.patience:
                    status = "errored"
                    final_error = f"iter {i}: {e}"
                    break
                continue

            console.print(
                f"[bold]score[/bold]: CUS={result.sus:.1f}  "
                f"objective_penalty={result.axe_penalty:.1f}  "
                f"[green]composite={result.composite:.1f}[/green]  "
                f"[dim]cost=${iter_cost_usd:.3f}  "
                f"turns={trans.turns_taken}  satisfied={trans.satisfied}[/dim]"
            )
            console.print(f"[dim]feedback:[/dim] {sus.feedback[:200]}")

            if best_score is None or result.composite > best_score + cfg.tolerance:
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

            prev_prompt = prompt
            prev_feedback = sus.feedback
            prev_suggestions = sus.suggestions
    except Exception as e:
        status = "errored"
        final_error = str(e)
        console.print(f"[red]run errored: {e}[/red]")
        console.print(traceback.format_exc())
    finally:
        final_best_iter = best_iter if best_iter > 0 else None
        store.finish_run(run_id, final_best_iter, best_score, status, error=final_error)

    return LoopResult(
        run_id=run_id,
        best_iter=best_iter,
        best_score=best_score,
        iterations=i,
        status=status,
    )


def run_conversation_loop_sync(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    return asyncio.run(run_conversation_loop(cfg, console, run_id=run_id))
