"""Main loop: generate -> render/evaluate -> critique -> score; stop on a plateau.

Design runs and conversation runs share one loop skeleton (`_run_shared_loop`)
and differ only in their per-iteration body:

- design:       generator.generate -> renderer.render -> critic.critique(*)
- conversation: conversation_generator.generate -> transcript_renderer.run_conversation
                -> critic.cus_critique(*)

The search is a greedy hill-climb: each iteration evolves from the best-scoring
iteration so far, not the latest one. When an iteration regresses, the next
generation is re-seeded from the best artifact and its critique, so the loop
never spends its remaining patience exploring from a bad ancestor.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from . import (
    browser_evaluator,
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
PHASE_EVALUATING = "evaluating"
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
    # Frozen once at run start and replayed unchanged against every design
    # candidate.  v2 begins with one concrete primary-action task.
    design_tasks: tuple[browser_evaluator.BrowserTask, ...] = browser_evaluator.DEFAULT_DESIGN_TASKS


@dataclass
class LoopResult:
    run_id: int
    best_iter: int
    best_score: float | None
    iterations: int
    status: str  # "converged" | "exhausted" | "errored" | "budget_exhausted"


@dataclass
class _IterState:
    """The artifacts fed forward into the next generation call."""

    artifact: str | None = None  # design: site HTML; conversation: system prompt
    feedback: str | None = None
    suggestions: list[str] | None = field(default=None)


@dataclass
class _IterOutcome:
    """What one completed iteration hands back to the shared loop."""

    artifact: str
    score: scorer.Score
    sus: critic.SUSResponse
    critic_breakdown: list[dict[str, Any]] | None
    cost_usd: float
    feedback: str | None = None
    suggestions: list[str] | None = None
    console_extra: str = ""  # appended to the score line, e.g. turns/satisfied


_IterateFn = Callable[
    [LoopConfig, int, "_IterState", Path, "storage.Storage", int, Console],
    Awaitable[_IterOutcome],
]


async def _design_iterate(
    cfg: LoopConfig,
    run_id: int,
    prev: _IterState,
    run_dir: Path,
    store: storage.Storage,
    i: int,
    console: Console,
) -> _IterOutcome:
    # --- generate -----------------------------------------------------------
    store.update_progress(run_id, i, PHASE_GENERATING)
    console.print("[dim]generating...[/dim]")
    html, gen_cost = await generator.generate(
        cfg.model,
        generator.GenerationRequest(
            brief=cfg.brief,
            prior_html=prev.artifact,
            critic_feedback=prev.feedback,
            suggestions=prev.suggestions,
        ),
    )
    cost = gen_cost

    iter_dir = run_dir / f"iter_{i:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "site.html").write_text(html, encoding="utf-8")

    # --- render ---------------------------------------------------------------
    store.update_progress(run_id, i, PHASE_RENDERING)
    console.print("[dim]rendering...[/dim]")
    render = await renderer.render(html, viewport=cfg.viewport)
    artifacts = renderer.write_artifacts(render, iter_dir)

    # --- behavioral evaluation ------------------------------------------------
    # Replay the exact same frozen task suite on a fresh page.  This completion
    # rate is the north-star signal for design search; it is not blended with
    # the critic's SUS opinion.
    store.update_progress(run_id, i, PHASE_EVALUATING)
    console.print(f"[dim]evaluating ({len(cfg.design_tasks)} frozen browser task(s))...[/dim]")
    evaluation = await browser_evaluator.evaluate(
        html,
        tasks=cfg.design_tasks,
        viewport=cfg.viewport,
    )
    browser_evaluator.write_artifact(evaluation, iter_dir)

    # --- critique ---------------------------------------------------------------
    store.update_progress(run_id, i, PHASE_CRITIQUING)
    critic_breakdown: list[dict[str, Any]] | None = None
    if cfg.critics:
        console.print(f"[dim]critiquing (ensemble of {len(cfg.critics)})...[/dim]")
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
    cost += crit_cost

    result = scorer.design_score(
        list(sus.sus),
        render.axe_violations,
        task_score=evaluation.score,
        task_results=[task.to_dict() for task in evaluation.tasks],
        axe_error=render.axe_error,
        console_errors=render.console_errors,
        evaluator_errors=evaluation.correctness_errors,
    )
    optimization_feedback = (
        f"{evaluation.feedback()}\n\n"
        f"Diagnostic SUS feedback (not the primary score): {sus.feedback}"
    )
    optimization_suggestions = list(sus.suggestions)
    if evaluation.passed < evaluation.total:
        optimization_suggestions.insert(
            0,
            "Make the page's primary call to action obvious, enabled, and behaviorally "
            "functional: activating it must navigate, scroll, open a dialog/window, or "
            "change visible content without runtime errors.",
        )
    return _IterOutcome(
        artifact=html,
        score=result,
        sus=sus,
        critic_breakdown=critic_breakdown,
        cost_usd=cost,
        feedback=optimization_feedback,
        suggestions=optimization_suggestions,
        console_extra=f"  promotable={result.promotion_eligible}",
    )


async def _conversation_iterate(
    cfg: LoopConfig,
    run_id: int,
    prev: _IterState,
    run_dir: Path,
    store: storage.Storage,
    i: int,
    console: Console,
) -> _IterOutcome:
    # --- generate assistant system prompt ------------------------------------
    store.update_progress(run_id, i, PHASE_GENERATING)
    console.print("[dim]generating system prompt...[/dim]")
    prompt, gen_cost = await conversation_generator.generate(
        cfg.model,
        conversation_generator.ConversationGenerationRequest(
            goal=cfg.brief,
            max_turns=cfg.max_conversation_turns,
            prior_system_prompt=prev.artifact,
            critic_feedback=prev.feedback,
            suggestions=prev.suggestions,
        ),
    )
    cost = gen_cost

    iter_dir = run_dir / f"iter_{i:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # --- run conversation -----------------------------------------------------
    store.update_progress(run_id, i, PHASE_CONVERSING)
    console.print("[dim]conversing...[/dim]")
    trans = await transcript_renderer.run_conversation(
        model=cfg.model,
        assistant_system_prompt=prompt,
        goal=cfg.brief,
        max_turns=cfg.max_conversation_turns,
    )
    cost += trans.total_cost_usd
    transcript_renderer.write_transcript_artifacts(trans, iter_dir)

    # --- critique ---------------------------------------------------------------
    store.update_progress(run_id, i, PHASE_CRITIQUING)
    critic_breakdown: list[dict[str, Any]] | None = None
    if cfg.critics:
        console.print(f"[dim]CUS critique (ensemble of {len(cfg.critics)})...[/dim]")
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
    cost += crit_cost

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
    return _IterOutcome(
        artifact=prompt,  # persisted in the html column (the evolving artifact)
        score=result,
        sus=sus,
        critic_breakdown=critic_breakdown,
        cost_usd=cost,
        console_extra=f"  turns={trans.turns_taken}  satisfied={trans.satisfied}",
    )


async def _run_shared_loop(
    cfg: LoopConfig,
    console: Console | None,
    run_id: int | None,
    *,
    kind: str,
    rule_prefix: str,
    score_label: str,
    penalty_label: str,
    feedback_limit: int | None,
    iterate: _IterateFn,
) -> LoopResult:
    console = console or Console()
    store = storage.Storage(cfg.db_path)
    if run_id is None:
        suite = (
            [task.to_dict() for task in browser_evaluator.frozen_suite(cfg.design_tasks)]
            if kind == KIND_DESIGN
            else None
        )
        run_id = store.create_run(
            cfg.brief,
            cfg.model,
            kind=kind,
            evaluation_suite=suite,
        )
    run_dir = cfg.runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_score: float | None = None
    best_iter = 0
    stale = 0
    prev = _IterState()
    best = _IterState()  # artifacts + critique of the best-scoring iteration

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

            console.rule(f"[bold cyan]{rule_prefix} {run_id} iter {i}/{cfg.max_iters}")

            try:
                out = await iterate(cfg, run_id, prev, run_dir, store, i, console)
                store.save_iteration(
                    storage.IterationRecord(
                        run_id=run_id,
                        iter=i,
                        html=out.artifact,
                        sus_score=out.score.sus,
                        axe_penalty=out.score.axe_penalty,
                        composite_score=out.score.composite,
                        sus_answers=list(out.sus.sus),
                        feedback=out.feedback or out.sus.feedback,
                        suggestions=out.suggestions or out.sus.suggestions,
                        artifacts_dir=str(run_dir / f"iter_{i:03d}"),
                        cost_usd=out.cost_usd,
                        critic_breakdown=out.critic_breakdown,
                        primary_score=out.score.composite,
                        primary_metric=out.score.primary_metric,
                        promotion_eligible=out.score.promotion_eligible,
                        guardrails=out.score.guardrails,
                        task_results=out.score.breakdown.get("task_results"),
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
                f"[bold]score[/bold]: {score_label}={out.score.composite:.1f}  "
                f"diagnostic={out.score.sus:.1f}  "
                f"{penalty_label}={out.score.axe_penalty:.1f}  "
                f"[dim]cost=${out.cost_usd:.3f}{out.console_extra}[/dim]"
            )
            next_feedback = out.feedback or out.sus.feedback
            feedback_txt = next_feedback[:feedback_limit] if feedback_limit else next_feedback
            console.print(f"[dim]feedback:[/dim] {feedback_txt}")

            current = _IterState(
                artifact=out.artifact,
                feedback=next_feedback,
                suggestions=out.suggestions or out.sus.suggestions,
            )
            if out.score.promotion_eligible and (
                best_score is None or out.score.composite > best_score + cfg.tolerance
            ):
                best_score = out.score.composite
                best_iter = i
                stale = 0
                best = current
                prev = current
            else:
                stale += 1
                # Regression or plateau: re-seed the next generation from the
                # best iteration so the search never drifts downhill from a
                # bad ancestor.
                # Before any candidate clears the guardrails, keep evolving
                # from the latest artifact so its concrete failures can be
                # repaired.  Once a promotable best exists, regressions and
                # blocked candidates re-seed from that safe ancestor.
                prev = best if best_score is not None else current

            if stale >= cfg.patience:
                status = "converged"
                console.print(
                    f"[yellow]No improvement over {cfg.patience} iters — stopping.[/yellow]"
                )
                break
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


async def run_loop(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    return await _run_shared_loop(
        cfg,
        console,
        run_id,
        kind=KIND_DESIGN,
        rule_prefix="Run",
        score_label="tasks",
        penalty_label="a11y_penalty",
        feedback_limit=None,
        iterate=_design_iterate,
    )


def run_loop_sync(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    return asyncio.run(run_loop(cfg, console, run_id=run_id))


async def run_conversation_loop(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    """Autoresearch loop over conversations instead of pixels.

    Same skeleton as run_loop with the conversation iteration body:
    - conversation_generator.generate evolves a system prompt
    - transcript_renderer.run_conversation "renders" it as a dialogue
    - critic.cus_critique(*) scores the transcript on the CUS
    """
    return await _run_shared_loop(
        cfg,
        console,
        run_id,
        kind=KIND_CONVERSATION,
        rule_prefix="Conversation run",
        score_label="CUS composite",
        penalty_label="objective_penalty",
        feedback_limit=200,
        iterate=_conversation_iterate,
    )


def run_conversation_loop_sync(
    cfg: LoopConfig, console: Console | None = None, run_id: int | None = None
) -> LoopResult:
    return asyncio.run(run_conversation_loop(cfg, console, run_id=run_id))
