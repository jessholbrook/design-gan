"""FastAPI viewer: dashboard of runs, per-run detail with live SSE updates."""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import (
    artifact_policy,
    evaluator_benchmark,
    incumbent_ledger,
    product_domains,
    storage,
)


def _runs_dir() -> Path:
    return Path(os.environ.get("DESIGN_GAN_RUNS_DIR", "./runs"))


def _default_model() -> str:
    return os.environ.get("DESIGN_GAN_MODEL", "claude-sonnet-4-6")


def _required_start_token() -> str | None:
    """Shared token for run starts and evaluator-label writes; unset disables the gate."""
    tok = os.environ.get("DESIGN_GAN_START_TOKEN")
    return tok if tok else None


def _daily_budget_usd() -> float | None:
    """Daily (rolling 24h) spending cap. Unset or <= 0 disables the gate."""
    raw = os.environ.get("DESIGN_GAN_DAILY_BUDGET_USD")
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


# Runs with no heartbeat for this long are presumed dead (machine restart,
# OOM kill, etc.). Sweep clears them at boot and could be called periodically.
ABANDONED_RUN_TIMEOUT_SECONDS = int(os.environ.get("DESIGN_GAN_ABANDONED_TIMEOUT_S", "600"))


def _configured_critics() -> list[Any] | None:
    """DESIGN_GAN_CRITICS=trio opts into the 3-critic ensemble.

    Unset or 'solo' keeps the single Usability critic (backward compat).
    Runs triggered while this env var is set will use the ensemble.
    """
    mode = (os.environ.get("DESIGN_GAN_CRITICS") or "").strip().lower()
    if mode == "trio":
        from . import critic

        return list(critic.TRIO)
    return None


@lru_cache(maxsize=8)
def _storage_for(db_path: str) -> storage.Storage:
    """One Storage per db path. Storage.__init__ runs the schema script and
    migrations, so constructing it per request (or per SSE poll) is wasteful."""
    return storage.Storage(db_path)


def _store() -> storage.Storage:
    return _storage_for(str(_runs_dir() / "design-gan.sqlite"))


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Boot-time cleanup: any run still marked 'running' is a ghost from a
    prior machine (restart, OOM, etc.). Mark them errored so the UI doesn't
    show a spinner that never resolves."""
    log = logging.getLogger(__name__)
    try:
        swept = _store().sweep_abandoned_runs(0)
        if swept:
            log.info("swept %d abandoned run(s) on boot: %s", len(swept), swept)
    except Exception:
        log.exception("startup sweep failed")
    yield


app = FastAPI(title="design-gan viewer", lifespan=_lifespan)


# ---------- HTML helpers ----------

_STATIC_DIR = Path(__file__).parent / "static"


def _layout(title: str, body: str, body_attrs: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body{body_attrs}>
  <header class="topbar">
    <a href="/" class="brand">design-gan</a>
    <span class="muted">Autoresearch dual-agent loop</span>
    <nav class="topnav"><a href="/evaluator-review">Evaluator review</a></nav>
  </header>
  {body}
  <script src="/static/app.js"></script>
</body>
</html>"""


def _score_class(score: float | None) -> str:
    if score is None:
        return "score-none"
    if score >= 80:
        return "score-good"
    if score >= 60:
        return "score-ok"
    return "score-bad"


def _status_badge(status: str) -> str:
    cls = {
        "running": "running",
        "converged": "converged",
        "exhausted": "exhausted",
        "errored": "errored",
        "budget_exhausted": "errored",
    }.get(status, "unknown")
    if status == "running":
        return (
            '<span class="status status-running"><span class="status-beacon" '
            'aria-hidden="true"></span><span>running</span></span>'
        )
    return f'<span class="status status-{cls}">{html.escape(status)}</span>'


_PROGRESS_PHASES = (
    ("generating", "Generating candidate"),
    ("rendering", "Rendering in browser"),
    ("evaluating", "Evaluating frozen tasks"),
    ("critiquing", "Critiquing the result"),
)


def _progress_state(
    current_iter: int | None, current_phase: str | None, max_iters: int | None
) -> dict[str, Any]:
    """Translate persisted loop progress into an honest display estimate."""
    phase_ids = [phase for phase, _ in _PROGRESS_PHASES]
    if not current_iter or current_phase not in phase_ids:
        max_text = f" of {max_iters}" if max_iters else ""
        return {
            "title": "Preparing run",
            "meta": f"Waiting for iteration 1{max_text}",
            "percent": 0,
            "phase_index": -1,
        }
    phase_index = phase_ids.index(current_phase)
    if max_iters:
        completed = max(0, current_iter - 1) + (phase_index + 1) / len(_PROGRESS_PHASES)
        percent = min(99, max(1, round(completed / max_iters * 100)))
        iteration_text = f"Iteration {current_iter} of {max_iters}"
    else:
        percent = 0
        iteration_text = f"Iteration {current_iter}"
    return {
        "title": _PROGRESS_PHASES[phase_index][1],
        "meta": f"{iteration_text} · stage {phase_index + 1} of {len(_PROGRESS_PHASES)}",
        "percent": percent,
        "phase_index": phase_index,
    }


def _progress_html(
    *, running: bool, current_iter: int | None, current_phase: str | None, max_iters: int | None
) -> str:
    state = _progress_state(current_iter, current_phase, max_iters)
    steps = []
    for index, (phase, label) in enumerate(_PROGRESS_PHASES):
        step_class = " is-active" if index == state["phase_index"] else ""
        if 0 <= index < state["phase_index"]:
            step_class = " is-complete"
        steps.append(f'<span class="progress-step{step_class}" data-phase="{phase}">{label}</span>')
    display = "grid" if running else "none"
    max_iters_value = max_iters or 0
    percent = state["percent"]
    aria_text = f"{state['meta']}: {state['title']}"
    return f"""<section id="progress-indicator" class="run-progress"
        style="display:{display}" data-max-iters="{max_iters_value}" aria-live="polite">
      <div class="progress-summary">
        <span class="progress-activity" aria-hidden="true"><i></i><i></i><i></i></span>
        <span class="progress-copy"><strong id="progress-title">{state["title"]}</strong>
          <span id="progress-text">{state["meta"]}</span></span>
        <b id="progress-percent">{percent}%</b>
      </div>
      <div id="progress-track" class="progress-track" role="progressbar"
        aria-label="Run progress" aria-valuemin="0" aria-valuemax="100"
        aria-valuenow="{percent}" aria-valuetext="{html.escape(aria_text)}">
        <span id="progress-bar" style="width:{percent}%"></span>
      </div>
      <div id="progress-steps" class="progress-steps">{"".join(steps)}</div>
    </section>"""


def _evaluator_cases() -> tuple[evaluator_benchmark.BenchmarkCase, ...]:
    try:
        return evaluator_benchmark.load_case_directory(_runs_dir() / "evaluator-corpus")
    except ValueError as exc:
        raise HTTPException(500, f"invalid evaluator corpus: {exc}") from exc


def _review_candidates(
    *, mode: str = "balanced", limit: int = 100
) -> tuple[evaluator_benchmark.ReviewCandidate, ...]:
    if mode == "balanced":
        return evaluator_benchmark.balanced_review_candidates(
            _store(), captured_cases=_evaluator_cases(), limit=limit
        )
    if mode not in {"failures", "all"}:
        raise HTTPException(422, "mode must be balanced, failures, or all")
    return evaluator_benchmark.review_candidates(
        _store(),
        captured_cases=_evaluator_cases(),
        failed_only=mode == "failures",
        limit=limit,
    )


def _corpus_readiness() -> evaluator_benchmark.CorpusReadinessReport:
    return evaluator_benchmark.audit_provenance_corpus(_evaluator_cases())


def _runs_sidebar(active_id: int | None) -> str:
    rows = _store().list_runs()
    items = []
    for r in rows:
        active = " active" if active_id == r["id"] else ""
        brief = (r["brief"] or "")[:60]
        score_txt = f"{r['best_score']:.0f}" if r["best_score"] is not None else "—"
        items.append(
            f"""<a href="/runs/{r["id"]}" class="side-item{active}">
              <span class="side-id">#{r["id"]}</span>
              <span class="side-brief">{html.escape(brief)}</span>
              <span class="side-score {_score_class(r["best_score"])}">{score_txt}</span>
            </a>"""
        )
    if not items:
        items.append('<div class="side-empty muted">No runs yet.</div>')
    return f'<aside class="sidebar"><h3>Runs</h3>{"".join(items)}</aside>'


def _field_term(
    field_name: str,
    label: str,
    explanation: str,
    *,
    term_name: str | None = None,
) -> str:
    """Render one form term with hover and keyboard-focus help."""
    term_attr = f' data-form-term="{html.escape(term_name, quote=True)}"' if term_name else ""
    escaped_field_name = html.escape(field_name, quote=True)
    escaped_label = html.escape(label)
    escaped_explanation = html.escape(explanation, quote=True)
    return (
        '<span class="field-term">'
        f'<span id="{escaped_field_name}-label"{term_attr}>{escaped_label}</span>'
        f'<span id="{escaped_field_name}-help" class="info-tip" tabindex="0" role="note" '
        f'aria-label="Help: {escaped_explanation}" '
        f'data-tooltip="{escaped_explanation}"><span class="sr-only">'
        f"{escaped_explanation}</span></span></span>"
    )


def _new_run_form() -> str:
    gated = _required_start_token() is not None
    gated_attr = ' data-requires-token="1"' if gated else ""
    access_token_term = _field_term(
        "token",
        "Access token",
        "Shared write token required to start runs on a gated deployment. Browsing existing "
        "results does not require it.",
    )
    token_field = (
        f'<label class="token-field">{access_token_term}'
        '<input type="password" name="token" autocomplete="off" '
        'aria-labelledby="token-label" aria-describedby="token-help" '
        'placeholder="Required to start a run on this deployment" /></label>'
        if gated
        else ""
    )
    gated_note = (
        '<p class="muted gated-note">Starting a run requires a shared token '
        "on this deployment — ask the owner. Browsing existing runs is open.</p>"
        if gated
        else ""
    )
    kind_term = _field_term(
        "kind",
        "Kind",
        "Choose whether this run optimizes a single-page website or an assistant conversation.",
    )
    brief_term = _field_term(
        "brief",
        "Brief",
        "For a design run, describe the page to build. For a conversation run, state the "
        "assistant goal or user request.",
        term_name="brief",
    )
    max_iterations_term = _field_term(
        "max-iters",
        "Max iterations",
        "Maximum number of candidate generations before the run stops.",
    )
    patience_term = _field_term(
        "patience",
        "Patience",
        "Stop after this many consecutive candidates fail to produce an eligible improvement.",
    )
    tolerance_term = _field_term(
        "tolerance",
        "Tolerance",
        "Minimum task-score improvement, in percentage points, required to count as progress "
        "and reset patience.",
    )
    model_term = _field_term(
        "model",
        "Model",
        "Claude model used for generation and critique. Keep the default unless comparing models.",
    )
    domain_term = _field_term(
        "design-domain",
        "Product domain",
        "Selects the versioned frozen browser-task suite used as the primary product-quality "
        "measure.",
    )
    trials_term = _field_term(
        "evaluation-trials",
        "Trials per task",
        "Repeated browser attempts for each frozen task. More trials take longer but provide "
        "more stable promotion evidence.",
    )
    alpha_term = _field_term(
        "promotion-alpha",
        "Promotion alpha",
        "Maximum one-sided sign-test p-value allowed for promotion. Lower values require "
        "stronger evidence that the candidate improved.",
    )
    optimization_key_term = _field_term(
        "optimization-key",
        "Optimization key (optional)",
        "Stable identity shared by runs optimizing the same product. Leave blank to derive it "
        "from the brief.",
    )
    conversation_turns_term = _field_term(
        "conversation-turns",
        "Max conversation turns",
        "Maximum number of user-assistant exchanges evaluated in a conversation run.",
    )
    return f"""<section class="card new-run">
  <h2>Start a new run</h2>
  {gated_note}
  <form id="new-run-form"{gated_attr}>
    <label>{kind_term}
      <select name="kind" aria-labelledby="kind-label" aria-describedby="kind-help">
        <option value="design">Design — evolve a single-page website</option>
        <option value="conversation">Conversation — evolve an assistant for a 1–5 turn chat</option>
      </select>
    </label>
    <label data-brief-label>{brief_term}
      <textarea name="brief" rows="3" required aria-labelledby="brief-label"
        aria-describedby="brief-help"
        placeholder="A landing page for a weekend cycling tour in rural Vermont."></textarea>
    </label>
    <div class="row">
      <label>{max_iterations_term}<input type="number" name="max_iters" value="15" min="1" max="50" aria-labelledby="max-iters-label" aria-describedby="max-iters-help" /></label>
      <label>{patience_term}<input type="number" name="patience" value="3" min="1" max="10" aria-labelledby="patience-label" aria-describedby="patience-help" /></label>
      <label>{tolerance_term}<input type="number" name="tolerance" value="1.0" step="0.5" min="0" aria-labelledby="tolerance-label" aria-describedby="tolerance-help" /></label>
      <label>{model_term}<input type="text" name="model" value="{html.escape(_default_model())}" aria-labelledby="model-label" aria-describedby="model-help" /></label>
    </div>
    <div class="row" data-design-only>
      <label>{domain_term}
        <select name="design_domain" aria-labelledby="design-domain-label" aria-describedby="design-domain-help">
          <option value="landing-page">Landing page — primary action</option>
          <option value="lead-generation">Lead generation — complete form</option>
          <option value="storefront">Storefront — add product to cart</option>
        </select>
      </label>
      <label>{trials_term}
        <input type="number" name="evaluation_trials" value="{product_domains.DEFAULT_EVALUATION_TRIALS}" min="1" max="50" aria-labelledby="evaluation-trials-label" aria-describedby="evaluation-trials-help" />
      </label>
      <label>{alpha_term}
        <input type="number" name="promotion_alpha" value="{product_domains.DEFAULT_PROMOTION_ALPHA}" step="0.0001" min="0.0001" max="1" aria-labelledby="promotion-alpha-label" aria-describedby="promotion-alpha-help" />
      </label>
    </div>
    <label data-design-only>{optimization_key_term}
      <input type="text" name="optimization_key" maxlength="160"
        aria-labelledby="optimization-key-label" aria-describedby="optimization-key-help"
        placeholder="Shared key for runs optimizing the same product" />
    </label>
    <label data-conversation-only hidden>{conversation_turns_term}
      <input type="number" name="max_conversation_turns" value="5" min="1" max="10"
        aria-labelledby="conversation-turns-label" aria-describedby="conversation-turns-help" />
    </label>
    {token_field}
    <button type="submit">Run</button>
    <span id="new-run-status" class="muted"></span>
  </form>
</section>"""


def _transcript_preview_html(run_id: int, it: int) -> str:
    """Compact preview of the first user+assistant turns for the card thumb."""
    path = _runs_dir() / f"run_{run_id:04d}" / f"iter_{it:03d}" / "transcript.json"
    if not path.exists():
        return '<div class="thumb-empty muted">no transcript</div>'
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        turns = data.get("transcript", [])[:2]
    except Exception:
        return '<div class="thumb-empty muted">transcript unreadable</div>'
    bubbles = []
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "")
        if len(content) > 160:
            content = content[:160].rstrip() + "…"
        bubbles.append(
            f'<div class="bubble bubble-{role}">'
            f'<span class="bubble-role">{html.escape(role)}</span>'
            f'<span class="bubble-text">{html.escape(content)}</span>'
            f"</div>"
        )
    return '<div class="transcript-preview">' + "".join(bubbles) + "</div>"


def _iter_card_html(run_id: int, it: dict, kind: str = "design") -> str:
    suggestions = "".join(f"<li>{html.escape(s)}</li>" for s in (it.get("suggestions") or []))
    if kind == "conversation":
        thumb = (
            f'<a href="/runs/{run_id}/iters/{it["iter"]}/transcript-view" '
            f'target="_blank" class="thumb thumb-transcript">'
            f"{_transcript_preview_html(run_id, it['iter'])}"
            f"</a>"
        )
        stats = (
            f"<span>CUS <b>{it['sus_score']:.0f}</b></span>"
            f"<span>penalty <b>{it['axe_penalty']:.0f}</b></span>"
        )
    else:
        thumb = (
            f'<a href="/runs/{run_id}/iters/{it["iter"]}/site" target="_blank" '
            f'class="thumb">'
            f'<img src="/runs/{run_id}/iters/{it["iter"]}/screenshot" '
            f'loading="lazy" alt="Iter {it["iter"]}" />'
            f"</a>"
        )
        if it.get("primary_metric") == "task_completion_rate":
            gate = "eligible" if it.get("promotion_eligible") else "blocked"
            decision = "promoted" if it.get("promoted") else "rejected"
            stats = (
                f"<span>tasks <b>{(it.get('primary_score') or 0):.0f}</b></span>"
                f"<span>SUS diagnostic <b>{it['sus_score']:.0f}</b></span>"
                f"<span>guardrails <b>{gate}</b></span>"
                f"<span>decision <b>{decision}</b></span>"
            )
        else:
            stats = (
                f"<span>SUS <b>{it['sus_score']:.0f}</b></span>"
                f"<span>a11y penalty <b>{it['axe_penalty']:.0f}</b></span>"
            )
    blocked_class = (
        " score-blocked"
        if it.get("primary_metric") == "task_completion_rate" and not it.get("promotion_eligible")
        else ""
    )
    return f"""<article class="iter-card" data-iter="{it["iter"]}"
  data-score="{it["composite_score"]}" data-sus="{it["sus_score"]}"
  data-eligible="{1 if it.get("promoted", True) else 0}">
  <header>
    <span class="iter-num">#{it["iter"]}</span>
    <span class="badge {_score_class(it["composite_score"])}{blocked_class}">
      {it["composite_score"]:.0f}
    </span>
  </header>
  {thumb}
  <div class="stats">
    {stats}
  </div>
  <p class="feedback">{html.escape(it["feedback"])}</p>
  <details>
    <summary>Suggestions</summary>
    <ul>{suggestions}</ul>
  </details>
</article>"""


# ---------- Routes: pages ----------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    rows = _store().list_runs()[:12]
    recent_cards = []
    for r in rows:
        score_txt = f"{r['best_score']:.0f}" if r["best_score"] is not None else "—"
        brief = (r["brief"] or "")[:140]
        recent_cards.append(
            f"""<a href="/runs/{r["id"]}" class="run-card">
              <div class="run-card-head">
                <span class="run-id">#{r["id"]}</span>
                {_status_badge(r["status"])}
              </div>
              <div class="run-brief">{html.escape(brief)}</div>
              <div class="run-score {_score_class(r["best_score"])}">{score_txt}</div>
            </a>"""
        )
    recent = (
        f'<section class="card"><h2>Recent runs</h2>'
        f'<div class="run-grid">{"".join(recent_cards)}</div></section>'
        if recent_cards
        else ""
    )
    body = f"""<main class="layout">
  {_runs_sidebar(None)}
  <section class="content">
    {_new_run_form()}
    {recent}
  </section>
</main>"""
    return _layout("design-gan", body)


@app.get("/evaluator-review", response_class=HTMLResponse)
def evaluator_review(
    mode: str = "balanced",
    show_observation: bool = False,
    failed_only: bool | None = None,
) -> str:
    # Preserve old bookmarked filters while making independent review the default.
    if failed_only is not None:
        mode = "failures" if failed_only else "all"
    readiness = _corpus_readiness()
    candidates = _review_candidates(mode=mode)
    cards = []
    for candidate in candidates:
        observed = "PASS" if candidate.observed_pass else "FAIL"
        observed_class = "review-pass" if candidate.observed_pass else "review-fail"
        if show_observation:
            errors = (
                "<ul>"
                + "".join(f"<li>{html.escape(error)}</li>" for error in candidate.errors)
                + "</ul>"
                if candidate.errors
                else '<p class="muted">No runtime errors recorded.</p>'
            )
            observation = (
                f'<p>Evaluator observed <b class="{observed_class}">{observed}</b> · '
                f"{candidate.passed_trials}/{candidate.total_trials} trials passed</p>{errors}"
            )
        else:
            observation = (
                '<p class="review-blind">Evaluator observation and runtime evidence are '
                "hidden for independent labeling.</p>"
            )
        captured = ""
        case_id_value = candidate.suggested_case_id
        if case_id_value in candidate.captured_case_ids:
            case_id_value = f"{case_id_value[:71].rstrip('-')}-reviewed"
        actions = f"""
          <label>Stable case id
            <input name="case_id" value="{html.escape(case_id_value)}"
              pattern="[a-z0-9][a-z0-9-]{{2,79}}" required />
          </label>
          <label>Review rationale
            <textarea name="rationale" minlength="10" maxlength="2000" required
              placeholder="Why should this frozen task pass or fail on the artifact?"></textarea>
          </label>
          <div class="review-actions">
            <button type="button" data-label="pass">Label should pass</button>
            <button type="button" data-label="fail" class="secondary">Label should fail</button>
          </div>"""
        if candidate.audited_case_ids:
            if show_observation:
                case_ids = ", ".join(candidate.audited_case_ids)
                captured = (
                    '<p class="review-captured">Already captured with review metadata: '
                    f"<code>{html.escape(case_ids)}</code></p>"
                )
            else:
                captured = '<p class="review-captured">Already captured and audited.</p>'
            actions = '<p class="muted">This exact run, iteration, and task is already labeled.</p>'
        elif candidate.captured_case_ids:
            if show_observation:
                case_ids = ", ".join(candidate.captured_case_ids)
                captured = (
                    '<p class="review-warning">Legacy fixture lacks review metadata: '
                    f"<code>{html.escape(case_ids)}</code>. Save an audited replacement under "
                    "a new case id.</p>"
                )
            else:
                captured = (
                    '<p class="review-warning">A legacy fixture exists for this outcome but '
                    "lacks review metadata. Save an audited replacement under the proposed "
                    "new case id.</p>"
                )
        task_id_attr = html.escape(candidate.task_id)
        site_url = f"/runs/{candidate.run_id}/iters/{candidate.iteration}/site"
        screenshot_url = f"/runs/{candidate.run_id}/iters/{candidate.iteration}/screenshot"
        cards.append(
            f"""<article class="review-card" data-run-id="{candidate.run_id}"
              data-iteration="{candidate.iteration}" data-task-id="{task_id_attr}">
              <a class="review-preview" href="{site_url}" target="_blank" rel="noopener">
                <img src="{screenshot_url}"
                  loading="lazy" alt="Run {candidate.run_id} iteration {candidate.iteration}" />
                <span>Inspect artifact in sandbox →</span>
              </a>
              <div class="review-detail">
                <div class="review-meta">run #{candidate.run_id} · iter {candidate.iteration} ·
                  {html.escape(candidate.domain)} · {task_id_attr}</div>
                <h2>{html.escape(candidate.task_name)}</h2>
                <p>{html.escape(candidate.task_instruction)}</p>
                {observation}
                <p class="muted">artifact <code>{candidate.artifact_sha256[:12]}</code></p>
                {captured}
                <form class="review-form">{actions}
                  <p class="review-status" aria-live="polite"></p></form>
              </div>
            </article>"""
        )
    empty = (
        '<section class="card"><p>No review candidates match this filter. Run a v2 design '
        "evaluation first or select a diagnostic queue.</p></section>"
        if not cards
        else ""
    )
    token_field = (
        '<label class="review-token">Write token '
        '<input id="review-token" type="password" autocomplete="off" '
        'placeholder="Required to save labels" /></label>'
        if _required_start_token()
        else ""
    )
    readiness_class = "review-pass" if readiness.ready else "review-fail"
    readiness_label = "READY" if readiness.ready else "BLOCKED"
    blockers = (
        "<ul>"
        + "".join(f"<li>{html.escape(blocker)}</li>" for blocker in readiness.blockers)
        + "</ul>"
        if readiness.blockers
        else "<p>The provenance corpus clears the initial actor-comparison gate.</p>"
    )
    diagnostic_link = (
        '<a href="/evaluator-review?mode=balanced&show_observation=true">'
        "Reveal evaluator diagnostics</a>"
        if not show_observation
        else '<a href="/evaluator-review">Return to blinded review</a>'
    )
    body = f"""<main class="review-page">
      <header class="review-head">
        <div><h1>Evaluator review</h1>
          <p>Inspect the stored artifact, then independently label whether the frozen task
          should pass. The default policy balances domains and evaluator outcomes without
          revealing those outcomes during labeling.</p></div>
        <div class="review-links">{diagnostic_link}</div>
      </header>
      <section class="review-controls">{token_field}
        <label class="review-token">Reviewer id
          <input id="reviewer-id" autocomplete="off" placeholder="Stable operator id" />
        </label>
        <p class="muted">Labels are stored locally with full artifact provenance and are
        automatically included by benchmark and calibration commands using this corpus.
        Queue sampling policy v{evaluator_benchmark.REVIEW_SAMPLING_POLICY_VERSION} caps
        each source run at {evaluator_benchmark.REVIEW_MAX_CANDIDATES_PER_RUN} items.</p>
      </section>
      <section class="card corpus-readiness">
        <h2>Actor-comparison readiness: <span class="{readiness_class}">{readiness_label}</span></h2>
        <p>{readiness.qualifying_cases} qualifying case(s) ·
          {len(readiness.excluded_cases)} excluded · policy v{readiness.policy_version}</p>
        {blockers}
      </section>
      {empty}{"".join(cards)}
      <script src="/static/evaluator-review.js"></script>
    </main>"""
    return _layout("design-gan · evaluator review", body)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: int) -> str:
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    iters = _store().iterations_for_run(run_id)
    kind = run.get("kind") or "design"

    best_score = run.get("best_score")
    best_score_txt = f"{best_score:.0f}" if best_score is not None else "—"
    v2_design = kind == "design" and bool(run.get("evaluation_plan") or run.get("evaluation_suite"))
    best_score_label = "best task score" if v2_design else "best score"
    suite_html = ""
    if v2_design:
        plan = run.get("evaluation_plan") or {}
        plan_tasks = plan.get("tasks") or run.get("evaluation_suite") or []
        tasks = "".join(
            f"<li><b>{html.escape(task.get('split') or 'development')}</b> · "
            f"{html.escape(task.get('name') or task.get('id') or 'task')}: "
            f"{html.escape(task.get('instruction') or '')}</li>"
            for task in plan_tasks
        )
        plan_meta = (
            f'<p class="muted">domain {html.escape(run.get("domain") or "legacy")} · '
            f"{plan.get('trials_per_task', 1)} trial(s)/task · "
            f"promotion α={plan.get('promotion_alpha', 'legacy')} · product "
            f"<code>{html.escape(run.get('optimization_key') or 'legacy')}</code></p>"
        )
        policy = run.get("artifact_policy") or {}
        policy_meta = (
            f'<p class="muted">artifact {html.escape(policy.get("kind") or "legacy")} · '
            f"max {policy.get('max_bytes', '—')} bytes · "
            f"network {'allowed' if policy.get('network_access') else 'blocked'}</p>"
        )
        suite_html = (
            '<details class="run-evaluation-suite"><summary>Frozen browser tasks</summary>'
            f"{plan_meta}{policy_meta}<ul>{tasks}</ul></details>"
        )
    holdout_html = ""
    if kind == "design" and run.get("holdout_passed") is not None:
        holdout_passed = bool(run["holdout_passed"])
        holdout_score = run.get("holdout_score")
        holdout_score_text = f" · score {holdout_score:.0f}" if holdout_score is not None else ""
        holdout_html = (
            '<p class="run-holdout"><b>Final untouched holdout: '
            f"{'PASS' if holdout_passed else 'FAIL'}</b>{holdout_score_text}</p>"
        )
    challenge_html = ""
    if kind == "design" and run.get("challenge_outcome"):
        outcome = str(run["challenge_outcome"]).replace("_", " ")
        incumbent_id = run.get("incumbent_id")
        incumbent_text = f" · incumbent {incumbent_id}" if incumbent_id is not None else ""
        conflicts = (run.get("challenge_results") or {}).get("arbitration_conflicts", [])
        retry_text = f" · concurrent retries {len(conflicts)}" if conflicts else ""
        challenge_html = (
            '<p class="run-challenge"><b>Cross-run challenge: '
            f"{html.escape(outcome)}</b>{incumbent_text}{retry_text} · product "
            f"<code>{html.escape(run.get('optimization_key') or 'unscoped')}</code></p>"
        )
    cards = "".join(_iter_card_html(run_id, it, kind=kind) for it in iters)

    running = run["status"] == "running"
    attrs = (
        f' data-run-id="{run_id}" data-running="{"1" if running else "0"}"'
        f' data-last-iter="{iters[-1]["iter"] if iters else 0}"'
        f' data-kind="{html.escape(kind)}"'
    )
    cur_iter = run.get("current_iter")
    cur_phase = run.get("current_phase")
    progress_html = _progress_html(
        running=running,
        current_iter=cur_iter,
        current_phase=cur_phase,
        max_iters=run.get("max_iters"),
    )
    error_html = (
        f'<p class="run-error">Last error: {html.escape(run["error"])}</p>'
        if run.get("error")
        else ""
    )

    scrub_link = f'<a class="scrub-link" href="/runs/{run_id}/scrub">Scrub ▸</a>' if iters else ""

    body = f"""<main class="layout">
  {_runs_sidebar(run_id)}
  <section class="content">
    <section class="card run-header">
      <div class="run-header-top">
        <h1>Run #{run_id} {_status_badge(run["status"])} {scrub_link}</h1>
        <div class="run-stats">
          <div><span class="muted">best iter</span>
            <b id="stat-best-iter">{run.get("best_iter") or "—"}</b></div>
          <div><span class="muted">{best_score_label}</span>
            <b id="stat-best-score" class="{_score_class(best_score)}">{best_score_txt}</b></div>
          <div><span class="muted">iterations</span>
            <b id="stat-iter-count">{len(iters)}</b></div>
          <div><span class="muted">holdout</span>
            <b>{"pass" if run.get("holdout_passed") is True else ("fail" if run.get("holdout_passed") is False else "—")}</b></div>
        </div>
      </div>
      {progress_html}
      <p class="brief">{html.escape(run["brief"])}</p>
      {suite_html}
      {holdout_html}
      {challenge_html}
      {error_html}
      <div class="chart-wrap">
        <svg id="score-chart" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>
      </div>
    </section>
    <section class="iter-grid" id="iter-grid">
      {cards}
    </section>
  </section>
</main>"""
    return _layout(f"design-gan — run {run_id}", body, body_attrs=attrs)


@app.get("/runs/{run_id}/scrub", response_class=HTMLResponse)
def scrub(run_id: int) -> str:
    """Immersive scrubber: step through iterations, screenshot + critique side
    by side. The shell is static; scrub.js hydrates it from /api/runs/{id}."""
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    kind = run.get("kind") or "design"
    brief = (run["brief"] or "")[:120]
    attrs = f' data-scrub-run-id="{run_id}" data-kind="{html.escape(kind)}"'
    body = f"""<main class="scrub">
  <div class="scrub-head">
    <div class="scrub-head-left">
      <a class="scrub-back" href="/runs/{run_id}">← Run #{run_id}</a>
      <p class="scrub-brief">{html.escape(brief)}</p>
    </div>
    <div class="scrub-modes" id="scrub-modes" hidden>
      <button type="button" data-mode="single" class="active">Single</button>
      <button type="button" data-mode="prev">vs prev</button>
      <button type="button" data-mode="best">vs best</button>
    </div>
  </div>
  <div class="scrub-top">
    <div class="scrub-stage" id="scrub-stage">
      <div class="scrub-loading muted">Loading run…</div>
    </div>
    <aside class="scrub-panel" id="scrub-panel"></aside>
  </div>
  <div class="scrub-timeline" id="scrub-timeline"></div>
  <script src="/static/scrub.js"></script>
</main>"""
    return _layout(f"design-gan — run {run_id} · scrub", body, body_attrs=attrs)


# ---------- Routes: static ----------


@app.get("/static/{name}")
def static_file(name: str) -> FileResponse:
    # Resolve both sides and require the final path stay inside the static dir.
    # This defeats both `..` traversal and absolute-path escapes (`Path(a) / "/b"`
    # silently discards `a`), which a naive substring check would miss.
    candidate = (_STATIC_DIR / name).resolve()
    try:
        candidate.relative_to(_STATIC_DIR.resolve())
    except ValueError:
        raise HTTPException(404)
    if not candidate.is_file():
        raise HTTPException(404)
    return FileResponse(candidate)


@app.get("/runs/{run_id}/iters/{it}/screenshot")
def screenshot(run_id: int, it: int) -> FileResponse:
    path = _runs_dir() / f"run_{run_id:04d}" / f"iter_{it:03d}" / "screenshot.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/runs/{run_id}/iters/{it}/site")
def site(run_id: int, it: int) -> HTMLResponse:
    path = _runs_dir() / f"run_{run_id:04d}" / f"iter_{it:03d}" / "site.html"
    if not path.exists():
        raise HTTPException(404)
    # The generated HTML is LLM-authored and shaped by user-supplied briefs, so
    # treat it as untrusted. CSP `sandbox` gives the document an opaque origin:
    # its scripts still run (the site stays viewable) but it can't reach this
    # origin's localStorage (where the start token is cached) or call the API
    # with our origin's authority.
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Security-Policy": "sandbox allow-scripts"},
    )


@app.get("/runs/{run_id}/iters/{it}/transcript")
def transcript_json(run_id: int, it: int) -> FileResponse:
    path = _runs_dir() / f"run_{run_id:04d}" / f"iter_{it:03d}" / "transcript.json"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/json")


@app.get("/runs/{run_id}/iters/{it}/transcript-view", response_class=HTMLResponse)
def transcript_view(run_id: int, it: int) -> str:
    """Styled render of the transcript, opened when a user clicks the card thumb."""
    path = _runs_dir() / f"run_{run_id:04d}" / f"iter_{it:03d}" / "transcript.json"
    if not path.exists():
        raise HTTPException(404)
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data.get("transcript", [])
    satisfied = data.get("satisfied")
    turns_taken = data.get("turns_taken")

    bubbles = []
    for t in turns:
        role = t.get("role", "?")
        content = t.get("content", "")
        bubbles.append(
            f'<div class="bubble bubble-{role}">'
            f'<div class="bubble-role">{html.escape(role)}</div>'
            f'<div class="bubble-text">{html.escape(content)}</div>'
            f"</div>"
        )
    meta = (
        f'<div class="transcript-meta muted">'
        f"turns: {turns_taken} · "
        f"satisfied: {'yes' if satisfied else 'no'} · "
        f'<a href="/runs/{run_id}">back to run #{run_id}</a>'
        f"</div>"
    )
    body = (
        f'<main class="transcript-full">'
        f"<h1>Run {run_id} · iter {it} · transcript</h1>"
        f"{meta}"
        f'<div class="transcript-body">{"".join(bubbles)}</div>'
        f"</main>"
    )
    return _layout(f"design-gan · run {run_id} iter {it} transcript", body)


# ---------- Routes: JSON API ----------


@app.get("/api/runs")
def api_runs() -> JSONResponse:
    return JSONResponse(_store().list_runs())


@app.get("/api/runs/{run_id}")
def api_run(run_id: int) -> JSONResponse:
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(404)
    return JSONResponse({"run": run, "iterations": _store().iterations_for_run(run_id)})


@app.get("/api/incumbents")
def api_incumbents() -> JSONResponse:
    return JSONResponse(_store().list_incumbents(active_only=True))


@app.get("/api/evaluator-cases")
def api_evaluator_cases() -> JSONResponse:
    """List operator-labeled cases without exposing their generated HTML."""
    return JSONResponse(
        [
            {
                "id": case.id,
                "domain": case.domain,
                "task_id": case.task.id,
                "expected_pass": case.expected_pass,
                "provenance": case.provenance,
            }
            for case in _evaluator_cases()
        ]
    )


@app.get("/api/evaluator-case-candidates")
def api_evaluator_case_candidates(failed_only: bool = True, limit: int = 100) -> JSONResponse:
    """List run/task outcomes for review without exposing stored HTML."""
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit must be between 1 and 500")
    return JSONResponse(
        [
            {
                **candidate.to_dict(),
                "site_url": (f"/runs/{candidate.run_id}/iters/{candidate.iteration}/site"),
                "screenshot_url": (
                    f"/runs/{candidate.run_id}/iters/{candidate.iteration}/screenshot"
                ),
            }
            for candidate in _review_candidates(
                mode="failures" if failed_only else "all", limit=limit
            )
        ]
    )


@app.get("/api/evaluator-review-queue")
def api_evaluator_review_queue(limit: int = 100) -> JSONResponse:
    """List the balanced queue without exposing evaluator observations or HTML."""
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit must be between 1 and 500")
    items = []
    for candidate in _review_candidates(mode="balanced", limit=limit):
        items.append(
            {
                "run_id": candidate.run_id,
                "iteration": candidate.iteration,
                "task_id": candidate.task_id,
                "task_name": candidate.task_name,
                "task_instruction": candidate.task_instruction,
                "domain": candidate.domain,
                "artifact_sha256": candidate.artifact_sha256,
                "suggested_case_id": candidate.suggested_case_id,
                "site_url": f"/runs/{candidate.run_id}/iters/{candidate.iteration}/site",
                "screenshot_url": (
                    f"/runs/{candidate.run_id}/iters/{candidate.iteration}/screenshot"
                ),
            }
        )
    return JSONResponse(
        {
            "sampling_policy_version": evaluator_benchmark.REVIEW_SAMPLING_POLICY_VERSION,
            "maximum_candidates_per_run": (evaluator_benchmark.REVIEW_MAX_CANDIDATES_PER_RUN),
            "blinded": True,
            "items": items,
        }
    )


@app.get("/api/evaluator-corpus-readiness")
def api_evaluator_corpus_readiness() -> JSONResponse:
    """Report whether real-run evidence clears the actor-comparison gate."""
    return JSONResponse(_corpus_readiness().to_dict())


# ---------- Routes: start + stream ----------


class StartRunRequest(BaseModel):
    brief: str = Field(min_length=1, max_length=2000)
    max_iters: int = Field(default=15, ge=1, le=50)
    patience: int = Field(default=3, ge=1, le=20)
    tolerance: float = Field(default=1.0, ge=0.0, le=100.0)
    model: str | None = None
    token: str | None = None  # required iff DESIGN_GAN_START_TOKEN is set
    kind: str = Field(default="design", pattern="^(design|conversation)$")
    max_conversation_turns: int = Field(default=5, ge=1, le=10)
    design_domain: str = Field(
        default="landing-page",
        pattern="^(landing-page|lead-generation|storefront)$",
    )
    evaluation_trials: int = Field(default=product_domains.DEFAULT_EVALUATION_TRIALS, ge=1, le=50)
    promotion_alpha: float = Field(default=product_domains.DEFAULT_PROMOTION_ALPHA, gt=0.0, le=1.0)
    optimization_key: str | None = Field(default=None, min_length=1, max_length=160)


class CaptureEvaluatorCaseRequest(BaseModel):
    run_id: int = Field(ge=1)
    iteration: int = Field(ge=1)
    task_id: str = Field(min_length=1, max_length=160)
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    expected_pass: bool
    reviewer: str = Field(min_length=2, max_length=80)
    rationale: str = Field(min_length=10, max_length=2000)
    token: str | None = None


def _check_write_token(token: str | None, authorization: str | None) -> None:
    """Enforce the shared mutation token when configured."""
    required = _required_start_token()
    if not required:
        return
    provided = token
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    if not provided or not hmac.compare_digest(provided, required):
        raise HTTPException(status_code=401, detail="invalid or missing token")


@app.post("/api/evaluator-cases", status_code=201)
def capture_evaluator_case(
    req: CaptureEvaluatorCaseRequest,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Persist an operator label for one immutable run/task artifact."""
    _check_write_token(req.token, authorization)
    try:
        case = evaluator_benchmark.capture_run_case(
            _store(),
            run_id=req.run_id,
            iteration=req.iteration,
            task_id=req.task_id,
            case_id=req.case_id,
            expected_pass=req.expected_pass,
            reviewer=req.reviewer,
            rationale=req.rationale,
        )
        evaluator_benchmark.write_case_fixture(
            case,
            _runs_dir() / "evaluator-corpus" / f"{case.id}.json",
            overwrite=False,
        )
    except FileExistsError as exc:
        raise HTTPException(409, f"evaluator case already exists: {req.case_id}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return JSONResponse(
        {
            "id": case.id,
            "domain": case.domain,
            "task_id": case.task.id,
            "expected_pass": case.expected_pass,
            "provenance": case.provenance,
        },
        status_code=201,
    )


@app.get("/api/config")
def api_config() -> JSONResponse:
    """Surface gating + budget state so the UI can show accurate affordances."""
    budget = _daily_budget_usd()
    used = _store().cost_usd_last_24h() if budget is not None else 0.0
    critics = _configured_critics()
    return JSONResponse(
        {
            "requires_token": _required_start_token() is not None,
            "daily_budget_usd": budget,
            "budget_used_24h_usd": round(used, 4),
            "budget_remaining_usd": (
                round(max(0.0, budget - used), 4) if budget is not None else None
            ),
            "critics": [c.name for c in critics] if critics else ["Usability"],
            "design_domains": [
                {"id": domain.id, "name": domain.name, "version": domain.version}
                for domain in product_domains.DOMAINS.values()
            ],
        }
    )


@app.post("/api/runs")
async def start_run(
    req: StartRunRequest, authorization: str | None = Header(default=None)
) -> JSONResponse:
    from . import critic, orchestrator

    _check_write_token(req.token, authorization)

    # Reject up-front when the daily budget is already spent. The orchestrator
    # re-checks between iterations, so a single run starting with headroom
    # can at worst overshoot by one iteration's cost.
    budget = _daily_budget_usd()
    if budget is not None:
        used = _store().cost_usd_last_24h()
        if used >= budget:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "daily_budget_exhausted",
                    "daily_budget_usd": budget,
                    "used_24h_usd": round(used, 4),
                    "message": (
                        f"Daily budget of ${budget:.2f} is spent "
                        f"(${used:.2f} used in the last 24h). Try again tomorrow."
                    ),
                },
            )

    runs_dir = _runs_dir()
    model = req.model or _default_model()
    # When DESIGN_GAN_CRITICS=trio is set we want the conversation CUS trio
    # for conversation runs, and the design TRIO for design runs.
    enabled_critics = _configured_critics()
    if enabled_critics and req.kind == "conversation":
        enabled_critics = list(critic.CUS_TRIO)

    cfg = orchestrator.LoopConfig(
        brief=req.brief,
        runs_dir=runs_dir,
        db_path=runs_dir / "design-gan.sqlite",
        model=model,
        max_iters=req.max_iters,
        patience=req.patience,
        tolerance=req.tolerance,
        daily_budget_usd=budget,
        critics=enabled_critics,
        max_conversation_turns=req.max_conversation_turns,
        design_domain=req.design_domain,
        evaluation_trials=req.evaluation_trials,
        promotion_alpha=req.promotion_alpha,
        optimization_key=req.optimization_key,
    )
    # Pre-create the run so we can return its id immediately.
    plan = (
        product_domains.make_plan(
            req.design_domain,
            trials_per_task=req.evaluation_trials,
            promotion_alpha=req.promotion_alpha,
            minimum_effect=req.tolerance,
        )
        if req.kind == "design"
        else None
    )
    suite = [task.to_dict() for task in plan.tasks] if plan else None
    ledger_key = (
        incumbent_ledger.optimization_key(req.brief, req.optimization_key)
        if plan is not None
        else None
    )
    run_id = _store().create_run(
        req.brief,
        model,
        kind=req.kind,
        evaluation_suite=suite,
        evaluation_plan=plan.to_dict() if plan else None,
        artifact_policy=(
            artifact_policy.DEFAULT_ARTIFACT_POLICY.to_dict() if req.kind == "design" else None
        ),
        domain=plan.domain if plan else None,
        optimization_key=ledger_key,
        max_iters=req.max_iters,
    )
    entry = (
        orchestrator.run_conversation_loop_sync
        if req.kind == "conversation"
        else orchestrator.run_loop_sync
    )
    # Run the loop in a background thread so the event loop stays free to serve SSE.
    task = asyncio.create_task(asyncio.to_thread(entry, cfg, None, run_id))
    _track_run_task(task, run_id)
    return JSONResponse({"run_id": run_id})


# Hold references to in-flight run tasks: asyncio only keeps weak references,
# so an untracked fire-and-forget task can be garbage-collected mid-run and
# its exception silently dropped.
_run_tasks: set[asyncio.Task] = set()


def _track_run_task(task: asyncio.Task, run_id: int) -> None:
    _run_tasks.add(task)

    def _done(t: asyncio.Task) -> None:
        _run_tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            logging.getLogger(__name__).error(
                "background task for run %s died", run_id, exc_info=t.exception()
            )

    task.add_done_callback(_done)


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: int, since: int = 0) -> StreamingResponse:
    """Server-Sent Events: push newly-completed iterations and phase changes."""

    async def event_source():
        last_iter = since
        last_phase_key: tuple[int | None, str | None] | None = None
        store = _store()
        # Short keep-alive loop; stop once the run has a terminal status.
        while True:
            # SQLite calls are synchronous; run them in a worker thread so a
            # slow read never stalls the event loop (and other SSE clients).
            run = await asyncio.to_thread(store.get_run, run_id)
            if not run:
                yield _sse("error", {"message": "run not found"})
                return
            # Newly completed iterations.
            new = await asyncio.to_thread(store.iterations_for_run, run_id, last_iter)
            for it in new:
                yield _sse("iteration", {"run_id": run_id, "iter": it})
                last_iter = it["iter"]
            # Phase transitions (generating / rendering / critiquing / None).
            phase_key = (run.get("current_iter"), run.get("current_phase"))
            if phase_key != last_phase_key:
                yield _sse(
                    "phase",
                    {"iter": phase_key[0], "phase": phase_key[1]},
                )
                last_phase_key = phase_key
            if run["status"] != "running":
                yield _sse("done", {"run": run})
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(event_source(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
