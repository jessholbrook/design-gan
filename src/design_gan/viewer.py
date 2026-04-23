"""FastAPI viewer: dashboard of runs, per-run detail with live SSE updates."""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import critic, orchestrator, storage


def _runs_dir() -> Path:
    return Path(os.environ.get("DESIGN_GAN_RUNS_DIR", "./runs"))


def _default_model() -> str:
    return os.environ.get("DESIGN_GAN_MODEL", "claude-sonnet-4-6")


def _required_start_token() -> str | None:
    """If set, POST /api/runs must present this token. Unset -> gate disabled."""
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
ABANDONED_RUN_TIMEOUT_SECONDS = int(
    os.environ.get("DESIGN_GAN_ABANDONED_TIMEOUT_S", "600")
)


def _configured_critics() -> list[critic.CriticProfile] | None:
    """DESIGN_GAN_CRITICS=trio opts into the 3-critic ensemble.

    Unset or 'solo' keeps the single Usability critic (backward compat).
    Runs triggered while this env var is set will use the ensemble.
    """
    mode = (os.environ.get("DESIGN_GAN_CRITICS") or "").strip().lower()
    if mode == "trio":
        return list(critic.TRIO)
    return None


def _store() -> storage.Storage:
    return storage.Storage(_runs_dir() / "design-gan.sqlite")


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Boot-time cleanup: any run still marked 'running' is a ghost from a
    prior machine (restart, OOM, etc.). Mark them errored so the UI doesn't
    show a spinner that never resolves."""
    import logging
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
    return f'<span class="status status-{cls}">{html.escape(status)}</span>'


def _runs_sidebar(active_id: int | None) -> str:
    rows = _store().list_runs()
    items = []
    for r in rows:
        active = " active" if active_id == r["id"] else ""
        brief = (r["brief"] or "")[:60]
        score_txt = f"{r['best_score']:.0f}" if r["best_score"] is not None else "—"
        items.append(
            f"""<a href="/runs/{r['id']}" class="side-item{active}">
              <span class="side-id">#{r['id']}</span>
              <span class="side-brief">{html.escape(brief)}</span>
              <span class="side-score {_score_class(r['best_score'])}">{score_txt}</span>
            </a>"""
        )
    if not items:
        items.append('<div class="side-empty muted">No runs yet.</div>')
    return f'<aside class="sidebar"><h3>Runs</h3>{"".join(items)}</aside>'


def _new_run_form() -> str:
    gated = _required_start_token() is not None
    gated_attr = ' data-requires-token="1"' if gated else ""
    token_field = (
        '<label class="token-field">Access token'
        '<input type="password" name="token" autocomplete="off" '
        'placeholder="Required to start a run on this deployment" /></label>'
        if gated else ""
    )
    gated_note = (
        '<p class="muted gated-note">Starting a run requires a shared token '
        'on this deployment — ask the owner. Browsing existing runs is open.</p>'
        if gated else ""
    )
    return f"""<section class="card new-run">
  <h2>Start a new run</h2>
  {gated_note}
  <form id="new-run-form"{gated_attr}>
    <label>Kind
      <select name="kind">
        <option value="design">Design — evolve a single-page website</option>
        <option value="conversation">Conversation — evolve an assistant for a 1–5 turn chat</option>
      </select>
    </label>
    <label data-brief-label>Brief
      <textarea name="brief" rows="3" required
        placeholder="A landing page for a weekend cycling tour in rural Vermont."></textarea>
    </label>
    <div class="row">
      <label>Max iterations<input type="number" name="max_iters" value="15" min="1" max="50" /></label>
      <label>Patience<input type="number" name="patience" value="3" min="1" max="10" /></label>
      <label>Tolerance<input type="number" name="tolerance" value="1.0" step="0.5" min="0" /></label>
      <label>Model<input type="text" name="model" value="{html.escape(_default_model())}" /></label>
    </div>
    <label data-conversation-only hidden>Max conversation turns
      <input type="number" name="max_conversation_turns" value="5" min="1" max="10" />
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
            f'</div>'
        )
    return '<div class="transcript-preview">' + "".join(bubbles) + '</div>'


def _iter_card_html(run_id: int, it: dict, kind: str = "design") -> str:
    suggestions = "".join(
        f"<li>{html.escape(s)}</li>" for s in (it.get("suggestions") or [])
    )
    if kind == "conversation":
        thumb = (
            f'<a href="/runs/{run_id}/iters/{it["iter"]}/transcript-view" '
            f'target="_blank" class="thumb thumb-transcript">'
            f'{_transcript_preview_html(run_id, it["iter"])}'
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
        stats = (
            f"<span>SUS <b>{it['sus_score']:.0f}</b></span>"
            f"<span>a11y penalty <b>{it['axe_penalty']:.0f}</b></span>"
        )
    return f"""<article class="iter-card" data-iter="{it['iter']}">
  <header>
    <span class="iter-num">#{it['iter']}</span>
    <span class="badge {_score_class(it['composite_score'])}">
      {it['composite_score']:.0f}
    </span>
  </header>
  {thumb}
  <div class="stats">
    {stats}
  </div>
  <p class="feedback">{html.escape(it['feedback'])}</p>
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
            f"""<a href="/runs/{r['id']}" class="run-card">
              <div class="run-card-head">
                <span class="run-id">#{r['id']}</span>
                {_status_badge(r['status'])}
              </div>
              <div class="run-brief">{html.escape(brief)}</div>
              <div class="run-score {_score_class(r['best_score'])}">{score_txt}</div>
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


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: int) -> str:
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    iters = _store().iterations_for_run(run_id)
    kind = run.get("kind") or "design"

    best_score = run.get("best_score")
    best_score_txt = f"{best_score:.0f}" if best_score is not None else "—"
    cards = "".join(_iter_card_html(run_id, it, kind=kind) for it in iters)

    running = run["status"] == "running"
    attrs = (
        f' data-run-id="{run_id}" data-running="{"1" if running else "0"}"'
        f' data-last-iter="{iters[-1]["iter"] if iters else 0}"'
        f' data-kind="{html.escape(kind)}"'
    )
    cur_iter = run.get("current_iter")
    cur_phase = run.get("current_phase")
    progress_display = "flex" if running and cur_phase else "none"
    progress_text = (
        f"iter {cur_iter} · {cur_phase}" if (running and cur_iter and cur_phase) else ""
    )
    error_html = (
        f'<p class="run-error">Last error: {html.escape(run["error"])}</p>'
        if run.get("error")
        else ""
    )

    body = f"""<main class="layout">
  {_runs_sidebar(run_id)}
  <section class="content">
    <section class="card run-header">
      <div class="run-header-top">
        <h1>Run #{run_id} {_status_badge(run['status'])}
          <span id="progress-indicator" class="progress" style="display:{progress_display}">
            <span class="spinner"></span>
            <span id="progress-text">{html.escape(progress_text)}</span>
          </span>
        </h1>
        <div class="run-stats">
          <div><span class="muted">best iter</span>
            <b id="stat-best-iter">{run.get('best_iter') or '—'}</b></div>
          <div><span class="muted">best score</span>
            <b id="stat-best-score" class="{_score_class(best_score)}">{best_score_txt}</b></div>
          <div><span class="muted">iterations</span>
            <b id="stat-iter-count">{len(iters)}</b></div>
        </div>
      </div>
      <p class="brief">{html.escape(run['brief'])}</p>
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


@app.get("/runs/{run_id}/iters/{it}/site", response_class=HTMLResponse)
def site(run_id: int, it: int) -> str:
    path = _runs_dir() / f"run_{run_id:04d}" / f"iter_{it:03d}" / "site.html"
    if not path.exists():
        raise HTTPException(404)
    return path.read_text(encoding="utf-8")


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
        f'turns: {turns_taken} · '
        f'satisfied: {"yes" if satisfied else "no"} · '
        f'<a href="/runs/{run_id}">back to run #{run_id}</a>'
        f"</div>"
    )
    body = (
        f'<main class="transcript-full">'
        f'<h1>Run {run_id} · iter {it} · transcript</h1>'
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


def _check_start_token(req: StartRunRequest, authorization: str | None) -> None:
    """Enforce DESIGN_GAN_START_TOKEN if set. Accept body.token or `Authorization: Bearer`."""
    required = _required_start_token()
    if not required:
        return
    provided = req.token
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    if not provided or not hmac.compare_digest(provided, required):
        raise HTTPException(status_code=401, detail="invalid or missing token")


@app.get("/api/config")
def api_config() -> JSONResponse:
    """Surface gating + budget state so the UI can show accurate affordances."""
    budget = _daily_budget_usd()
    used = _store().cost_usd_last_24h() if budget is not None else 0.0
    critics = _configured_critics()
    return JSONResponse({
        "requires_token": _required_start_token() is not None,
        "daily_budget_usd": budget,
        "budget_used_24h_usd": round(used, 4),
        "budget_remaining_usd": (
            round(max(0.0, budget - used), 4) if budget is not None else None
        ),
        "critics": [c.name for c in critics] if critics else ["Usability"],
    })


@app.post("/api/runs")
async def start_run(
    req: StartRunRequest, authorization: str | None = Header(default=None)
) -> JSONResponse:
    _check_start_token(req, authorization)

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
    )
    # Pre-create the run so we can return its id immediately.
    run_id = _store().create_run(req.brief, model, kind=req.kind)
    entry = (
        orchestrator.run_conversation_loop_sync
        if req.kind == "conversation"
        else orchestrator.run_loop_sync
    )
    # Run the loop in a background thread so the event loop stays free to serve SSE.
    asyncio.create_task(asyncio.to_thread(entry, cfg, None, run_id))
    return JSONResponse({"run_id": run_id})


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: int, since: int = 0) -> StreamingResponse:
    """Server-Sent Events: push newly-completed iterations and phase changes."""

    async def event_source():
        last_iter = since
        last_phase_key: tuple[int | None, str | None] | None = None
        # Short keep-alive loop; stop once the run has a terminal status.
        while True:
            store = _store()
            run = store.get_run(run_id)
            if not run:
                yield _sse("error", {"message": "run not found"})
                return
            # Newly completed iterations.
            new = store.iterations_for_run(run_id, after_iter=last_iter)
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
