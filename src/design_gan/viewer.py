"""FastAPI viewer: dashboard of runs, per-run detail with live SSE updates."""

from __future__ import annotations

import asyncio
import html
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import orchestrator, storage


def _runs_dir() -> Path:
    return Path(os.environ.get("DESIGN_GAN_RUNS_DIR", "./runs"))


def _default_model() -> str:
    return os.environ.get("DESIGN_GAN_MODEL", "claude-sonnet-4-6")


def _store() -> storage.Storage:
    return storage.Storage(_runs_dir() / "design-gan.sqlite")


app = FastAPI(title="design-gan viewer")


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
    return f"""<section class="card new-run">
  <h2>Start a new run</h2>
  <form id="new-run-form">
    <label>Brief
      <textarea name="brief" rows="3" required
        placeholder="A landing page for a weekend cycling tour in rural Vermont."></textarea>
    </label>
    <div class="row">
      <label>Max iterations<input type="number" name="max_iters" value="15" min="1" max="50" /></label>
      <label>Patience<input type="number" name="patience" value="3" min="1" max="10" /></label>
      <label>Tolerance<input type="number" name="tolerance" value="1.0" step="0.5" min="0" /></label>
      <label>Model<input type="text" name="model" value="{html.escape(_default_model())}" /></label>
    </div>
    <button type="submit">Run</button>
    <span id="new-run-status" class="muted"></span>
  </form>
</section>"""


def _iter_card_html(run_id: int, it: dict) -> str:
    suggestions = "".join(
        f"<li>{html.escape(s)}</li>" for s in (it.get("suggestions") or [])
    )
    return f"""<article class="iter-card" data-iter="{it['iter']}">
  <header>
    <span class="iter-num">#{it['iter']}</span>
    <span class="badge {_score_class(it['composite_score'])}">
      {it['composite_score']:.0f}
    </span>
  </header>
  <a href="/runs/{run_id}/iters/{it['iter']}/site" target="_blank" class="thumb">
    <img src="/runs/{run_id}/iters/{it['iter']}/screenshot" loading="lazy" alt="Iter {it['iter']}" />
  </a>
  <div class="stats">
    <span>SUS <b>{it['sus_score']:.0f}</b></span>
    <span>a11y penalty <b>{it['axe_penalty']:.0f}</b></span>
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

    best_score = run.get("best_score")
    best_score_txt = f"{best_score:.0f}" if best_score is not None else "—"
    cards = "".join(_iter_card_html(run_id, it) for it in iters)

    running = run["status"] == "running"
    attrs = (
        f' data-run-id="{run_id}" data-running="{"1" if running else "0"}"'
        f' data-last-iter="{iters[-1]["iter"] if iters else 0}"'
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


@app.post("/api/runs")
async def start_run(req: StartRunRequest) -> JSONResponse:
    runs_dir = _runs_dir()
    model = req.model or _default_model()
    cfg = orchestrator.LoopConfig(
        brief=req.brief,
        runs_dir=runs_dir,
        db_path=runs_dir / "design-gan.sqlite",
        model=model,
        max_iters=req.max_iters,
        patience=req.patience,
        tolerance=req.tolerance,
    )
    # Pre-create the run so we can return its id immediately.
    run_id = _store().create_run(req.brief, model)
    # Run the loop in a background thread so the event loop stays free to serve SSE.
    asyncio.create_task(asyncio.to_thread(orchestrator.run_loop_sync, cfg, None, run_id))
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
