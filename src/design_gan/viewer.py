"""Minimal FastAPI viewer for browsing runs and iterations."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import storage


def _runs_dir() -> Path:
    return Path(os.environ.get("DESIGN_GAN_RUNS_DIR", "./runs"))


def _store() -> storage.Storage:
    return storage.Storage(_runs_dir() / "design-gan.sqlite")


app = FastAPI(title="design-gan viewer")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    rows = _store().list_runs()
    body = ["<h1>design-gan runs</h1>"]
    if not rows:
        body.append("<p>No runs yet.</p>")
    else:
        body.append(
            "<table border=1 cellpadding=6 style='border-collapse:collapse'>"
            "<tr><th>id</th><th>brief</th><th>model</th><th>best iter</th>"
            "<th>best score</th><th>status</th></tr>"
        )
        for r in rows:
            brief = (r["brief"] or "")[:80]
            best_iter = r["best_iter"] if r["best_iter"] is not None else "-"
            best_score = f"{r['best_score']:.1f}" if r["best_score"] is not None else "-"
            body.append(
                f"<tr><td><a href='/runs/{r['id']}'>{r['id']}</a></td>"
                f"<td>{brief}</td><td>{r['model']}</td>"
                f"<td>{best_iter}</td><td>{best_score}</td><td>{r['status']}</td></tr>"
            )
        body.append("</table>")
    return "<html><body style='font-family:system-ui'>" + "\n".join(body) + "</body></html>"


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(run_id: int) -> str:
    run = _store().get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    iters = _store().iterations_for_run(run_id)
    body = [f"<h1>Run {run_id}</h1>", f"<p><b>Brief:</b> {run['brief']}</p>"]
    for it in iters:
        body.append(
            f"<h2>Iter {it['iter']} — composite {it['composite_score']:.1f} "
            f"(SUS {it['sus_score']:.1f}, a11y penalty {it['axe_penalty']:.1f})</h2>"
            f"<img src='/runs/{run_id}/iters/{it['iter']}/screenshot' "
            f"style='max-width:640px;border:1px solid #ccc' />"
            f"<p><b>Feedback:</b> {it['feedback']}</p>"
            f"<p><b>Suggestions:</b></p><ul>"
            + "".join(f"<li>{s}</li>" for s in it["suggestions"])
            + "</ul>"
            f"<p><a href='/runs/{run_id}/iters/{it['iter']}/site'>open site</a></p>"
        )
    return "<html><body style='font-family:system-ui'>" + "\n".join(body) + "</body></html>"


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
