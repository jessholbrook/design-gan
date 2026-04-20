"""SQLite-backed run/iteration history."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brief TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL,
    ended_at REAL,
    best_iter INTEGER,
    best_score REAL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    iter INTEGER NOT NULL,
    created_at REAL NOT NULL,
    html TEXT NOT NULL,
    sus_score REAL NOT NULL,
    axe_penalty REAL NOT NULL,
    composite_score REAL NOT NULL,
    sus_answers TEXT NOT NULL,
    feedback TEXT NOT NULL,
    suggestions TEXT NOT NULL,
    artifacts_dir TEXT NOT NULL,
    UNIQUE(run_id, iter)
);

CREATE INDEX IF NOT EXISTS iterations_run ON iterations(run_id);
"""


@dataclass
class IterationRecord:
    run_id: int
    iter: int
    html: str
    sus_score: float
    axe_penalty: float
    composite_score: float
    sus_answers: list[int]
    feedback: str
    suggestions: list[str]
    artifacts_dir: str


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(self, brief: str, model: str) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs(brief, model, created_at) VALUES (?, ?, ?)",
                (brief, model, time.time()),
            )
            return cur.lastrowid

    def finish_run(self, run_id: int, best_iter: int, best_score: float, status: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET ended_at=?, best_iter=?, best_score=?, status=? WHERE id=?",
                (time.time(), best_iter, best_score, status, run_id),
            )

    def save_iteration(self, rec: IterationRecord) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO iterations(
                    run_id, iter, created_at, html, sus_score, axe_penalty,
                    composite_score, sus_answers, feedback, suggestions, artifacts_dir
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.run_id,
                    rec.iter,
                    time.time(),
                    rec.html,
                    rec.sus_score,
                    rec.axe_penalty,
                    rec.composite_score,
                    json.dumps(rec.sus_answers),
                    rec.feedback,
                    json.dumps(rec.suggestions),
                    rec.artifacts_dir,
                ),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def iterations_for_run(self, run_id: int) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM iterations WHERE run_id=? ORDER BY iter ASC", (run_id,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["sus_answers"] = json.loads(d["sus_answers"])
                d["suggestions"] = json.loads(d["suggestions"])
                out.append(d)
            return out
