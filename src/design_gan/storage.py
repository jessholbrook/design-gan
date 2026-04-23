"""SQLite-backed run/iteration history."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SECONDS_PER_DAY = 86_400

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brief TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL,
    ended_at REAL,
    best_iter INTEGER,
    best_score REAL,
    status TEXT NOT NULL DEFAULT 'running',
    current_iter INTEGER,
    current_phase TEXT,
    current_phase_at REAL,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    error TEXT
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
    cost_usd REAL NOT NULL DEFAULT 0.0,
    critic_breakdown TEXT,    -- JSON list of per-critic responses; NULL for single-critic
    UNIQUE(run_id, iter)
);

CREATE INDEX IF NOT EXISTS iterations_run ON iterations(run_id);
CREATE INDEX IF NOT EXISTS iterations_created ON iterations(created_at);
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
    cost_usd: float = 0.0
    # Optional per-critic breakdown when the run used an ensemble. Each item:
    #   {"name": str, "sus": list[int], "feedback": str, "suggestions": list[str]}
    critic_breakdown: list[dict[str, Any]] | None = None


class Storage:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns added to an existing deployment."""
        run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        for col, ddl in (
            ("current_iter", "INTEGER"),
            ("current_phase", "TEXT"),
            ("current_phase_at", "REAL"),
            ("total_cost_usd", "REAL NOT NULL DEFAULT 0.0"),
            ("error", "TEXT"),
        ):
            if col not in run_cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")

        iter_cols = {row["name"] for row in conn.execute("PRAGMA table_info(iterations)")}
        if "cost_usd" not in iter_cols:
            conn.execute("ALTER TABLE iterations ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0.0")
        if "critic_breakdown" not in iter_cols:
            conn.execute("ALTER TABLE iterations ADD COLUMN critic_breakdown TEXT")

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

    def finish_run(
        self,
        run_id: int,
        best_iter: int | None,
        best_score: float | None,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET ended_at=?, best_iter=?, best_score=?, status=?, "
                "current_iter=NULL, current_phase=NULL, current_phase_at=NULL, "
                "error=? WHERE id=?",
                (time.time(), best_iter, best_score, status, error, run_id),
            )

    def update_progress(
        self, run_id: int, current_iter: int | None, current_phase: str | None
    ) -> None:
        """Set the in-flight iteration/phase and stamp it with the current time.

        The timestamp lets sweep_abandoned_runs distinguish slow-but-alive runs
        from crashed-out ones without having to poll.
        """
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET current_iter=?, current_phase=?, current_phase_at=? WHERE id=?",
                (current_iter, current_phase, time.time(), run_id),
            )

    def save_iteration(self, rec: IterationRecord) -> None:
        breakdown_json = (
            json.dumps(rec.critic_breakdown) if rec.critic_breakdown is not None else None
        )
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO iterations(
                    run_id, iter, created_at, html, sus_score, axe_penalty,
                    composite_score, sus_answers, feedback, suggestions, artifacts_dir,
                    cost_usd, critic_breakdown
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    rec.cost_usd,
                    breakdown_json,
                ),
            )
            # Roll iteration cost up onto the parent run for cheap dashboard reads.
            c.execute(
                "UPDATE runs SET total_cost_usd = total_cost_usd + ? WHERE id=?",
                (rec.cost_usd, rec.run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def iterations_for_run(self, run_id: int, after_iter: int = 0) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM iterations WHERE run_id=? AND iter>? ORDER BY iter ASC",
                (run_id, after_iter),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["sus_answers"] = json.loads(d["sus_answers"])
                d["suggestions"] = json.loads(d["suggestions"])
                if d.get("critic_breakdown"):
                    d["critic_breakdown"] = json.loads(d["critic_breakdown"])
                out.append(d)
            return out

    def cost_usd_since(self, epoch: float) -> float:
        """Sum of iterations.cost_usd for iters created since `epoch`."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) AS total "
                "FROM iterations WHERE created_at >= ?",
                (epoch,),
            ).fetchone()
            return float(row["total"] or 0.0)

    def cost_usd_last_24h(self) -> float:
        return self.cost_usd_since(time.time() - SECONDS_PER_DAY)

    def sweep_abandoned_runs(self, timeout_seconds: float) -> list[int]:
        """Mark status='running' rows with no recent heartbeat as errored.

        A "heartbeat" is `current_phase_at` — set by update_progress on every
        phase transition. Rows that have been running without a phase stamp
        for longer than `timeout_seconds` (or never got one) are presumed
        dead (e.g. the machine restarted mid-run). Returns the swept ids.
        """
        cutoff = time.time() - timeout_seconds
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM runs WHERE status='running' AND "
                "(current_phase_at IS NULL AND created_at < ? "
                " OR current_phase_at IS NOT NULL AND current_phase_at < ?)",
                (cutoff, cutoff),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                qmarks = ",".join("?" * len(ids))
                c.execute(
                    f"UPDATE runs SET status='errored', ended_at=?, "
                    f"current_iter=NULL, current_phase=NULL, current_phase_at=NULL, "
                    f"error='abandoned: no heartbeat for {int(timeout_seconds)}s' "
                    f"WHERE id IN ({qmarks})",
                    (time.time(), *ids),
                )
            return ids
