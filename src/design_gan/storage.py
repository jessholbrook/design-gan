"""SQLite-backed run/iteration history."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SECONDS_PER_DAY = 86_400


class IncumbentConflict(RuntimeError):
    """The active incumbent changed after a challenger evaluated it."""

    def __init__(self, current_incumbent_id: int | None):
        self.current_incumbent_id = current_incumbent_id
        super().__init__(f"active incumbent changed to {current_incumbent_id}")


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brief TEXT NOT NULL,            -- for kind='design': the site brief; for kind='conversation': the goal
    model TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'design',  -- 'design' | 'conversation'
    created_at REAL NOT NULL,
    ended_at REAL,
    best_iter INTEGER,
    best_score REAL,
    status TEXT NOT NULL DEFAULT 'running',
    current_iter INTEGER,
    current_phase TEXT,
    current_phase_at REAL,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    domain TEXT,
    evaluation_suite TEXT,        -- frozen JSON browser scenarios for design runs
    evaluation_plan TEXT,         -- versioned frozen evaluator + promotion policy
    artifact_policy TEXT,         -- versioned mutable-artifact boundary
    optimization_key TEXT,        -- explicit or brief-derived cross-run product scope
    incumbent_id INTEGER,
    challenge_outcome TEXT,
    challenge_results TEXT,
    holdout_score REAL,
    holdout_passed INTEGER,
    holdout_results TEXT,
    holdout_evaluated_at REAL,
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
    primary_score REAL,       -- task completion for v2 design runs
    primary_metric TEXT,
    promotion_eligible INTEGER NOT NULL DEFAULT 1,
    guardrails TEXT,          -- JSON accessibility/correctness gate results
    task_results TEXT,        -- JSON behavioral browser task results
    artifact_validation TEXT,
    parent_iter INTEGER,
    promoted INTEGER NOT NULL DEFAULT 1,
    promotion_reason TEXT,
    promotion_effect REAL,
    promotion_p_value REAL,
    promotion_comparable_trials INTEGER NOT NULL DEFAULT 0,
    promotion_wins INTEGER NOT NULL DEFAULT 0,
    promotion_losses INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_id, iter)
);

CREATE INDEX IF NOT EXISTS iterations_run ON iterations(run_id);
CREATE INDEX IF NOT EXISTS iterations_created ON iterations(created_at);

CREATE TABLE IF NOT EXISTS incumbents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    optimization_key TEXT NOT NULL,
    domain TEXT NOT NULL,
    domain_version INTEGER NOT NULL,
    evaluator_version INTEGER NOT NULL,
    artifact_policy_version INTEGER NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    iter INTEGER NOT NULL,
    artifact_hash TEXT NOT NULL,
    html TEXT NOT NULL,
    primary_score REAL NOT NULL,
    holdout_score REAL NOT NULL,
    holdout_results TEXT NOT NULL,
    created_at REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    supersedes_id INTEGER REFERENCES incumbents(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS incumbents_active_contract
ON incumbents(
    optimization_key, domain, domain_version, evaluator_version, artifact_policy_version
) WHERE active = 1;

CREATE TABLE IF NOT EXISTS incumbent_challenges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    challenger_run_id INTEGER NOT NULL UNIQUE REFERENCES runs(id),
    prior_incumbent_id INTEGER REFERENCES incumbents(id),
    resulting_incumbent_id INTEGER REFERENCES incumbents(id),
    outcome TEXT NOT NULL,
    evidence TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS incumbent_challenges_created
ON incumbent_challenges(created_at);
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
    primary_score: float | None = None
    primary_metric: str | None = None
    promotion_eligible: bool = True
    guardrails: dict[str, Any] | None = None
    task_results: list[dict[str, Any]] | None = None
    artifact_validation: dict[str, Any] | None = None
    parent_iter: int | None = None
    promoted: bool = True
    promotion_reason: str | None = None
    promotion_effect: float | None = None
    promotion_p_value: float | None = None
    promotion_comparable_trials: int = 0
    promotion_wins: int = 0
    promotion_losses: int = 0


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
            ("domain", "TEXT"),
            ("evaluation_suite", "TEXT"),
            ("evaluation_plan", "TEXT"),
            ("artifact_policy", "TEXT"),
            ("optimization_key", "TEXT"),
            ("incumbent_id", "INTEGER"),
            ("challenge_outcome", "TEXT"),
            ("challenge_results", "TEXT"),
            ("holdout_score", "REAL"),
            ("holdout_passed", "INTEGER"),
            ("holdout_results", "TEXT"),
            ("holdout_evaluated_at", "REAL"),
            ("error", "TEXT"),
            ("kind", "TEXT NOT NULL DEFAULT 'design'"),
        ):
            if col not in run_cols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")

        iter_cols = {row["name"] for row in conn.execute("PRAGMA table_info(iterations)")}
        promoted_added = "promoted" not in iter_cols
        if "cost_usd" not in iter_cols:
            conn.execute("ALTER TABLE iterations ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0.0")
        if "critic_breakdown" not in iter_cols:
            conn.execute("ALTER TABLE iterations ADD COLUMN critic_breakdown TEXT")
        for col, ddl in (
            ("primary_score", "REAL"),
            ("primary_metric", "TEXT"),
            ("promotion_eligible", "INTEGER NOT NULL DEFAULT 1"),
            ("guardrails", "TEXT"),
            ("task_results", "TEXT"),
            ("artifact_validation", "TEXT"),
            ("parent_iter", "INTEGER"),
            ("promoted", "INTEGER NOT NULL DEFAULT 1"),
            ("promotion_reason", "TEXT"),
            ("promotion_effect", "REAL"),
            ("promotion_p_value", "REAL"),
            ("promotion_comparable_trials", "INTEGER NOT NULL DEFAULT 0"),
            ("promotion_wins", "INTEGER NOT NULL DEFAULT 0"),
            ("promotion_losses", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if col not in iter_cols:
                conn.execute(f"ALTER TABLE iterations ADD COLUMN {col} {ddl}")

        if promoted_added:
            # v2 milestone-one databases predate explicit promotion decisions.
            # Preserve their visible best candidate without labeling every
            # historical proposal as promoted.
            conn.execute(
                """
                UPDATE iterations
                SET promoted = CASE
                    WHEN iter = (
                        SELECT runs.best_iter FROM runs WHERE runs.id = iterations.run_id
                    ) THEN 1
                    ELSE 0
                END,
                promotion_reason = CASE
                    WHEN iter = (
                        SELECT runs.best_iter FROM runs WHERE runs.id = iterations.run_id
                    ) THEN 'legacy_best_candidate'
                    ELSE 'legacy_not_selected'
                END
                WHERE primary_metric = 'task_completion_rate'
                """
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # The run loop writes from a worker thread while the viewer reads from
        # the web server, so configure for concurrency: WAL lets readers and
        # the writer proceed without blocking each other, and the generous
        # busy timeout rides out any residual lock contention.
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(
        self,
        brief: str,
        model: str,
        kind: str = "design",
        evaluation_suite: list[dict[str, Any]] | None = None,
        evaluation_plan: dict[str, Any] | None = None,
        artifact_policy: dict[str, Any] | None = None,
        domain: str | None = None,
        optimization_key: str | None = None,
    ) -> int:
        suite_json = json.dumps(evaluation_suite) if evaluation_suite is not None else None
        plan_json = json.dumps(evaluation_plan) if evaluation_plan is not None else None
        policy_json = json.dumps(artifact_policy) if artifact_policy is not None else None
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs(brief, model, kind, created_at, domain, evaluation_suite, "
                "evaluation_plan, artifact_policy, optimization_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    brief,
                    model,
                    kind,
                    time.time(),
                    domain,
                    suite_json,
                    plan_json,
                    policy_json,
                    optimization_key,
                ),
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

    def set_run_optimization_key(self, run_id: int, optimization_key: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET optimization_key=? WHERE id=?",
                (optimization_key, run_id),
            )

    def save_holdout_audit(
        self,
        run_id: int,
        *,
        score: float | None,
        passed: bool,
        results: dict[str, Any],
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET holdout_score=?, holdout_passed=?, holdout_results=?, "
                "holdout_evaluated_at=? WHERE id=?",
                (score, int(passed), json.dumps(results), time.time(), run_id),
            )

    def get_active_incumbent(
        self,
        *,
        optimization_key: str,
        domain: str,
        domain_version: int,
        evaluator_version: int,
        artifact_policy_version: int,
    ) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM incumbents WHERE optimization_key=? AND domain=? "
                "AND domain_version=? AND evaluator_version=? "
                "AND artifact_policy_version=? AND active=1",
                (
                    optimization_key,
                    domain,
                    domain_version,
                    evaluator_version,
                    artifact_policy_version,
                ),
            ).fetchone()
            return self._decode_incumbent(dict(row)) if row else None

    def resolve_incumbent_challenge(
        self,
        *,
        run_id: int,
        contract: dict[str, Any],
        prior_incumbent_id: int | None,
        outcome: str,
        evidence: dict[str, Any],
        candidate_iter: int,
        candidate_html: str,
        candidate_artifact_hash: str,
        candidate_primary_score: float,
        candidate_holdout_score: float,
        candidate_holdout_results: dict[str, Any],
    ) -> int | None:
        """Atomically record a challenge and install its candidate when selected."""
        if outcome not in {
            "established",
            "replaced",
            "retained",
            "rejected_holdout",
            "inconclusive",
        }:
            raise ValueError(f"unsupported challenge outcome: {outcome}")
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute(
                "SELECT resulting_incumbent_id FROM incumbent_challenges WHERE challenger_run_id=?",
                (run_id,),
            ).fetchone()
            if existing:
                return existing["resulting_incumbent_id"]

            current = c.execute(
                "SELECT id FROM incumbents WHERE optimization_key=? AND domain=? "
                "AND domain_version=? AND evaluator_version=? "
                "AND artifact_policy_version=? AND active=1",
                (
                    contract["optimization_key"],
                    contract["domain"],
                    contract["domain_version"],
                    contract["evaluator_version"],
                    contract["artifact_policy_version"],
                ),
            ).fetchone()
            current_incumbent_id = current["id"] if current else None
            if current_incumbent_id != prior_incumbent_id:
                raise IncumbentConflict(current_incumbent_id)

            resulting_incumbent_id = prior_incumbent_id
            if outcome in {"established", "replaced"}:
                if outcome == "replaced":
                    if prior_incumbent_id is None:
                        raise ValueError("replaced challenge requires a prior incumbent")
                    c.execute(
                        "UPDATE incumbents SET active=0 WHERE id=? AND active=1",
                        (prior_incumbent_id,),
                    )
                cur = c.execute(
                    "INSERT INTO incumbents(optimization_key, domain, domain_version, "
                    "evaluator_version, artifact_policy_version, run_id, iter, "
                    "artifact_hash, html, primary_score, holdout_score, holdout_results, "
                    "created_at, active, supersedes_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                    (
                        contract["optimization_key"],
                        contract["domain"],
                        contract["domain_version"],
                        contract["evaluator_version"],
                        contract["artifact_policy_version"],
                        run_id,
                        candidate_iter,
                        candidate_artifact_hash,
                        candidate_html,
                        candidate_primary_score,
                        candidate_holdout_score,
                        json.dumps(candidate_holdout_results),
                        time.time(),
                        prior_incumbent_id,
                    ),
                )
                resulting_incumbent_id = cur.lastrowid

            c.execute(
                "INSERT INTO incumbent_challenges(challenger_run_id, prior_incumbent_id, "
                "resulting_incumbent_id, outcome, evidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    prior_incumbent_id,
                    resulting_incumbent_id,
                    outcome,
                    json.dumps(evidence),
                    time.time(),
                ),
            )
            c.execute(
                "UPDATE runs SET incumbent_id=?, challenge_outcome=?, challenge_results=? "
                "WHERE id=?",
                (
                    resulting_incumbent_id,
                    outcome,
                    json.dumps(evidence),
                    run_id,
                ),
            )
            return resulting_incumbent_id

    def list_incumbents(
        self, *, active_only: bool = False, include_html: bool = False
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM incumbents"
        if active_only:
            query += " WHERE active=1"
        query += " ORDER BY created_at DESC"
        with self._conn() as c:
            rows = c.execute(query).fetchall()
            items = [self._decode_incumbent(dict(row)) for row in rows]
            if not include_html:
                for item in items:
                    item.pop("html", None)
            return items

    def record_inconclusive_challenge(
        self,
        *,
        run_id: int,
        contract: dict[str, Any],
        evidence: dict[str, Any],
    ) -> int | None:
        """Record a terminal non-mutating result after bounded CAS conflicts."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute(
                "SELECT resulting_incumbent_id FROM incumbent_challenges WHERE challenger_run_id=?",
                (run_id,),
            ).fetchone()
            if existing:
                return existing["resulting_incumbent_id"]
            current = c.execute(
                "SELECT id FROM incumbents WHERE optimization_key=? AND domain=? "
                "AND domain_version=? AND evaluator_version=? "
                "AND artifact_policy_version=? AND active=1",
                (
                    contract["optimization_key"],
                    contract["domain"],
                    contract["domain_version"],
                    contract["evaluator_version"],
                    contract["artifact_policy_version"],
                ),
            ).fetchone()
            current_id = current["id"] if current else None
            c.execute(
                "INSERT INTO incumbent_challenges(challenger_run_id, prior_incumbent_id, "
                "resulting_incumbent_id, outcome, evidence, created_at) "
                "VALUES (?, ?, ?, 'inconclusive', ?, ?)",
                (run_id, current_id, current_id, json.dumps(evidence), time.time()),
            )
            c.execute(
                "UPDATE runs SET incumbent_id=?, challenge_outcome='inconclusive', "
                "challenge_results=? WHERE id=?",
                (current_id, json.dumps(evidence), run_id),
            )
            return current_id

    @staticmethod
    def _decode_incumbent(item: dict[str, Any]) -> dict[str, Any]:
        item["holdout_results"] = json.loads(item["holdout_results"])
        item["active"] = bool(item["active"])
        return item

    def save_iteration(self, rec: IterationRecord) -> None:
        breakdown_json = (
            json.dumps(rec.critic_breakdown) if rec.critic_breakdown is not None else None
        )
        guardrails_json = json.dumps(rec.guardrails) if rec.guardrails is not None else None
        task_results_json = json.dumps(rec.task_results) if rec.task_results is not None else None
        artifact_validation_json = (
            json.dumps(rec.artifact_validation) if rec.artifact_validation is not None else None
        )
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO iterations(
                    run_id, iter, created_at, html, sus_score, axe_penalty,
                    composite_score, sus_answers, feedback, suggestions, artifacts_dir,
                    cost_usd, critic_breakdown, primary_score, primary_metric,
                    promotion_eligible, guardrails, task_results
                    , artifact_validation, parent_iter, promoted, promotion_reason,
                    promotion_effect, promotion_p_value, promotion_comparable_trials,
                    promotion_wins, promotion_losses
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    rec.primary_score,
                    rec.primary_metric,
                    int(rec.promotion_eligible),
                    guardrails_json,
                    task_results_json,
                    artifact_validation_json,
                    rec.parent_iter,
                    int(rec.promoted),
                    rec.promotion_reason,
                    rec.promotion_effect,
                    rec.promotion_p_value,
                    rec.promotion_comparable_trials,
                    rec.promotion_wins,
                    rec.promotion_losses,
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
            return [self._decode_run(dict(r)) for r in rows]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            return self._decode_run(dict(row)) if row else None

    @staticmethod
    def _decode_run(run: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "evaluation_suite",
            "evaluation_plan",
            "artifact_policy",
            "challenge_results",
            "holdout_results",
        ):
            if run.get(key):
                run[key] = json.loads(run[key])
        if run.get("holdout_passed") is not None:
            run["holdout_passed"] = bool(run["holdout_passed"])
        return run

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
                if d.get("guardrails"):
                    d["guardrails"] = json.loads(d["guardrails"])
                if d.get("task_results"):
                    d["task_results"] = json.loads(d["task_results"])
                if d.get("artifact_validation"):
                    d["artifact_validation"] = json.loads(d["artifact_validation"])
                d["promotion_eligible"] = bool(d.get("promotion_eligible", 1))
                d["promoted"] = bool(d.get("promoted", 1))
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
