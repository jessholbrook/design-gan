"""Unit tests for storage.py — SQLite schema, CRUD, migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from design_gan.storage import IterationRecord, Storage


@pytest.fixture
def store(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.sqlite")


def _sample_record(run_id: int, iter_num: int, composite: float = 50.0) -> IterationRecord:
    return IterationRecord(
        run_id=run_id,
        iter=iter_num,
        html="<html>x</html>",
        sus_score=50.0,
        axe_penalty=0.0,
        composite_score=composite,
        sus_answers=[3] * 10,
        feedback="meh",
        suggestions=["do better"],
        artifacts_dir=f"/tmp/run_{run_id:04d}/iter_{iter_num:03d}",
    )


class TestInitAndSchema:
    def test_creates_db_file(self, tmp_path: Path):
        path = tmp_path / "sub" / "test.sqlite"
        Storage(path)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "c" / "test.sqlite"
        Storage(path)
        assert path.parent.exists()

    def test_has_required_tables(self, store: Storage):
        with sqlite3.connect(store.db_path) as c:
            tables = {row[0] for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert {"runs", "iterations"}.issubset(tables)

    def test_runs_has_progress_columns(self, store: Storage):
        with sqlite3.connect(store.db_path) as c:
            cols = {row[1] for row in c.execute("PRAGMA table_info(runs)").fetchall()}
        assert {"current_iter", "current_phase", "error"}.issubset(cols)


class TestMigration:
    def test_adds_missing_columns_to_existing_db(self, tmp_path: Path):
        # Simulate an old deployment without the progress columns.
        db = tmp_path / "old.sqlite"
        with sqlite3.connect(db) as c:
            c.execute("""CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brief TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL,
                ended_at REAL,
                best_iter INTEGER,
                best_score REAL,
                status TEXT NOT NULL DEFAULT 'running'
            )""")
            c.execute(
                "INSERT INTO runs(brief, model, created_at) VALUES ('b', 'm', 0.0)"
            )
            c.commit()
        # Open with the current Storage — migration should add missing columns.
        store = Storage(db)
        run = store.get_run(1)
        assert run is not None
        # New columns exist (and are None on this legacy row).
        assert "current_iter" in run
        assert "current_phase" in run
        assert "error" in run

    def test_migration_is_idempotent(self, store: Storage):
        # Running Storage() again on the same path must not raise.
        Storage(store.db_path)
        Storage(store.db_path)


class TestRuns:
    def test_create_then_list(self, store: Storage):
        store.create_run("brief one", "model-a")
        store.create_run("brief two", "model-b")
        rows = store.list_runs()
        assert len(rows) == 2
        # Ordered DESC by id.
        assert rows[0]["brief"] == "brief two"
        assert rows[1]["brief"] == "brief one"

    def test_create_returns_monotonic_ids(self, store: Storage):
        ids = [store.create_run(f"b{i}", "m") for i in range(3)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 3

    def test_get_run_missing_returns_none(self, store: Storage):
        assert store.get_run(999) is None

    def test_finish_run_sets_final_fields(self, store: Storage):
        rid = store.create_run("b", "m")
        store.finish_run(rid, best_iter=3, best_score=87.5, status="converged")
        run = store.get_run(rid)
        assert run["status"] == "converged"
        assert run["best_iter"] == 3
        assert run["best_score"] == 87.5
        assert run["ended_at"] is not None

    def test_finish_run_with_error(self, store: Storage):
        rid = store.create_run("b", "m")
        store.finish_run(rid, 0, -1.0, "errored", error="boom")
        run = store.get_run(rid)
        assert run["status"] == "errored"
        assert run["error"] == "boom"

    def test_finish_clears_progress_fields(self, store: Storage):
        rid = store.create_run("b", "m")
        store.update_progress(rid, 2, "rendering")
        store.finish_run(rid, 1, 50.0, "converged")
        run = store.get_run(rid)
        assert run["current_iter"] is None
        assert run["current_phase"] is None


class TestProgress:
    def test_update_progress_roundtrips(self, store: Storage):
        rid = store.create_run("b", "m")
        store.update_progress(rid, 5, "critiquing")
        run = store.get_run(rid)
        assert run["current_iter"] == 5
        assert run["current_phase"] == "critiquing"

    def test_update_progress_can_clear(self, store: Storage):
        rid = store.create_run("b", "m")
        store.update_progress(rid, 1, "generating")
        store.update_progress(rid, None, None)
        run = store.get_run(rid)
        assert run["current_iter"] is None
        assert run["current_phase"] is None


class TestIterations:
    def test_save_and_list(self, store: Storage):
        rid = store.create_run("b", "m")
        store.save_iteration(_sample_record(rid, 1, composite=40.0))
        store.save_iteration(_sample_record(rid, 2, composite=60.0))
        iters = store.iterations_for_run(rid)
        assert [it["iter"] for it in iters] == [1, 2]
        assert iters[0]["composite_score"] == 40.0
        assert iters[1]["composite_score"] == 60.0

    def test_list_filters_by_after_iter(self, store: Storage):
        rid = store.create_run("b", "m")
        for i in range(1, 5):
            store.save_iteration(_sample_record(rid, i))
        iters = store.iterations_for_run(rid, after_iter=2)
        assert [it["iter"] for it in iters] == [3, 4]

    def test_sus_answers_and_suggestions_deserialize(self, store: Storage):
        rid = store.create_run("b", "m")
        store.save_iteration(_sample_record(rid, 1))
        it = store.iterations_for_run(rid)[0]
        assert it["sus_answers"] == [3] * 10
        assert it["suggestions"] == ["do better"]

    def test_unique_iter_per_run(self, store: Storage):
        rid = store.create_run("b", "m")
        store.save_iteration(_sample_record(rid, 1))
        with pytest.raises(sqlite3.IntegrityError):
            store.save_iteration(_sample_record(rid, 1))

    def test_iterations_isolated_by_run(self, store: Storage):
        r1 = store.create_run("b1", "m")
        r2 = store.create_run("b2", "m")
        store.save_iteration(_sample_record(r1, 1))
        store.save_iteration(_sample_record(r2, 1))
        assert len(store.iterations_for_run(r1)) == 1
        assert len(store.iterations_for_run(r2)) == 1
