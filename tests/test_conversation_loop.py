"""End-to-end tests for run_conversation_loop with all agents faked."""

from __future__ import annotations

from pathlib import Path

import pytest

from design_gan import (
    conversation_generator,
    critic as critic_mod,
    orchestrator,
    storage,
    transcript_renderer,
    user_simulator,
)
from design_gan.critic import SUSResponse


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scores_per_iter: list[list[int]],
    satisfied: bool = True,
    objective_penalty: float = 0.0,
    objective_metrics: dict | None = None,
):
    """Wire up a fully-scripted conversation loop."""
    counter = {"iter": 0}

    async def fake_gen(model, req):
        counter["iter"] += 1
        return f"System prompt for iter {counter['iter']}", 0.01

    async def fake_run_conversation(*, model, assistant_system_prompt, goal, max_turns):
        return transcript_renderer.TranscriptResult(
            transcript=[
                {"role": "user", "content": "hi", "cost_usd": 0.0},
                {"role": "assistant", "content": f"reply to goal: {goal}",
                 "cost_usd": 0.01},
            ],
            assistant_system_prompt=assistant_system_prompt,
            objective_metrics=objective_metrics or {
                "assistant_turn_count": 1, "unresolved": not satisfied,
                "boilerplate_count": 0, "repetition_hits": [],
                "length_bloat_hits": [],
            },
            objective_penalty=objective_penalty,
            total_cost_usd=0.02,
            satisfied=satisfied,
            turns_taken=1,
        )

    def fake_write(result, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transcript.json").write_text("{}")
        (out_dir / "metrics.json").write_text("{}")
        (out_dir / "system_prompt.txt").write_text(result.assistant_system_prompt)
        return {
            "transcript": out_dir / "transcript.json",
            "metrics": out_dir / "metrics.json",
            "system_prompt": out_dir / "system_prompt.txt",
        }

    async def fake_cus_critique(model, *, goal, transcript, objective_metrics,
                                assistant_system_prompt):
        idx = counter["iter"] - 1
        scores = scores_per_iter[idx] if idx < len(scores_per_iter) else [3] * 10
        return SUSResponse(
            sus=scores, feedback=f"feedback for iter {counter['iter']}",
            suggestions=[f"suggestion {counter['iter']}"],
        ), 0.01

    monkeypatch.setattr(conversation_generator, "generate", fake_gen)
    monkeypatch.setattr(transcript_renderer, "run_conversation", fake_run_conversation)
    monkeypatch.setattr(transcript_renderer, "write_transcript_artifacts", fake_write)
    monkeypatch.setattr(critic_mod, "cus_critique", fake_cus_critique)


@pytest.fixture
def cfg(tmp_path: Path) -> orchestrator.LoopConfig:
    return orchestrator.LoopConfig(
        brief="Teach me cold brew basics",
        runs_dir=tmp_path,
        db_path=tmp_path / "design-gan.sqlite",
        max_iters=5, patience=3, tolerance=1.0,
    )


class TestConversationLoopBasics:
    def test_kind_is_conversation(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        # Single successful iter then plateau.
        _install_fakes(monkeypatch, scores_per_iter=[[4, 2] * 5] * 5)
        result = orchestrator.run_conversation_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        run = store.get_run(result.run_id)
        assert run["kind"] == "conversation"

    def test_patience_triggers_convergence(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        # First iter sets a score; next 3 are equal -> stale hits patience.
        _install_fakes(
            monkeypatch, scores_per_iter=[[5, 1] * 5] * 5,  # all 100
        )
        result = orchestrator.run_conversation_loop_sync(cfg)
        assert result.status == "converged"
        assert result.best_score == 100.0

    def test_satisfied_vs_unsatisfied_affects_penalty(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.max_iters = 1
        cfg.patience = 1
        _install_fakes(
            monkeypatch,
            scores_per_iter=[[3] * 10],  # SUS = 50
            satisfied=False,
            objective_penalty=5.0,  # matches unresolved weight
        )
        orchestrator.run_conversation_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        it = store.iterations_for_run(1)[0]
        assert it["axe_penalty"] == 5.0
        assert it["composite_score"] == 45.0


class TestConversationPersistence:
    def test_iteration_stores_transcript_artifacts_on_disk(
        self, cfg: orchestrator.LoopConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
    ):
        cfg.max_iters = 1
        cfg.patience = 1
        _install_fakes(monkeypatch, scores_per_iter=[[5, 1] * 5])
        result = orchestrator.run_conversation_loop_sync(cfg)
        iter_dir = tmp_path / f"run_{result.run_id:04d}" / "iter_001"
        assert (iter_dir / "transcript.json").is_file()
        assert (iter_dir / "metrics.json").is_file()
        assert (iter_dir / "system_prompt.txt").is_file()

    def test_html_column_stores_system_prompt(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.max_iters = 1
        cfg.patience = 1
        _install_fakes(monkeypatch, scores_per_iter=[[3] * 10])
        orchestrator.run_conversation_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        it = store.iterations_for_run(1)[0]
        # The html column is semantically overloaded for conversation runs:
        # it holds the assistant's system prompt (the evolving artifact).
        assert "System prompt for iter 1" in it["html"]

    def test_cost_rolls_up(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.max_iters = 1
        cfg.patience = 1
        _install_fakes(monkeypatch, scores_per_iter=[[3] * 10])
        orchestrator.run_conversation_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        run = store.get_run(1)
        # gen 0.01 + conversation 0.02 + critic 0.01 = 0.04
        assert run["total_cost_usd"] == pytest.approx(0.04)


class TestEnsembleInConversationLoop:
    def test_cus_trio_called_when_critics_set(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.max_iters = 1
        cfg.patience = 1
        cfg.critics = list(critic_mod.CUS_TRIO)

        ensemble_calls = {"n": 0}

        async def fake_gen(model, req):
            return "system prompt", 0.01

        async def fake_run_conversation(*, model, assistant_system_prompt, goal, max_turns):
            return transcript_renderer.TranscriptResult(
                transcript=[{"role": "user", "content": "q", "cost_usd": 0.0},
                            {"role": "assistant", "content": "a", "cost_usd": 0.01}],
                assistant_system_prompt=assistant_system_prompt,
                objective_metrics={"assistant_turn_count": 1, "unresolved": False,
                                   "boilerplate_count": 0, "repetition_hits": [],
                                   "length_bloat_hits": []},
                objective_penalty=0.0, total_cost_usd=0.01,
                satisfied=True, turns_taken=1,
            )

        def fake_write(result, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            return {"transcript": out_dir / "transcript.json",
                    "metrics": out_dir / "metrics.json",
                    "system_prompt": out_dir / "system_prompt.txt"}

        async def fake_ensemble(model, profiles, **kw):
            ensemble_calls["n"] += 1
            aggregated = SUSResponse(sus=[4, 2] * 5, feedback="agg", suggestions=["s"])
            breakdown = [
                {"name": p.name, "sus": [4] * 10, "feedback": p.name + " fb",
                 "suggestions": ["t"]} for p in profiles
            ]
            return aggregated, breakdown, 0.03

        async def fake_solo(*a, **kw):
            raise AssertionError("solo path should not run with cfg.critics set")

        monkeypatch.setattr(conversation_generator, "generate", fake_gen)
        monkeypatch.setattr(transcript_renderer, "run_conversation", fake_run_conversation)
        monkeypatch.setattr(transcript_renderer, "write_transcript_artifacts", fake_write)
        monkeypatch.setattr(critic_mod, "cus_critique_ensemble", fake_ensemble)
        monkeypatch.setattr(critic_mod, "cus_critique", fake_solo)

        result = orchestrator.run_conversation_loop_sync(cfg)
        assert ensemble_calls["n"] == 1

        store = storage.Storage(cfg.db_path)
        it = store.iterations_for_run(result.run_id)[0]
        assert it["critic_breakdown"] is not None
        assert {c["name"] for c in it["critic_breakdown"]} == {
            "Conversation usability", "Tone & register", "Trust & specificity"
        }


class TestCreateRunKind:
    def test_default_is_design(self, tmp_path: Path):
        s = storage.Storage(tmp_path / "db.sqlite")
        rid = s.create_run("b", "m")
        assert s.get_run(rid)["kind"] == "design"

    def test_explicit_conversation_kind(self, tmp_path: Path):
        s = storage.Storage(tmp_path / "db.sqlite")
        rid = s.create_run("g", "m", kind="conversation")
        assert s.get_run(rid)["kind"] == "conversation"

    def test_migration_backfills_design(self, tmp_path: Path):
        import sqlite3
        # Pre-create a runs table without the kind column (old schema).
        db = tmp_path / "old.sqlite"
        with sqlite3.connect(db) as c:
            c.execute("""CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brief TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL,
                ended_at REAL, best_iter INTEGER, best_score REAL,
                status TEXT NOT NULL DEFAULT 'running'
            )""")
            c.execute("INSERT INTO runs(brief, model, created_at) VALUES ('b', 'm', 0.0)")
            c.commit()
        s = storage.Storage(db)
        assert s.get_run(1)["kind"] == "design"
