"""Integration tests for the orchestrator loop — with generator/critic/renderer faked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from design_gan import orchestrator, storage
from design_gan.critic import SUSResponse
from design_gan.renderer import RenderResult


def _fake_render_artifacts(out_dir: Path) -> None:
    """Write the minimum artifacts the orchestrator expects on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (out_dir / "dom.html").write_text("<html></html>")
    (out_dir / "axe.json").write_text("{}")


class FakeRun:
    """Script the per-iteration behavior of generator/critic/renderer."""

    def __init__(
        self,
        *,
        scores: list[list[int]],
        feedbacks: list[str] | None = None,
        suggestions: list[list[str]] | None = None,
        fail_on: set[int] | None = None,
        fail_phase: str = "generate",
    ):
        self.scores = scores
        self.feedbacks = feedbacks or ["feedback"] * len(scores)
        self.suggestions = suggestions or [["s1"]] * len(scores)
        self.fail_on = fail_on or set()
        self.fail_phase = fail_phase
        self.iter = 0
        self.generated_requests: list[Any] = []
        self.critiqued_briefs: list[str] = []

    def install(
        self, monkeypatch: pytest.MonkeyPatch, *, gen_cost: float = 0.0,
        critic_cost: float = 0.0,
    ) -> None:
        async def fake_generate(model, req):
            self.iter += 1
            self.generated_requests.append(req)
            if self.iter in self.fail_on and self.fail_phase == "generate":
                raise RuntimeError(f"boom generate iter {self.iter}")
            return (
                f"<!doctype html><html><body>iter {self.iter}</body></html>",
                gen_cost,
            )

        async def fake_render(html, viewport=(1280, 800)):
            if self.iter in self.fail_on and self.fail_phase == "render":
                raise RuntimeError(f"boom render iter {self.iter}")
            return RenderResult(
                screenshot_png=b"\x89PNG\r\n\x1a\nfake",
                dom_html="<html></html>",
                axe_violations=[],
                console_errors=[],
            )

        def fake_write_artifacts(render, out_dir):
            _fake_render_artifacts(out_dir)
            return {
                "screenshot": out_dir / "screenshot.png",
                "dom": out_dir / "dom.html",
                "axe": out_dir / "axe.json",
            }

        async def fake_critique(model, *, screenshot_path, dom_html, axe_violations, brief):
            self.critiqued_briefs.append(brief)
            if self.iter in self.fail_on and self.fail_phase == "critique":
                raise RuntimeError(f"boom critique iter {self.iter}")
            idx = self.iter - 1
            return (
                SUSResponse(
                    sus=self.scores[idx],
                    feedback=self.feedbacks[idx],
                    suggestions=self.suggestions[idx],
                ),
                critic_cost,
            )

        monkeypatch.setattr(orchestrator.generator, "generate", fake_generate)
        monkeypatch.setattr(orchestrator.renderer, "render", fake_render)
        monkeypatch.setattr(orchestrator.renderer, "write_artifacts", fake_write_artifacts)
        monkeypatch.setattr(orchestrator.critic, "critique", fake_critique)


@pytest.fixture
def cfg(tmp_path: Path) -> orchestrator.LoopConfig:
    return orchestrator.LoopConfig(
        brief="A cycling tour site.",
        runs_dir=tmp_path,
        db_path=tmp_path / "design-gan.sqlite",
        model="fake-model",
        max_iters=8,
        patience=3,
        tolerance=1.0,
    )


class TestConvergence:
    def test_converges_when_scores_plateau(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        # Iter 1: SUS=100. Subsequent iters: SUS=100 (no gain) -> stale accumulates.
        high = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
        FakeRun(scores=[high] * 5).install(monkeypatch)
        result = orchestrator.run_loop_sync(cfg)
        # First iter sets best=100. Next 3 hit patience -> converged on iter 4.
        assert result.status == "converged"
        assert result.best_score == 100.0
        assert result.best_iter == 1
        assert result.iterations == 4

    def test_exhausted_when_scores_keep_improving(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        # Monotonically rising scores so patience never triggers.
        improving = [[n, 5 - n % 5, n, 5 - n % 5, n, 5 - n % 5, n, 5 - n % 5, n, 5 - n % 5]
                     for n in range(1, 6)]
        # Keep it simple: alternate SUS 30, 50, 70, 90, 100.
        scores = [
            [2, 4, 2, 4, 2, 4, 2, 4, 2, 4],  # SUS 20
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # SUS 50
            [4, 2, 4, 2, 4, 2, 4, 2, 4, 2],  # SUS 75
            [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],  # SUS 100
            [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],  # SUS 100 (no gain)
        ]
        cfg.max_iters = 4  # stop before patience triggers
        FakeRun(scores=scores).install(monkeypatch)
        result = orchestrator.run_loop_sync(cfg)
        assert result.status == "exhausted"
        assert result.best_score == 100.0
        assert result.iterations == 4


class TestErrorHandling:
    def test_single_iter_failure_continues(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        # Iter 1 fails. Iters 2-3 succeed at SUS 50. Iter 4 no-op -> converged on 5.
        scores = [
            [5, 1] * 5,  # (placeholder, iter 1 fails)
            [3, 3] * 5,  # SUS 50
            [3, 3] * 5,  # SUS 50 (stale)
            [3, 3] * 5,  # SUS 50 (stale)
            [3, 3] * 5,  # SUS 50 (stale, converges)
        ]
        FakeRun(scores=scores, fail_on={1}).install(monkeypatch)
        result = orchestrator.run_loop_sync(cfg)
        # Best-score iter is the first successful one (iter 2, SUS 50).
        assert result.best_score == 50.0
        assert result.status == "converged"

    def test_consecutive_failures_trigger_errored_status(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        # Every iter fails -> stale hits patience -> status=errored.
        dummy = [[3] * 10] * 5
        FakeRun(scores=dummy, fail_on={1, 2, 3, 4, 5}).install(monkeypatch)
        result = orchestrator.run_loop_sync(cfg)
        assert result.status == "errored"

    def test_errored_run_does_not_leave_phase_lingering(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        FakeRun(scores=[[3] * 10] * 3, fail_on={1, 2, 3}).install(monkeypatch)
        orchestrator.run_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        runs = store.list_runs()
        assert runs[0]["current_iter"] is None
        assert runs[0]["current_phase"] is None

    def test_all_iters_failing_records_no_sentinel_score(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """When zero iterations complete, best_score should be None, not -1."""
        FakeRun(scores=[[3] * 10] * 5, fail_on={1, 2, 3, 4, 5}).install(monkeypatch)
        result = orchestrator.run_loop_sync(cfg)
        assert result.status == "errored"
        # The persisted run should not show -1 as a score.
        store = storage.Storage(cfg.db_path)
        run = store.list_runs()[0]
        assert run["best_score"] is None or run["best_score"] >= 0
        # LoopResult.best_score should likewise be None or sentinel that callers can detect.
        assert result.best_score is None or result.best_score >= 0


class TestArtifacts:
    def test_writes_site_html_per_iter(
        self, cfg: orchestrator.LoopConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        FakeRun(scores=[[5, 1] * 5] * 4).install(monkeypatch)
        result = orchestrator.run_loop_sync(cfg)
        run_dir = tmp_path / f"run_{result.run_id:04d}"
        assert (run_dir / "iter_001" / "site.html").is_file()

    def test_persists_iterations_with_feedback_and_suggestions(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        FakeRun(
            scores=[[5, 1] * 5],
            feedbacks=["great job"],
            suggestions=[["ship it"]],
        ).install(monkeypatch)
        cfg.max_iters = 1
        cfg.patience = 1
        orchestrator.run_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        iters = store.iterations_for_run(1)
        assert iters[0]["feedback"] == "great job"
        assert iters[0]["suggestions"] == ["ship it"]


class TestBudget:
    """Daily-budget guard aborts the loop before spending more."""

    def test_breaks_loop_when_budget_reached(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.daily_budget_usd = 1.00  # very tight
        cfg.max_iters = 5
        cfg.patience = 5
        # Each iteration costs $0.40 (gen) + $0.20 (critic) = $0.60.
        # Iter 1 spends $0.60 (used $0.60). Iter 2 starts at $0.60 < $1.00
        # so proceeds, spends another $0.60, total $1.20. Iter 3 sees
        # $1.20 >= $1.00 and short-circuits.
        scores = [[5, 1] * 5] * 5
        FakeRun(scores=scores).install(monkeypatch, gen_cost=0.40, critic_cost=0.20)
        result = orchestrator.run_loop_sync(cfg)
        assert result.status == "budget_exhausted"
        # Exactly 2 iters committed to DB.
        store = storage.Storage(cfg.db_path)
        assert len(store.iterations_for_run(result.run_id)) == 2

    def test_no_budget_means_no_check(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.daily_budget_usd = None
        cfg.max_iters = 2
        cfg.patience = 2
        FakeRun(scores=[[5, 1] * 5] * 2).install(
            monkeypatch, gen_cost=5.00, critic_cost=5.00
        )
        result = orchestrator.run_loop_sync(cfg)
        # Both iters ran despite spending $20 — no cap.
        assert result.iterations == 2
        assert result.status in ("converged", "exhausted")

    def test_iter_cost_persisted(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        cfg.max_iters = 1
        cfg.patience = 1
        FakeRun(scores=[[5, 1] * 5]).install(monkeypatch, gen_cost=0.30, critic_cost=0.15)
        orchestrator.run_loop_sync(cfg)
        store = storage.Storage(cfg.db_path)
        iters = store.iterations_for_run(1)
        assert iters[0]["cost_usd"] == pytest.approx(0.45)
        run = store.get_run(1)
        assert run["total_cost_usd"] == pytest.approx(0.45)


class TestFeedbackFlow:
    def test_critic_output_feeds_next_generator_call(
        self, cfg: orchestrator.LoopConfig, monkeypatch: pytest.MonkeyPatch
    ):
        run = FakeRun(
            scores=[[3] * 10, [3] * 10, [3] * 10, [3] * 10],
            feedbacks=["f1", "f2", "f3", "f4"],
            suggestions=[["s1"], ["s2"], ["s3"], ["s4"]],
        )
        run.install(monkeypatch)
        orchestrator.run_loop_sync(cfg)

        # 1st request has no prior feedback; 2nd should have f1+s1 etc.
        reqs = run.generated_requests
        assert reqs[0].prior_html is None
        assert reqs[0].critic_feedback is None
        assert reqs[1].critic_feedback == "f1"
        assert reqs[1].suggestions == ["s1"]
        assert reqs[1].prior_html is not None
