"""Tests for the multi-critic ensemble (aggregation + integration)."""

from __future__ import annotations

from pathlib import Path

import pytest

from design_gan.critic import (
    CONTENT_CRITIC,
    CriticProfile,
    SUSResponse,
    TRIO,
    USABILITY_CRITIC,
    VISUAL_CRITIC,
    _aggregate,
    _dedupe_suggestions,
)


class TestCriticProfile:
    def test_trio_has_three_distinct_critics(self):
        assert len(TRIO) == 3
        names = {c.name for c in TRIO}
        assert names == {"Usability", "Visual design", "Content & clarity"}

    def test_each_profile_renders_a_system_prompt(self):
        for p in TRIO:
            prompt = p.system_prompt()
            # Base structure must be present.
            assert "System Usability Scale" in prompt
            # Lens-specific phrasing must land in the prompt.
            assert p.lens.split(".")[0][:30] in prompt
            # Role should appear in the opener.
            assert p.role in prompt


class TestDedupe:
    def test_removes_exact_and_near_duplicates_by_prefix(self):
        # The most common case in practice: two critics surface the same
        # suggestion verbatim. A third says something different.
        sugs = [
            "Fix the 8 color-contrast failures flagged by axe-core.",
            "Fix the 8 color-contrast failures flagged by axe-core.",
            "Add a visible focus indicator.",
        ]
        out = _dedupe_suggestions(sugs)
        assert len(out) == 2

    def test_different_prefix_keeps_both(self):
        # Paraphrases with different leading wording are NOT deduped — the
        # prefix rule is intentionally conservative to avoid losing information.
        sugs = [
            "Fix the 8 color-contrast failures flagged by axe-core.",
            "Resolve all 8 colour-contrast violations the axe-core report found.",
        ]
        out = _dedupe_suggestions(sugs)
        assert len(out) == 2

    def test_preserves_order_of_first_occurrence(self):
        sugs = ["A first thing.", "B second thing.", "A first thing."]
        out = _dedupe_suggestions(sugs)
        assert out == ["A first thing.", "B second thing."]

    def test_caps_at_requested_count(self):
        sugs = [f"Unique suggestion number {i}" for i in range(20)]
        out = _dedupe_suggestions(sugs, cap=5)
        assert len(out) == 5


class TestAggregate:
    def _r(self, sus, feedback="f", suggestions=None):
        return SUSResponse(
            sus=sus, feedback=feedback, suggestions=suggestions or ["s"]
        )

    def test_single_response_pass_through(self):
        r = self._r([3] * 10)
        out = _aggregate([USABILITY_CRITIC], [r])
        assert out.sus == [3] * 10

    def test_mean_rounds_to_int_1_to_5(self):
        # Three critics, all 3s and one 5 on position 0 -> mean 3.67 -> 4
        responses = [
            self._r([3] * 10),
            self._r([3] * 10),
            self._r([5] + [3] * 9),
        ]
        out = _aggregate(TRIO, responses)
        assert out.sus[0] == 4
        assert out.sus[1:] == [3] * 9

    def test_mean_never_exits_1_to_5(self):
        # Extreme values — aggregate must still satisfy the Likert Pydantic
        # constraints (ge=1, le=5) on SUSResponse validation.
        responses = [self._r([5] * 10), self._r([5] * 10), self._r([5] * 10)]
        out = _aggregate(TRIO, responses)
        assert all(1 <= v <= 5 for v in out.sus)

    def test_feedback_carries_critic_names(self):
        out = _aggregate(
            TRIO,
            [
                self._r([3] * 10, feedback="usable enough"),
                self._r([3] * 10, feedback="pretty looking"),
                self._r([3] * 10, feedback="clear copy"),
            ],
        )
        assert "**Usability**: usable enough" in out.feedback
        assert "**Visual design**: pretty looking" in out.feedback
        assert "**Content & clarity**: clear copy" in out.feedback

    def test_suggestions_deduped_across_critics(self):
        shared = "Fix the 8 colour-contrast violations."
        out = _aggregate(
            TRIO,
            [
                self._r([3] * 10, suggestions=[shared, "From critic A"]),
                self._r([3] * 10, suggestions=[shared, "From critic B"]),
                self._r([3] * 10, suggestions=[shared, "From critic C"]),
            ],
        )
        # Dedup should collapse the three copies of `shared` to one.
        assert sum(1 for s in out.suggestions if "colour-contrast" in s) == 1
        assert "From critic A" in out.suggestions


class TestOrchestratorEnsemble:
    """End-to-end: orchestrator with cfg.critics set runs the ensemble."""

    def test_ensemble_persists_per_critic_breakdown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from design_gan import critic as critic_mod
        from design_gan import generator as generator_mod
        from design_gan import orchestrator
        from design_gan import renderer as renderer_mod
        from design_gan.renderer import RenderResult
        from design_gan.storage import Storage

        # Track how many critic calls happen; fake them in parallel.
        call_count = {"n": 0}
        scripted_scores = {
            "Usability": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],          # SUS 50
            "Visual design": [4, 2, 4, 2, 4, 2, 4, 2, 4, 2],       # SUS 75
            "Content & clarity": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],   # SUS 100
        }

        async def fake_generate(model, req):
            return ("<html><body>x</body></html>", 0.01)

        async def fake_render(html, viewport=(1280, 800)):
            return RenderResult(
                screenshot_png=b"\x89PNG\r\n\x1a\nfake",
                dom_html="<html></html>",
                axe_violations=[],
                console_errors=[],
            )

        def fake_write_artifacts(render, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
            return {"screenshot": out_dir / "screenshot.png"}

        async def fake_critique_ensemble(model, profiles, *, screenshot_path,
                                         dom_html, axe_violations, brief):
            call_count["n"] += 1
            responses = [
                critic_mod.SUSResponse(
                    sus=scripted_scores[p.name],
                    feedback=f"{p.name} feedback",
                    suggestions=[f"{p.name} tip 1", "common tip"],
                )
                for p in profiles
            ]
            aggregated = critic_mod._aggregate(profiles, responses)
            breakdown = [
                {"name": p.name, "sus": list(r.sus), "feedback": r.feedback,
                 "suggestions": list(r.suggestions)}
                for p, r in zip(profiles, responses)
            ]
            return aggregated, breakdown, 0.05

        monkeypatch.setattr(generator_mod, "generate", fake_generate)
        monkeypatch.setattr(renderer_mod, "render", fake_render)
        monkeypatch.setattr(renderer_mod, "write_artifacts", fake_write_artifacts)
        monkeypatch.setattr(critic_mod, "critique_ensemble", fake_critique_ensemble)

        cfg = orchestrator.LoopConfig(
            brief="B", runs_dir=tmp_path,
            db_path=tmp_path / "design-gan.sqlite",
            max_iters=1, patience=1,
            critics=list(critic_mod.TRIO),
        )
        result = orchestrator.run_loop_sync(cfg)
        assert call_count["n"] == 1  # ensemble called once for one iter
        # Breakdown stored with all three critics.
        store = Storage(cfg.db_path)
        iters = store.iterations_for_run(result.run_id)
        assert iters[0]["critic_breakdown"] is not None
        names = {c["name"] for c in iters[0]["critic_breakdown"]}
        assert names == {"Usability", "Visual design", "Content & clarity"}
        # Aggregated sus in sus_answers is the mean rounded.
        assert iters[0]["sus_answers"] == [4, 2, 4, 2, 4, 2, 4, 2, 4, 2]
        # Feedback carries all three critic labels.
        for name in names:
            assert f"**{name}**" in iters[0]["feedback"]

    def test_no_critics_config_uses_single_critique(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Regression: when cfg.critics is None, orchestrator must call
        critic.critique (not ensemble) and leave critic_breakdown null."""
        from design_gan import critic as critic_mod
        from design_gan import generator as generator_mod
        from design_gan import orchestrator
        from design_gan import renderer as renderer_mod
        from design_gan.renderer import RenderResult
        from design_gan.storage import Storage

        async def fake_generate(model, req):
            return ("<html></html>", 0.01)

        async def fake_render(html, viewport=(1280, 800)):
            return RenderResult(
                screenshot_png=b"\x89PNG", dom_html="<html></html>",
                axe_violations=[], console_errors=[],
            )

        def fake_write_artifacts(render, out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "screenshot.png").write_bytes(b"\x89PNG")
            return {"screenshot": out_dir / "screenshot.png"}

        ensemble_calls = {"n": 0}

        async def fake_ensemble(*a, **kw):  # should never be called
            ensemble_calls["n"] += 1
            raise AssertionError("ensemble should not run when cfg.critics is None")

        async def fake_critique(model, *, screenshot_path, dom_html,
                                axe_violations, brief):
            return (
                critic_mod.SUSResponse(
                    sus=[3] * 10, feedback="ok", suggestions=["x"]
                ),
                0.01,
            )

        monkeypatch.setattr(generator_mod, "generate", fake_generate)
        monkeypatch.setattr(renderer_mod, "render", fake_render)
        monkeypatch.setattr(renderer_mod, "write_artifacts", fake_write_artifacts)
        monkeypatch.setattr(critic_mod, "critique", fake_critique)
        monkeypatch.setattr(critic_mod, "critique_ensemble", fake_ensemble)

        cfg = orchestrator.LoopConfig(
            brief="B", runs_dir=tmp_path,
            db_path=tmp_path / "design-gan.sqlite",
            max_iters=1, patience=1, critics=None,
        )
        result = orchestrator.run_loop_sync(cfg)
        assert ensemble_calls["n"] == 0

        store = Storage(cfg.db_path)
        iters = store.iterations_for_run(result.run_id)
        assert iters[0].get("critic_breakdown") is None


class TestViewerCriticsFlag:
    """DESIGN_GAN_CRITICS=trio flips the viewer into ensemble mode."""

    def test_config_advertises_ensemble_critics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("DESIGN_GAN_CRITICS", "trio")
        from fastapi.testclient import TestClient
        from design_gan import viewer

        c = TestClient(viewer.app)
        body = c.get("/api/config").json()
        assert body["critics"] == ["Usability", "Visual design", "Content & clarity"]

    def test_config_default_is_solo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        monkeypatch.delenv("DESIGN_GAN_CRITICS", raising=False)
        from fastapi.testclient import TestClient
        from design_gan import viewer

        c = TestClient(viewer.app)
        assert c.get("/api/config").json()["critics"] == ["Usability"]
