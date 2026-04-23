"""HTTP tests for the FastAPI viewer, running against a demo-seeded DB."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from design_gan.demo import seed_demo


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
    # Import after env var so the viewer picks up the tmp dir.
    from design_gan import viewer

    seed_demo(tmp_path)
    return TestClient(viewer.app)


class TestIndex:
    def test_index_returns_html(self, client: TestClient):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "design-gan" in r.text

    def test_index_shows_seeded_run(self, client: TestClient):
        r = client.get("/")
        assert "DEMO: A landing page" in r.text
        # Seed run composite peak is 75.5 from the real run, rendered as "76".
        assert ">76<" in r.text or '">76<' in r.text

    def test_index_sidebar_lists_runs(self, client: TestClient):
        r = client.get("/")
        assert 'class="side-item' in r.text


class TestRunDetail:
    def test_known_run_renders(self, client: TestClient):
        r = client.get("/runs/1")
        assert r.status_code == 200
        assert "Run #1" in r.text
        # 4 iteration cards.
        assert r.text.count('class="iter-card"') == 4

    def test_unknown_run_is_404(self, client: TestClient):
        r = client.get("/runs/999")
        assert r.status_code == 404

    def test_detail_exposes_running_flag_attr(self, client: TestClient):
        # Seed run is converged -> data-running="0".
        r = client.get("/runs/1")
        assert 'data-running="0"' in r.text


class TestArtifactRoutes:
    def test_screenshot_served(self, client: TestClient):
        r = client.get("/runs/1/iters/1/screenshot")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_site_html_served(self, client: TestClient):
        r = client.get("/runs/1/iters/1/site")
        assert r.status_code == 200
        assert "<html" in r.text.lower() or "<body" in r.text.lower()

    def test_missing_screenshot_is_404(self, client: TestClient):
        r = client.get("/runs/1/iters/99/screenshot")
        assert r.status_code == 404

    def test_missing_run_screenshot_is_404(self, client: TestClient):
        r = client.get("/runs/999/iters/1/screenshot")
        assert r.status_code == 404


class TestStatic:
    def test_serves_static_asset(self, client: TestClient):
        r = client.get("/static/style.css")
        assert r.status_code == 200
        assert "iter-card" in r.text

    def test_static_traversal_rejected(self, client: TestClient):
        # Path traversal via substring check.
        r = client.get("/static/..%2F..%2Fetc%2Fpasswd")
        assert r.status_code in (404, 400)

    def test_static_absolute_path_rejected(self, client: TestClient, tmp_path: Path):
        # Absolute URL-encoded path — Path() joining an absolute operand
        # silently escapes the static dir unless we resolve and clamp it.
        import os
        # Use a real file outside the static dir and try to fetch it.
        outside = tmp_path / "secret.txt"
        outside.write_text("SHOULD NOT BE SERVED")
        from urllib.parse import quote
        r = client.get(f"/static/{quote(str(outside.resolve()))}")
        assert r.status_code in (404, 400)
        assert "SHOULD NOT BE SERVED" not in r.text

    def test_static_missing_is_404(self, client: TestClient):
        r = client.get("/static/does-not-exist.txt")
        assert r.status_code == 404


class TestJsonApi:
    def test_api_runs_list(self, client: TestClient):
        r = client.get("/api/runs")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["status"] == "converged"

    def test_api_run_detail_shape(self, client: TestClient):
        r = client.get("/api/runs/1")
        assert r.status_code == 200
        data = r.json()
        assert "run" in data and "iterations" in data
        assert len(data["iterations"]) == 4
        it = data["iterations"][0]
        assert set(["iter", "composite_score", "feedback", "suggestions"]).issubset(it)

    def test_api_unknown_run_is_404(self, client: TestClient):
        r = client.get("/api/runs/999")
        assert r.status_code == 404


class TestStartRunValidation:
    def test_missing_brief_rejected(self, client: TestClient):
        r = client.post("/api/runs", json={})
        assert r.status_code == 422

    def test_empty_brief_rejected(self, client: TestClient):
        r = client.post("/api/runs", json={"brief": ""})
        assert r.status_code == 422

    def test_oversized_max_iters_rejected(self, client: TestClient):
        r = client.post(
            "/api/runs", json={"brief": "x", "max_iters": 1000}
        )
        assert r.status_code == 422


class TestStartTokenGate:
    """When DESIGN_GAN_START_TOKEN is set, /api/runs rejects unauthenticated POSTs."""

    @pytest.fixture
    def gated_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> TestClient:
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("DESIGN_GAN_START_TOKEN", "s3cret")
        from design_gan import viewer

        seed_demo(tmp_path)
        return TestClient(viewer.app)

    def test_config_reports_gate(self, gated_client: TestClient):
        r = gated_client.get("/api/config")
        assert r.status_code == 200
        assert r.json()["requires_token"] is True

    def test_config_reports_no_gate_by_default(self, client: TestClient):
        r = client.get("/api/config")
        assert r.json()["requires_token"] is False

    def test_missing_token_rejected(self, gated_client: TestClient):
        r = gated_client.post("/api/runs", json={"brief": "x"})
        assert r.status_code == 401

    def test_wrong_token_rejected(self, gated_client: TestClient):
        r = gated_client.post(
            "/api/runs", json={"brief": "x", "token": "nope"}
        )
        assert r.status_code == 401

    def test_correct_body_token_accepted(
        self, gated_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        # Prevent the orchestrator from actually being called.
        from design_gan import orchestrator
        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *a, **kw: None)
        r = gated_client.post(
            "/api/runs", json={"brief": "x", "token": "s3cret"}
        )
        assert r.status_code == 200
        assert "run_id" in r.json()

    def test_bearer_header_accepted(
        self, gated_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        from design_gan import orchestrator
        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *a, **kw: None)
        r = gated_client.post(
            "/api/runs",
            json={"brief": "x"},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert r.status_code == 200

    def test_form_shows_token_field_when_gated(self, gated_client: TestClient):
        r = gated_client.get("/")
        assert 'name="token"' in r.text
        assert "requires a shared token" in r.text

    def test_form_hides_token_field_by_default(self, client: TestClient):
        r = client.get("/")
        assert 'name="token"' not in r.text

    def test_browsing_history_still_open_when_gated(self, gated_client: TestClient):
        assert gated_client.get("/").status_code == 200
        assert gated_client.get("/api/runs").status_code == 200
        assert gated_client.get("/runs/1").status_code == 200


class TestBudgetGate:
    """Rejects POST /api/runs when the 24h spend has hit the cap."""

    @pytest.fixture
    def budgeted_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> TestClient:
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("DESIGN_GAN_DAILY_BUDGET_USD", "1.00")
        from design_gan import viewer
        seed_demo(tmp_path)
        return TestClient(viewer.app)

    def test_config_reports_budget(self, budgeted_client: TestClient):
        r = budgeted_client.get("/api/config").json()
        assert r["daily_budget_usd"] == 1.0
        assert r["budget_used_24h_usd"] == 0.0
        assert r["budget_remaining_usd"] == 1.0

    def test_over_budget_returns_429(
        self, budgeted_client: TestClient, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Force a big recent cost into the DB by inserting a synthetic iter.
        from design_gan.storage import IterationRecord, Storage
        store = Storage(tmp_path / "design-gan.sqlite")
        store.save_iteration(IterationRecord(
            run_id=1, iter=99, html="<html></html>",
            sus_score=0.0, axe_penalty=0.0, composite_score=0.0,
            sus_answers=[3]*10, feedback="f", suggestions=["s"],
            artifacts_dir=str(tmp_path), cost_usd=2.50,
        ))
        r = budgeted_client.post("/api/runs", json={"brief": "x"})
        assert r.status_code == 429
        body = r.json()["detail"]
        assert body["error"] == "daily_budget_exhausted"

    def test_under_budget_proceeds(
        self, budgeted_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        from design_gan import orchestrator
        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *a, **kw: None)
        r = budgeted_client.post("/api/runs", json={"brief": "x"})
        assert r.status_code == 200

    def test_no_budget_means_no_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        # budget env unset
        from design_gan import viewer, orchestrator
        seed_demo(tmp_path)
        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *a, **kw: None)
        c = TestClient(viewer.app)
        assert c.get("/api/config").json()["daily_budget_usd"] is None
        r = c.post("/api/runs", json={"brief": "x"})
        assert r.status_code == 200


class TestBootSweep:
    def test_startup_clears_running_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        # Pre-seed a stuck "running" row before the app boots.
        from design_gan.storage import Storage
        store = Storage(tmp_path / "design-gan.sqlite")
        rid = store.create_run("ghost", "m")
        from design_gan import viewer
        with TestClient(viewer.app) as client:
            # Opening TestClient triggers the startup event.
            pass
        # The ghost run should now be errored.
        run = store.get_run(rid)
        assert run["status"] == "errored"
        assert "abandoned" in (run["error"] or "")


class TestConversationRoutes:
    """Routes specific to conversation runs: transcript JSON + styled view."""

    @pytest.fixture
    def convo_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> TestClient:
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        import json as _json
        from design_gan import viewer
        from design_gan.storage import IterationRecord, Storage

        # Seed a conversation run with a transcript artifact on disk.
        store = Storage(tmp_path / "design-gan.sqlite")
        rid = store.create_run("how to make cold brew", "m", kind="conversation")
        iter_dir = tmp_path / f"run_{rid:04d}" / "iter_001"
        iter_dir.mkdir(parents=True, exist_ok=True)
        transcript = {
            "transcript": [
                {"role": "user", "content": "How do I make cold brew?"},
                {"role": "assistant", "content": "1:4 coffee to water, steep 18h cold."},
            ],
            "satisfied": True,
            "turns_taken": 1,
            "total_cost_usd": 0.04,
        }
        (iter_dir / "transcript.json").write_text(_json.dumps(transcript))
        (iter_dir / "system_prompt.txt").write_text("Be concrete.")
        store.save_iteration(IterationRecord(
            run_id=rid, iter=1, html="Be concrete.", sus_score=75.0,
            axe_penalty=0.0, composite_score=75.0, sus_answers=[4, 2]*5,
            feedback="good", suggestions=["tighten"], artifacts_dir=str(iter_dir),
            cost_usd=0.04,
        ))
        store.finish_run(rid, 1, 75.0, "converged")

        return TestClient(viewer.app)

    def test_transcript_json_served(self, convo_client: TestClient):
        r = convo_client.get("/runs/1/iters/1/transcript")
        assert r.status_code == 200
        assert "cold brew" in r.text
        assert r.headers["content-type"].startswith("application/json")

    def test_transcript_view_renders_bubbles(self, convo_client: TestClient):
        r = convo_client.get("/runs/1/iters/1/transcript-view")
        assert r.status_code == 200
        assert "bubble-user" in r.text
        assert "bubble-assistant" in r.text
        assert "How do I make cold brew?" in r.text
        assert "1:4 coffee to water" in r.text

    def test_transcript_view_missing_is_404(self, convo_client: TestClient):
        r = convo_client.get("/runs/1/iters/99/transcript-view")
        assert r.status_code == 404

    def test_run_detail_uses_conversation_card(self, convo_client: TestClient):
        r = convo_client.get("/runs/1")
        assert r.status_code == 200
        # Conversation card shows transcript preview, not <img src=screenshot>.
        assert "thumb-transcript" in r.text
        assert 'data-kind="conversation"' in r.text
        # And CUS label instead of SUS.
        assert "CUS" in r.text


class TestStartRunKindBranching:
    """POST /api/runs with kind=conversation must route through the conversation loop."""

    def test_kind_conversation_uses_conversation_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        from design_gan import orchestrator, viewer

        calls = {"design": 0, "conversation": 0}
        monkeypatch.setattr(
            orchestrator, "run_loop_sync",
            lambda *a, **kw: calls.__setitem__("design", calls["design"] + 1),
        )
        monkeypatch.setattr(
            orchestrator, "run_conversation_loop_sync",
            lambda *a, **kw: calls.__setitem__("conversation", calls["conversation"] + 1),
        )

        c = TestClient(viewer.app)
        r = c.post("/api/runs", json={"brief": "x", "kind": "conversation"})
        assert r.status_code == 200
        # Backgrounded via asyncio.create_task; give it a beat.
        import time
        time.sleep(0.2)
        assert calls["conversation"] == 1
        assert calls["design"] == 0

    def test_kind_design_still_uses_design_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        from design_gan import orchestrator, viewer

        calls = {"design": 0, "conversation": 0}
        monkeypatch.setattr(
            orchestrator, "run_loop_sync",
            lambda *a, **kw: calls.__setitem__("design", calls["design"] + 1),
        )
        monkeypatch.setattr(
            orchestrator, "run_conversation_loop_sync",
            lambda *a, **kw: calls.__setitem__("conversation", calls["conversation"] + 1),
        )

        c = TestClient(viewer.app)
        r = c.post("/api/runs", json={"brief": "x"})
        assert r.status_code == 200
        import time
        time.sleep(0.2)
        assert calls["design"] == 1
        assert calls["conversation"] == 0


class TestErroredRunDisplay:
    """A run that never completed an iteration shows '—' for best-score, not -1."""

    def test_best_score_none_renders_as_em_dash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        from design_gan import viewer
        from design_gan.storage import Storage

        store = Storage(tmp_path / "design-gan.sqlite")
        rid = store.create_run("failed run", "m")
        store.finish_run(rid, None, None, "errored", error="oops")

        c = TestClient(viewer.app)
        r = c.get(f"/runs/{rid}")
        assert r.status_code == 200
        # Should not see "-1" appearing as a legitimate score.
        assert ">-1<" not in r.text
        assert "—" in r.text
