"""HTTP tests for the FastAPI viewer, running against a demo-seeded DB."""

from __future__ import annotations

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


def _seed_review_candidate() -> tuple[int, str]:
    from design_gan import browser_evaluator, storage, viewer

    task = browser_evaluator.BrowserTask(
        id="landing-review-task",
        name="Primary action",
        instruction="Activate the primary action.",
        behavior="primary-action",
    )
    run_id = viewer._store().create_run(
        "Review this generated landing page",
        "model",
        domain="landing-page",
        evaluation_suite=[task.to_dict()],
        evaluation_plan={"domain": "landing-page", "tasks": [task.to_dict()]},
    )
    private_html = (
        "<!doctype html><html><body><button>Private generated content</button></body></html>"
    )
    viewer._store().save_iteration(
        storage.IterationRecord(
            run_id=run_id,
            iter=1,
            html=private_html,
            sus_score=50.0,
            axe_penalty=0.0,
            composite_score=0.0,
            sus_answers=[3] * 10,
            feedback="Primary action did not respond.",
            suggestions=[],
            artifacts_dir="",
            task_results=[
                {
                    "task_id": task.id,
                    "name": task.name,
                    "instruction": task.instruction,
                    "passed": False,
                    "trial": 1,
                    "errors": ["no response observed"],
                }
            ],
        )
    )
    return run_id, private_html


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

    def test_new_run_form_has_valid_alpha_default_and_help_for_every_term(self, client: TestClient):
        page = client.get("/").text

        assert 'name="promotion_alpha" value="0.05" step="0.0001" min="0.0001" max="1"' in page
        assert page.count('class="info-tip"') == 11
        assert page.count('tabindex="0" role="note" aria-label="Help:') == 11
        assert page.count("aria-labelledby=") == 11
        assert page.count("aria-describedby=") == 11
        assert "Maximum one-sided sign-test p-value allowed for promotion" in page
        assert "Minimum task-score improvement, in percentage points" in page
        assert 'data-form-term="brief"' in page


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

    def test_running_detail_shows_phase_weighted_progress(self, client: TestClient):
        from design_gan import viewer

        run_id = viewer._store().create_run("Progress demo", "model", max_iters=12)
        viewer._store().update_progress(run_id, 3, "evaluating")

        page = client.get(f"/runs/{run_id}").text

        assert 'data-running="1"' in page
        assert 'class="status-beacon"' in page
        assert 'id="progress-indicator" class="run-progress"' in page
        assert 'style="display:grid" data-max-iters="12"' in page
        assert "Evaluating frozen tasks" in page
        assert "Iteration 3 of 12 · stage 3 of 4" in page
        assert 'aria-valuenow="23"' in page
        assert 'id="progress-bar" style="width:23%"' in page
        assert 'class="progress-step is-active" data-phase="evaluating"' in page
        assert 'data-phase-started-at="' in page
        assert 'id="progress-elapsed"' in page

    def test_design_evaluation_is_structured_instead_of_wall_of_text(self, client: TestClient):
        from design_gan import storage, viewer

        run_id = viewer._store().create_run("Structured evaluation", "model")
        viewer._store().save_iteration(
            storage.IterationRecord(
                run_id=run_id,
                iter=1,
                html="<html><body><button>Go</button></body></html>",
                sus_score=72.5,
                axe_penalty=10.0,
                composite_score=50.0,
                sus_answers=[3] * 10,
                feedback=(
                    "Behavioral task completion: 1/2.\n"
                    "Diagnostic SUS feedback (not the primary score): Clear hierarchy, "
                    "but the action needs a stronger response."
                ),
                suggestions=["Make the action visibly update the page."],
                artifacts_dir="",
                primary_score=50.0,
                primary_metric="task_completion_rate",
                promotion_eligible=False,
                promoted=False,
                task_results=[
                    {
                        "task_id": "primary",
                        "name": "Primary action works",
                        "passed": True,
                        "trial": 1,
                    },
                    {
                        "task_id": "primary",
                        "name": "Primary action works",
                        "passed": False,
                        "trial": 2,
                        "observed": ["activation produced no observable response"],
                    },
                ],
                guardrails={
                    "accessibility": {
                        "passed": False,
                        "blocking_violations": [{"id": "color-contrast", "nodes": 4}],
                    },
                    "correctness": {"passed": True, "errors": []},
                    "artifact_boundary": {"passed": True, "violations": []},
                },
            )
        )

        page = client.get(f"/runs/{run_id}").text

        assert 'class="evaluation-metrics"' in page
        assert "1/2" in page
        assert "Primary action works" in page
        assert "activation produced no observable response" in page
        assert "color-contrast (4 node(s))" in page
        assert "Runtime correctness" in page
        assert "<summary>SUS diagnostic feedback</summary>" in page
        assert "Clear hierarchy, but the action needs a stronger response." in page
        assert "Behavioral task completion: 1/2." not in page


class TestScrub:
    def test_run_detail_links_to_scrub(self, client: TestClient):
        r = client.get("/runs/1")
        assert 'href="/runs/1/scrub"' in r.text

    def test_scrub_renders(self, client: TestClient):
        r = client.get("/runs/1/scrub")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        # Shell scaffold + the hydrating script must be present.
        assert 'class="scrub"' in r.text
        assert 'id="scrub-stage"' in r.text
        assert "/static/scrub.js" in r.text

    def test_scrub_carries_run_and_kind_attrs(self, client: TestClient):
        r = client.get("/runs/1/scrub")
        assert 'data-scrub-run-id="1"' in r.text
        # Seed run is a design run.
        assert 'data-kind="design"' in r.text

    def test_scrub_unknown_run_is_404(self, client: TestClient):
        r = client.get("/runs/999/scrub")
        assert r.status_code == 404


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

    def test_site_html_is_sandboxed(self, client: TestClient):
        """Generated HTML is untrusted (LLM-authored, brief-shaped). The CSP
        sandbox gives it an opaque origin so its scripts can't read this
        origin's localStorage (start token) or call the API as our origin."""
        r = client.get("/runs/1/iters/1/site")
        assert r.headers.get("content-security-policy") == "sandbox allow-scripts"

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
        assert "progress-wave" in r.text
        assert "prefers-reduced-motion" in r.text

    def test_serves_scrub_js(self, client: TestClient):
        r = client.get("/static/scrub.js")
        assert r.status_code == 200
        assert "scrub-stage" in r.text

    def test_serves_evaluator_review_js(self, client: TestClient):
        r = client.get("/static/evaluator-review.js")
        assert r.status_code == 200
        assert "/api/evaluator-cases" in r.text
        assert "design_gan_reviewer_id" in r.text

    def test_app_js_updates_the_brief_term_without_replacing_its_help(self, client: TestClient):
        r = client.get("/static/app.js")

        assert r.status_code == 200
        assert "querySelector('[data-form-term=\"brief\"]')" in r.text
        assert "briefTerm.textContent" in r.text
        assert "firstChild.nodeValue" not in r.text

    def test_app_js_updates_live_progress_and_reconnect_state(self, client: TestClient):
        r = client.get("/static/app.js")

        assert r.status_code == 200
        assert "phaseLabels" in r.text
        assert "progressBar.style.width" in r.text
        assert "Reconnecting to live progress" in r.text
        assert "Starting run" in r.text
        assert "payload.card_html" in r.text
        assert "Stage active for" in r.text

    def test_static_traversal_rejected(self, client: TestClient):
        # Path traversal via substring check.
        r = client.get("/static/..%2F..%2Fetc%2Fpasswd")
        assert r.status_code in (404, 400)

    def test_static_absolute_path_rejected(self, client: TestClient, tmp_path: Path):
        # Absolute URL-encoded path — Path() joining an absolute operand
        # silently escapes the static dir unless we resolve and clamp it.
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
        # The scrubber (static/scrub.js) reads every one of these per iteration;
        # this locks the JSON contract it depends on.
        assert {
            "iter",
            "composite_score",
            "sus_score",
            "axe_penalty",
            "sus_answers",
            "feedback",
            "suggestions",
            "primary_score",
            "primary_metric",
            "promotion_eligible",
            "guardrails",
            "task_results",
            "artifact_validation",
            "parent_iter",
            "promoted",
            "promotion_reason",
            "promotion_effect",
            "promotion_p_value",
        }.issubset(it)

    def test_config_lists_versioned_design_domains(self, client: TestClient):
        domains = client.get("/api/config").json()["design_domains"]
        assert {domain["id"] for domain in domains} == {
            "landing-page",
            "lead-generation",
            "storefront",
        }
        assert {domain["id"]: domain["version"] for domain in domains} == {
            "landing-page": 3,
            "lead-generation": 3,
            "storefront": 2,
        }

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
        r = client.post("/api/runs", json={"brief": "x", "max_iters": 1000})
        assert r.status_code == 422

    def test_unknown_design_domain_rejected(self, client: TestClient):
        r = client.post("/api/runs", json={"brief": "x", "design_domain": "checkout"})
        assert r.status_code == 422

    def test_invalid_trial_count_rejected(self, client: TestClient):
        r = client.post("/api/runs", json={"brief": "x", "evaluation_trials": 0})
        assert r.status_code == 422

    def test_selected_domain_and_promotion_policy_are_frozen_on_run(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        from design_gan import orchestrator

        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *args, **kwargs: None)
        response = client.post(
            "/api/runs",
            json={
                "brief": "Collect sales leads",
                "design_domain": "lead-generation",
                "evaluation_trials": 8,
                "max_iters": 12,
                "promotion_alpha": 0.1,
                "optimization_key": "product:sales",
            },
        )

        assert response.status_code == 200
        run_id = response.json()["run_id"]
        run = client.get(f"/api/runs/{run_id}").json()["run"]
        assert run["domain"] == "lead-generation"
        tasks = run["evaluation_plan"]["tasks"]
        assert [task["id"] for task in tasks] == [
            "lead-form-desktop",
            "lead-form-mobile",
            "lead-form-keyboard-holdout",
            "lead-form-mobile-keyboard-holdout",
        ]
        assert [task["split"] for task in tasks] == [
            "development",
            "development",
            "holdout",
            "holdout",
        ]
        assert run["evaluation_plan"]["trials_per_task"] == 8
        assert run["evaluation_plan"]["promotion_alpha"] == pytest.approx(0.1)
        assert run["optimization_key"] == "product:sales"
        assert run["max_iters"] == 12
        assert "product:sales" in client.get(f"/runs/{run_id}").text


class TestStartTokenGate:
    """When DESIGN_GAN_START_TOKEN is set, /api/runs rejects unauthenticated POSTs."""

    @pytest.fixture
    def gated_client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
        r = gated_client.post("/api/runs", json={"brief": "x", "token": "nope"})
        assert r.status_code == 401

    def test_correct_body_token_accepted(
        self, gated_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        # Prevent the orchestrator from actually being called.
        from design_gan import orchestrator

        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *a, **kw: None)
        r = gated_client.post("/api/runs", json={"brief": "x", "token": "s3cret"})
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        run = gated_client.get(f"/api/runs/{run_id}").json()["run"]
        assert [task["id"] for task in run["evaluation_suite"]] == [
            "landing-primary-desktop",
            "landing-primary-keyboard",
            "landing-primary-mobile-holdout",
            "landing-primary-mobile-keyboard-holdout",
        ]
        assert run["evaluation_plan"]["trials_per_task"] == 5
        assert run["evaluation_plan"]["promotion_alpha"] == pytest.approx(0.05)
        assert run["artifact_policy"]["kind"] == "standalone-html"
        assert run["artifact_policy"]["network_access"] is False
        detail = gated_client.get(f"/runs/{run_id}").text
        assert "Frozen browser tasks" in detail
        assert "best task score" in detail

    def test_final_holdout_status_is_visible_in_run_and_scrubber(self, gated_client: TestClient):
        from design_gan import viewer

        viewer._store().save_holdout_audit(
            1,
            score=50.0,
            passed=False,
            results={"score": 50.0, "audited_iter": 4},
        )
        contract = {
            "optimization_key": "product:demo",
            "domain": "landing-page",
            "domain_version": 2,
            "evaluator_version": 4,
            "artifact_policy_version": 1,
        }
        viewer._store().set_run_optimization_key(1, contract["optimization_key"])
        viewer._store().resolve_incumbent_challenge(
            run_id=1,
            contract=contract,
            prior_incumbent_id=None,
            outcome="established",
            evidence={
                "decision": {"outcome": "established"},
                "arbitration_conflicts": [{"attempt": 1}],
            },
            candidate_iter=1,
            candidate_html="<html>demo</html>",
            candidate_artifact_hash="hash",
            candidate_primary_score=100.0,
            candidate_holdout_score=100.0,
            candidate_holdout_results={"score": 100.0},
        )
        detail = gated_client.get("/runs/1")
        scrub_js = gated_client.get("/static/scrub.js")
        assert "Final untouched holdout: FAIL" in detail.text
        assert "Final untouched holdout" in scrub_js.text
        assert "Cross-run challenge: established" in detail.text
        assert "concurrent retries 1" in detail.text
        assert "Cross-run challenge" in scrub_js.text
        incumbents = gated_client.get("/api/incumbents").json()
        assert incumbents[0]["optimization_key"] == "product:demo"
        assert "html" not in incumbents[0]

    def test_evaluator_case_api_omits_captured_html(self, gated_client: TestClient):
        from design_gan import browser_evaluator, evaluator_benchmark, viewer

        case = evaluator_benchmark.BenchmarkCase(
            id="captured-demo-case",
            domain="landing-page",
            task=browser_evaluator.BrowserTask(
                id="landing-primary-desktop",
                name="Primary action",
                instruction="Activate the primary action.",
                behavior="primary-action",
            ),
            html="<html><body><button>Private generated content</button></body></html>",
            expected_pass=False,
            provenance={"source": "design-run", "run_id": 7, "iteration": 2},
        )
        evaluator_benchmark.write_case_fixture(
            case,
            viewer._runs_dir() / "evaluator-corpus" / "captured-demo-case.json",
        )

        response = gated_client.get("/api/evaluator-cases")

        assert response.status_code == 200
        assert response.json() == [
            {
                "id": "captured-demo-case",
                "domain": "landing-page",
                "task_id": "landing-primary-desktop",
                "expected_pass": False,
                "provenance": {"source": "design-run", "run_id": 7, "iteration": 2},
            }
        ]
        assert "Private generated content" not in response.text

    def test_review_queue_exposes_metadata_and_page_without_html(self, gated_client: TestClient):
        run_id, private_html = _seed_review_candidate()

        response = gated_client.get("/api/evaluator-case-candidates")
        blinded = gated_client.get("/api/evaluator-review-queue")
        page = gated_client.get("/evaluator-review")
        diagnostics = gated_client.get("/evaluator-review?show_observation=true")

        assert response.status_code == 200
        assert response.json()[0]["run_id"] == run_id
        assert response.json()[0]["task_id"] == "landing-review-task"
        assert response.json()[0]["observed_pass"] is False
        assert response.json()[0]["site_url"] == f"/runs/{run_id}/iters/1/site"
        assert private_html not in response.text
        assert blinded.status_code == 200
        assert blinded.json()["sampling_policy_version"] == 1
        assert blinded.json()["maximum_candidates_per_run"] == 4
        assert blinded.json()["blinded"] is True
        assert blinded.json()["items"][0]["run_id"] == run_id
        assert "observed_pass" not in blinded.json()["items"][0]
        assert "passed_trials" not in blinded.json()["items"][0]
        assert "errors" not in blinded.json()["items"][0]
        assert "captured_case_ids" not in blinded.json()["items"][0]
        assert private_html not in blinded.text
        assert page.status_code == 200
        assert "independently label whether the frozen task" in page.text
        assert "hidden for independent labeling" in page.text
        assert "Evaluator observed" not in page.text
        assert "no response observed" not in page.text
        assert "landing-review-task" in page.text
        assert "/static/evaluator-review.js" in page.text
        assert "Actor-comparison readiness" in page.text
        assert "BLOCKED" in page.text
        assert 'id="reviewer-id"' in page.text
        assert private_html not in page.text
        assert "Evaluator observed" in diagnostics.text
        assert "no response observed" in diagnostics.text

    def test_review_page_rejects_unknown_sampling_mode(self, gated_client: TestClient):
        response = gated_client.get("/evaluator-review?mode=unknown")

        assert response.status_code == 422

    def test_capture_api_requires_token_writes_fixture_and_rejects_duplicate(
        self, gated_client: TestClient
    ):
        run_id, private_html = _seed_review_candidate()
        payload = {
            "run_id": run_id,
            "iteration": 1,
            "task_id": "landing-review-task",
            "case_id": "reviewed-primary-failure",
            "expected_pass": False,
            "reviewer": "operator-1",
            "rationale": "The primary button produces no meaningful visible response.",
        }

        assert gated_client.post("/api/evaluator-cases", json=payload).status_code == 401
        response = gated_client.post(
            "/api/evaluator-cases",
            json=payload,
            headers={"Authorization": "Bearer s3cret"},
        )
        duplicate = gated_client.post(
            "/api/evaluator-cases",
            json=payload,
            headers={"Authorization": "Bearer s3cret"},
        )

        assert response.status_code == 201
        assert response.json()["id"] == "reviewed-primary-failure"
        assert response.json()["provenance"]["source"] == "design-gan-run"
        assert "review" not in response.json()
        assert "operator-1" not in response.text
        assert private_html not in response.text
        assert duplicate.status_code == 409
        listed = gated_client.get("/api/evaluator-cases").json()
        assert listed[0]["id"] == "reviewed-primary-failure"
        candidates = gated_client.get("/api/evaluator-case-candidates").json()
        assert candidates[0]["captured_case_ids"] == ["reviewed-primary-failure"]
        assert candidates[0]["audited_case_ids"] == ["reviewed-primary-failure"]
        readiness = gated_client.get("/api/evaluator-corpus-readiness").json()
        assert readiness["ready"] is False
        assert readiness["qualifying_cases"] == 1
        assert readiness["requirements"]["minimum_cases"] == 24

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
        assert "Shared write token required to start runs" in r.text

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
    def budgeted_client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
        self,
        budgeted_client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Force a big recent cost into the DB by inserting a synthetic iter.
        from design_gan.storage import IterationRecord, Storage

        store = Storage(tmp_path / "design-gan.sqlite")
        store.save_iteration(
            IterationRecord(
                run_id=1,
                iter=99,
                html="<html></html>",
                sus_score=0.0,
                axe_penalty=0.0,
                composite_score=0.0,
                sus_answers=[3] * 10,
                feedback="f",
                suggestions=["s"],
                artifacts_dir=str(tmp_path),
                cost_usd=2.50,
            )
        )
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

    def test_no_budget_means_no_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        # budget env unset
        from design_gan import orchestrator, viewer

        seed_demo(tmp_path)
        monkeypatch.setattr(orchestrator, "run_loop_sync", lambda *a, **kw: None)
        c = TestClient(viewer.app)
        assert c.get("/api/config").json()["daily_budget_usd"] is None
        r = c.post("/api/runs", json={"brief": "x"})
        assert r.status_code == 200


class TestBootSweep:
    def test_startup_clears_running_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DESIGN_GAN_RUNS_DIR", str(tmp_path))
        # Pre-seed a stuck "running" row before the app boots.
        from design_gan.storage import Storage

        store = Storage(tmp_path / "design-gan.sqlite")
        rid = store.create_run("ghost", "m")
        from design_gan import viewer

        with TestClient(viewer.app):
            # Opening TestClient triggers the startup event.
            pass
        # The ghost run should now be errored.
        run = store.get_run(rid)
        assert run["status"] == "errored"
        assert "abandoned" in (run["error"] or "")


class TestConversationRoutes:
    """Routes specific to conversation runs: transcript JSON + styled view."""

    @pytest.fixture
    def convo_client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
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
        store.save_iteration(
            IterationRecord(
                run_id=rid,
                iter=1,
                html="Be concrete.",
                sus_score=75.0,
                axe_penalty=0.0,
                composite_score=75.0,
                sus_answers=[4, 2] * 5,
                feedback="good",
                suggestions=["tighten"],
                artifacts_dir=str(iter_dir),
                cost_usd=0.04,
            )
        )
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
            orchestrator,
            "run_loop_sync",
            lambda *a, **kw: calls.__setitem__("design", calls["design"] + 1),
        )
        monkeypatch.setattr(
            orchestrator,
            "run_conversation_loop_sync",
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
            orchestrator,
            "run_loop_sync",
            lambda *a, **kw: calls.__setitem__("design", calls["design"] + 1),
        )
        monkeypatch.setattr(
            orchestrator,
            "run_conversation_loop_sync",
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
