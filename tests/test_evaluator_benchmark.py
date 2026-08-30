from __future__ import annotations

import json
from pathlib import Path

import pytest

from design_gan import browser_evaluator, incumbent_ledger, storage
from design_gan.evaluator_benchmark import (
    ADMISSION_DOMAINS,
    BENCHMARK_CASES,
    BenchmarkCase,
    audit_provenance_corpus,
    balanced_review_candidates,
    capture_run_case,
    case_review,
    corpus_composition,
    load_case_directory,
    proportion_interval,
    review_candidates,
    run_benchmark,
    write_case_fixture,
)


def test_corpus_covers_domains_interactions_and_expected_failures():
    assert {case.domain for case in BENCHMARK_CASES} == {
        "landing-page",
        "lead-generation",
        "storefront",
    }
    assert {case.task.interaction for case in BENCHMARK_CASES} == {"pointer", "keyboard"}
    assert {case.expected_pass for case in BENCHMARK_CASES} == {True, False}


def test_corpus_composition_exposes_concrete_coverage_axes():
    composition = corpus_composition(BENCHMARK_CASES)

    assert composition["cases"] == 22
    assert composition["by_domain"] == {
        "landing-page": 9,
        "lead-generation": 8,
        "storefront": 5,
    }
    assert composition["by_expected_outcome"] == {"fail": 10, "pass": 12}
    assert composition["by_interaction"] == {"keyboard": 6, "pointer": 16}
    assert composition["by_provenance"] == {"built-in": 22}


def test_wilson_interval_does_not_treat_perfect_observation_as_certain():
    interval = proportion_interval(22, 22, 0.95)

    assert interval["estimate"] == 1.0
    assert interval["lower"] == pytest.approx(0.8513, abs=0.0001)
    assert interval["upper"] == pytest.approx(1.0)


@pytest.mark.parametrize("confidence", [0.5, 1.0])
def test_wilson_interval_rejects_invalid_confidence(confidence: float):
    with pytest.raises(ValueError, match="between 0.5 and 1"):
        proportion_interval(1, 1, confidence)


@pytest.mark.asyncio
async def test_report_scores_actor_against_labels(tmp_path: Path):
    async def evaluator(html, *, tasks, trials_per_task):
        passed = "expected-pass" in html
        task = tuple(tasks)[0]
        return browser_evaluator.EvaluationResult(
            score=100.0 if passed else 0.0,
            tasks=[
                browser_evaluator.TaskResult(
                    task_id=task.id,
                    name=task.name,
                    instruction=task.instruction,
                    passed=passed,
                    target="target",
                )
            ],
        )

    cases = (
        BENCHMARK_CASES[0].__class__(
            "pass", "landing-page", BENCHMARK_CASES[0].task, "expected-pass", True
        ),
        BENCHMARK_CASES[0].__class__(
            "fail", "landing-page", BENCHMARK_CASES[0].task, "expected-fail", False
        ),
    )
    report = await run_benchmark(cases, evaluator=evaluator)

    assert report.accuracy == 1.0
    assert report.correct == report.total == 2
    assert report.composition["by_expected_outcome"] == {"fail": 1, "pass": 1}
    output = tmp_path / "benchmark.json"
    report.write(output)
    payload = json.loads(output.read_text())
    assert payload["accuracy"] == 1.0
    assert payload["report_version"] == 2
    assert payload["uncertainty"]["accuracy"]["lower"] < 1.0
    assert "not a random sample" in payload["uncertainty"]["scope"]


def test_operator_labeled_run_case_roundtrips_with_provenance(tmp_path: Path):
    store = storage.Storage(tmp_path / "runs.sqlite")
    task = browser_evaluator.BrowserTask(
        "landing-holdout",
        "Mobile holdout",
        "activate",
        behavior="primary-action",
        split="holdout",
        viewport=(390, 844),
    )
    run_id = store.create_run(
        "Coffee landing page",
        "model",
        domain="landing-page",
        evaluation_suite=[task.to_dict()],
        evaluation_plan={"domain": "landing-page", "tasks": [task.to_dict()]},
    )
    html = "<!doctype html><html><body><button>Start</button></body></html>"
    store.save_iteration(
        storage.IterationRecord(
            run_id=run_id,
            iter=2,
            html=html,
            sus_score=50.0,
            axe_penalty=0.0,
            composite_score=0.0,
            sus_answers=[3] * 10,
            feedback="failed",
            suggestions=[],
            artifacts_dir=str(tmp_path),
        )
    )

    case = capture_run_case(
        store,
        run_id=run_id,
        iteration=2,
        task_id=task.id,
        case_id="real-run-missed-action",
        expected_pass=False,
        reviewer="operator-1",
        rationale="The visible primary action does not produce the required response.",
    )
    fixture = tmp_path / "cases" / "case.json"
    write_case_fixture(case, fixture)
    loaded = load_case_directory(fixture.parent)

    assert loaded[0].html == html
    assert loaded[0].expected_pass is False
    assert loaded[0].task.viewport == (390, 844)
    assert loaded[0].provenance["run_id"] == run_id
    assert len(loaded[0].provenance["artifact_sha256"]) == 64
    assert loaded[0].review is not None
    assert loaded[0].review.reviewer == "operator-1"


def test_review_candidates_group_trials_prioritize_failures_and_mark_captured(tmp_path: Path):
    store = storage.Storage(tmp_path / "runs.sqlite")
    task = browser_evaluator.BrowserTask(
        "landing-primary",
        "Primary action",
        "Activate the primary action.",
        behavior="primary-action",
    )
    run_id = store.create_run(
        "Coffee landing page",
        "model",
        domain="landing-page",
        evaluation_suite=[task.to_dict()],
        evaluation_plan={"domain": "landing-page", "tasks": [task.to_dict()]},
    )
    html = "<!doctype html><html><body><button>Start</button></body></html>"
    common = {
        "run_id": run_id,
        "html": html,
        "sus_score": 50.0,
        "axe_penalty": 0.0,
        "composite_score": 50.0,
        "sus_answers": [3] * 10,
        "feedback": "review",
        "suggestions": [],
        "artifacts_dir": str(tmp_path),
    }
    store.save_iteration(
        storage.IterationRecord(
            iter=1,
            task_results=[
                {**task.to_dict(), "task_id": task.id, "passed": True, "trial": 1},
                {
                    **task.to_dict(),
                    "task_id": task.id,
                    "passed": False,
                    "trial": 2,
                    "errors": ["button did not respond"],
                },
            ],
            **common,
        )
    )
    store.save_iteration(
        storage.IterationRecord(
            iter=2,
            task_results=[{**task.to_dict(), "task_id": task.id, "passed": True, "trial": 1}],
            **common,
        )
    )
    captured = capture_run_case(
        store,
        run_id=run_id,
        iteration=1,
        task_id=task.id,
        case_id="operator-reviewed-failure",
        expected_pass=False,
        reviewer="operator-1",
        rationale="The second recorded trial did not produce a meaningful response.",
    )

    failures = review_candidates(store, captured_cases=(captured,))
    all_outcomes = review_candidates(store, captured_cases=(captured,), failed_only=False)
    legacy = BenchmarkCase(
        id="legacy-operator-label",
        domain=captured.domain,
        task=captured.task,
        html=captured.html,
        expected_pass=captured.expected_pass,
        provenance=captured.provenance,
    )
    legacy_candidates = review_candidates(store, captured_cases=(legacy,))
    balanced = balanced_review_candidates(store, captured_cases=(captured,))

    assert len(failures) == 1
    assert failures[0].observed_pass is False
    assert (failures[0].passed_trials, failures[0].total_trials) == (1, 2)
    assert failures[0].errors == ("button did not respond",)
    assert failures[0].captured_case_ids == ("operator-reviewed-failure",)
    assert failures[0].audited_case_ids == ("operator-reviewed-failure",)
    assert failures[0].suggested_case_id == "run-1-iter-1-landing-primary"
    assert [candidate.iteration for candidate in all_outcomes] == [2, 1]
    assert legacy_candidates[0].captured_case_ids == ("legacy-operator-label",)
    assert legacy_candidates[0].audited_case_ids == ()
    assert [candidate.iteration for candidate in balanced] == [2]


def test_balanced_review_candidates_stratify_outcomes_and_domains(tmp_path: Path):
    store = storage.Storage(tmp_path / "runs.sqlite")
    expected = []
    for observed_pass in (False, True):
        for domain in ADMISSION_DOMAINS:
            suffix = "pass" if observed_pass else "fail"
            task = browser_evaluator.BrowserTask(
                f"{domain}-{suffix}",
                f"{domain} {suffix}",
                "Perform the frozen behavior.",
                behavior="primary-action",
            )
            run_id = store.create_run(
                f"{domain} {suffix}",
                "model",
                domain=domain,
                evaluation_suite=[task.to_dict()],
                evaluation_plan={"domain": domain, "tasks": [task.to_dict()]},
            )
            store.save_iteration(
                storage.IterationRecord(
                    run_id=run_id,
                    iter=1,
                    html=(
                        "<!doctype html><html><body><button>"
                        f"{domain}-{suffix}</button></body></html>"
                    ),
                    sus_score=50.0,
                    axe_penalty=0.0,
                    composite_score=50.0,
                    sus_answers=[3] * 10,
                    feedback="review",
                    suggestions=[],
                    artifacts_dir=str(tmp_path),
                    task_results=[
                        {
                            **task.to_dict(),
                            "task_id": task.id,
                            "passed": observed_pass,
                            "trial": 1,
                        }
                    ],
                )
            )
            expected.append((domain, observed_pass))

    candidates = balanced_review_candidates(store, limit=6)

    assert [(candidate.domain, candidate.observed_pass) for candidate in candidates] == expected


def test_balanced_review_candidates_deduplicate_evidence_and_cap_runs(tmp_path: Path):
    store = storage.Storage(tmp_path / "runs.sqlite")
    task = browser_evaluator.BrowserTask(
        "landing-primary",
        "Primary action",
        "Activate the primary action.",
        behavior="primary-action",
    )
    run_id = store.create_run(
        "Landing page",
        "model",
        domain="landing-page",
        evaluation_suite=[task.to_dict()],
        evaluation_plan={"domain": "landing-page", "tasks": [task.to_dict()]},
    )
    for iteration, label in enumerate(("same", "same", "different"), start=1):
        store.save_iteration(
            storage.IterationRecord(
                run_id=run_id,
                iter=iteration,
                html=f"<!doctype html><html><body><button>{label}</button></body></html>",
                sus_score=50.0,
                axe_penalty=0.0,
                composite_score=0.0,
                sus_answers=[3] * 10,
                feedback="review",
                suggestions=[],
                artifacts_dir=str(tmp_path),
                task_results=[{**task.to_dict(), "task_id": task.id, "passed": False, "trial": 1}],
            )
        )

    candidates = balanced_review_candidates(store, limit=10, max_candidates_per_run=2)

    assert [candidate.iteration for candidate in candidates] == [3, 2]


def _admission_cases() -> tuple[BenchmarkCase, ...]:
    cases = []
    for domain_number, domain in enumerate(ADMISSION_DOMAINS, start=1):
        for index in range(8):
            task_id = f"{domain}-review-{index}"
            task = browser_evaluator.BrowserTask(
                task_id,
                f"{domain} task {index}",
                "Perform the frozen behavior.",
                behavior="primary-action",
            )
            html = f"<!doctype html><html><body><button>{domain}-{index}</button></body></html>"
            run_id = domain_number * 10 + index % 3
            cases.append(
                BenchmarkCase(
                    id=f"{domain}-review-{index}",
                    domain=domain,
                    task=task,
                    html=html,
                    expected_pass=index % 2 == 0,
                    provenance={
                        "source": "design-gan-run",
                        "run_id": run_id,
                        "iteration": index + 1,
                        "task_id": task_id,
                        "artifact_sha256": incumbent_ledger.artifact_hash(html),
                    },
                    review=case_review(
                        "operator-1",
                        "The frozen task outcome is unambiguous in the inspected artifact.",
                        reviewed_at=1.0,
                    ),
                )
            )
    return tuple(cases)


def test_provenance_corpus_admission_requires_balanced_diverse_reviewed_evidence():
    report = audit_provenance_corpus(_admission_cases())

    assert report.ready is True
    assert report.policy_version == 1
    assert report.qualifying_cases == 24
    assert report.by_domain == {domain: 8 for domain in ADMISSION_DOMAINS}
    assert report.by_label == {"fail": 12, "pass": 12}
    assert report.distinct_runs_by_domain == {domain: 3 for domain in ADMISSION_DOMAINS}
    assert report.blockers == ()


def test_provenance_corpus_admission_excludes_unaudited_and_duplicate_evidence():
    cases = list(_admission_cases())
    first = cases[0]
    cases[0] = BenchmarkCase(
        id=first.id,
        domain=first.domain,
        task=first.task,
        html=first.html,
        expected_pass=first.expected_pass,
        provenance=first.provenance,
    )
    cases.append(
        BenchmarkCase(
            id="duplicate-provenance",
            domain=cases[1].domain,
            task=cases[1].task,
            html=cases[1].html,
            expected_pass=cases[1].expected_pass,
            provenance=cases[1].provenance,
            review=cases[1].review,
        )
    )

    report = audit_provenance_corpus(cases)

    assert report.ready is False
    assert report.qualifying_cases == 23
    assert {item["id"] for item in report.excluded_cases} == {
        first.id,
        "duplicate-provenance",
    }
    assert any("need at least 24" in blocker for blocker in report.blockers)
    assert "auditable operator review is missing" in report.excluded_cases[0]["reasons"]


def test_captured_fixture_loader_reports_invalid_task_as_validation_error(tmp_path: Path):
    fixture = tmp_path / "bad-task.json"
    payload = BENCHMARK_CASES[0].to_fixture()
    payload["task"] = {"id": "missing-required-fields"}
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid evaluator task"):
        load_case_directory(tmp_path)


def test_captured_fixture_loader_rejects_incomplete_review_metadata(tmp_path: Path):
    fixture = tmp_path / "bad-review.json"
    payload = BENCHMARK_CASES[0].to_fixture()
    payload["review"] = {
        "reviewer": "operator-1",
        "rationale": "The artifact behavior was inspected directly.",
    }
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reviewed_at is required"):
        load_case_directory(tmp_path)
