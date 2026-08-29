from __future__ import annotations

import json
from pathlib import Path

import pytest

from design_gan import browser_evaluator, storage
from design_gan.evaluator_benchmark import (
    BENCHMARK_CASES,
    capture_run_case,
    load_case_directory,
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
    output = tmp_path / "benchmark.json"
    report.write(output)
    assert '"accuracy": 1.0' in output.read_text()


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
    )
    fixture = tmp_path / "cases" / "case.json"
    write_case_fixture(case, fixture)
    loaded = load_case_directory(fixture.parent)

    assert loaded[0].html == html
    assert loaded[0].expected_pass is False
    assert loaded[0].task.viewport == (390, 844)
    assert loaded[0].provenance["run_id"] == run_id
    assert len(loaded[0].provenance["artifact_sha256"]) == 64


def test_captured_fixture_loader_reports_invalid_task_as_validation_error(tmp_path: Path):
    fixture = tmp_path / "bad-task.json"
    payload = BENCHMARK_CASES[0].to_fixture()
    payload["task"] = {"id": "missing-required-fields"}
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid evaluator task"):
        load_case_directory(tmp_path)
