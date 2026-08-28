from __future__ import annotations

from pathlib import Path

import pytest

from design_gan import browser_evaluator
from design_gan.evaluator_benchmark import BENCHMARK_CASES, run_benchmark


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
