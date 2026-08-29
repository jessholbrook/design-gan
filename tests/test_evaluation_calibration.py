from __future__ import annotations

import pytest

from design_gan import browser_evaluator
from design_gan.evaluation_calibration import (
    majority_error_probability,
    minimum_discordant_wins,
    recommend_trials,
    run_calibration,
)
from design_gan.evaluator_benchmark import BENCHMARK_CASES


def test_default_alpha_needs_five_discordant_wins():
    assert minimum_discordant_wins(0.05) == 5
    assert recommend_trials(0.0, 0.05) == 5


def test_observed_flakes_raise_the_recommended_odd_trial_count():
    assert majority_error_probability(5, 0.2) > 0.05
    assert majority_error_probability(7, 0.2) < 0.05
    assert recommend_trials(0.2, 0.05) == 7


@pytest.mark.asyncio
async def test_calibration_records_mismatches_flakes_and_recommendation():
    async def evaluator(html, *, tasks, trials_per_task):
        task = tuple(tasks)[0]
        return browser_evaluator.EvaluationResult(
            score=80.0,
            tasks=[
                browser_evaluator.TaskResult(
                    task_id=task.id,
                    name=task.name,
                    instruction=task.instruction,
                    passed=trial != 3,
                    target="target",
                    trial=trial,
                )
                for trial in range(1, trials_per_task + 1)
            ],
        )

    report = await run_calibration((BENCHMARK_CASES[0],), repetitions=5, evaluator=evaluator)

    assert report.attempts == 5
    assert report.mismatches == 1
    assert report.max_flake_rate == pytest.approx(0.2)
    assert report.unstable_cases == 1
    assert report.recommended_trials == 7
    payload = report.to_dict()
    assert payload["report_version"] == 2
    assert payload["composition"]["cases"] == 1
    assert payload["uncertainty"]["accuracy"]["estimate"] == pytest.approx(0.8)
    assert payload["uncertainty"]["unstable_case_rate"]["estimate"] == 1.0


@pytest.mark.parametrize("repetitions", [1, 21])
@pytest.mark.asyncio
async def test_calibration_rejects_uninformative_repetition_counts(repetitions: int):
    with pytest.raises(ValueError, match="between 2 and 20"):
        await run_calibration((), repetitions=repetitions)


@pytest.mark.asyncio
async def test_calibration_rejects_invalid_confidence_level():
    with pytest.raises(ValueError, match="between 0.5 and 1"):
        await run_calibration((), confidence_level=1.0)
