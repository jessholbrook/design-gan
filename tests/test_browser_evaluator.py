"""Contract tests for the concrete v2 browser task evaluator."""

from __future__ import annotations

import pytest

from design_gan.browser_evaluator import (
    DEFAULT_DESIGN_TASKS,
    BrowserTask,
    EvaluationResult,
    TaskResult,
    _candidate_score,
    frozen_suite,
)


def test_default_suite_is_frozen_and_concrete():
    suite = frozen_suite()
    assert suite == DEFAULT_DESIGN_TASKS
    assert isinstance(suite, tuple)
    assert [task.id for task in suite] == ["primary-action"]


def test_unsupported_task_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError, match="unsupported"):
        frozen_suite([BrowserTask("generic-script", "Anything", "do anything")])


def test_empty_explicit_suite_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        frozen_suite([])


def test_form_completion_is_a_supported_concrete_task():
    task = BrowserTask("form-completion", "Complete form", "submit it")
    assert frozen_suite([task]) == (task,)


def test_explicit_primary_marker_outranks_generic_button():
    generic = {
        "tag": "button",
        "text": "Menu",
        "width": 100,
        "height": 40,
        "dataPrimary": None,
    }
    primary = {
        "tag": "a",
        "text": "Book now",
        "width": 100,
        "height": 40,
        "dataPrimary": "",
        "href": "#book",
    }
    assert _candidate_score(primary) > _candidate_score(generic)


def test_result_exposes_task_rate_and_correctness_separately():
    result = EvaluationResult(
        score=100.0,
        tasks=[
            TaskResult(
                task_id="primary-action",
                name="Primary action works",
                instruction="activate it",
                passed=True,
                target="Book now",
                observed=["page scrolled to new content"],
            )
        ],
        correctness_errors=[],
    )
    payload = result.to_dict()
    assert payload["primary_metric"] == "task_completion_rate"
    assert payload["passed"] == payload["total"] == 1
    assert "1/1" in result.feedback()


def test_feedback_groups_repeated_trials_by_task():
    result = EvaluationResult(
        score=50.0,
        tasks=[
            TaskResult(
                task_id="primary-action",
                name="Primary action works",
                instruction="activate it",
                passed=trial == 1,
                target="Start",
                trial=trial,
                observed=["visible content changed"] if trial == 1 else ["no response"],
            )
            for trial in (1, 2)
        ],
    )
    assert "1/2 trials passed" in result.feedback()
