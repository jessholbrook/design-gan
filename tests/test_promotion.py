from __future__ import annotations

import pytest

from design_gan.promotion import decide, one_sided_sign_test


def _results(passed: int, total: int = 6) -> list[dict]:
    return [
        {"task_id": "primary-action", "trial": trial, "passed": trial <= passed}
        for trial in range(1, total + 1)
    ]


def test_exact_one_sided_sign_test():
    assert one_sided_sign_test(6, 0) == pytest.approx(1 / 64)
    assert one_sided_sign_test(4, 0) == pytest.approx(1 / 16)
    assert one_sided_sign_test(3, 3) == 1.0


def test_first_eligible_candidate_is_promoted():
    result = decide(
        candidate_score=50,
        candidate_eligible=True,
        candidate_results=_results(3),
        baseline_score=None,
        baseline_results=None,
        minimum_effect=1,
        alpha=0.05,
    )
    assert result.promoted is True
    assert result.reason == "initial_eligible_candidate"


def test_guardrail_blocks_before_significance():
    result = decide(
        candidate_score=100,
        candidate_eligible=False,
        candidate_results=_results(6),
        baseline_score=0,
        baseline_results=_results(0),
        minimum_effect=1,
        alpha=0.05,
    )
    assert result.promoted is False
    assert result.reason == "blocked_by_guardrail"


def test_six_paired_wins_clear_default_alpha():
    result = decide(
        candidate_score=100,
        candidate_eligible=True,
        candidate_results=_results(6),
        baseline_score=0,
        baseline_results=_results(0),
        minimum_effect=1,
        alpha=0.05,
    )
    assert result.promoted is True
    assert result.reason == "significant_improvement"
    assert result.p_value == pytest.approx(1 / 64)


def test_four_paired_wins_do_not_clear_default_alpha():
    result = decide(
        candidate_score=100,
        candidate_eligible=True,
        candidate_results=_results(4, 4),
        baseline_score=0,
        baseline_results=_results(0, 4),
        minimum_effect=1,
        alpha=0.05,
    )
    assert result.promoted is False
    assert result.reason == "not_significant"
    assert result.p_value == pytest.approx(1 / 16)
