from __future__ import annotations

from design_gan.browser_evaluator import EvaluationResult, TaskResult
from design_gan.incumbent_ledger import decide_challenge, optimization_key


def _evaluation(passed: int, total: int = 5) -> EvaluationResult:
    return EvaluationResult(
        score=100 * passed / total,
        tasks=[
            TaskResult(
                task_id="holdout",
                name="Holdout",
                instruction="exercise",
                passed=trial <= passed,
                target="target",
                trial=trial,
                split="holdout",
            )
            for trial in range(1, total + 1)
        ],
    )


def test_brief_derived_product_key_is_normalized_and_stable():
    assert optimization_key("  COFFEE   shop ") == optimization_key("coffee shop")
    assert optimization_key("coffee shop").startswith("brief:")
    assert optimization_key("ignored", "product:coffee") == "product:coffee"


def test_first_verified_candidate_establishes_incumbent():
    decision = decide_challenge(
        candidate=_evaluation(5),
        candidate_passed=True,
        incumbent=None,
        minimum_effect=1.0,
        alpha=0.05,
    )
    assert decision.outcome == "established"
    assert decision.installs_candidate is True


def test_failed_candidate_is_not_admitted_without_an_incumbent():
    decision = decide_challenge(
        candidate=_evaluation(4),
        candidate_passed=False,
        incumbent=None,
        minimum_effect=1.0,
        alpha=0.05,
    )
    assert decision.outcome == "rejected_holdout"


def test_five_paired_holdout_wins_replace_incumbent():
    decision = decide_challenge(
        candidate=_evaluation(5),
        candidate_passed=True,
        incumbent=_evaluation(0),
        minimum_effect=1.0,
        alpha=0.05,
    )
    assert decision.outcome == "replaced"
    assert decision.promotion is not None
    assert decision.promotion.p_value == 0.03125


def test_tied_verified_candidate_retains_existing_incumbent():
    decision = decide_challenge(
        candidate=_evaluation(5),
        candidate_passed=True,
        incumbent=_evaluation(5),
        minimum_effect=1.0,
        alpha=0.05,
    )
    assert decision.outcome == "retained"
    assert decision.installs_candidate is False
