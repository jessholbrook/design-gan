"""Product scoring and promotion policy.

Design runs use browser-task completion as their one primary metric.  SUS is
retained as a diagnostic, while accessibility and runtime correctness are hard
promotion guardrails.  Conversation runs keep their existing CUS-minus-penalty
score until they receive a domain-specific behavioral evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Score:
    sus: float  # 0-100, pure SUS
    axe_penalty: float  # diagnostic for design; objective penalty for conversation
    composite: float  # backward-compatible storage/viewer value; primary score for v2 design
    breakdown: dict[str, Any]
    primary_metric: str = "legacy_composite"
    promotion_eligible: bool = True
    guardrails: dict[str, Any] | None = None


# Impact weights for axe violations.
_IMPACT_WEIGHT = {"critical": 5.0, "serious": 3.0, "moderate": 1.5, "minor": 0.5}


def sus_score(answers: list[int]) -> float:
    """Standard SUS scoring: odd items (x-1), even items (5-x), sum * 2.5 -> 0-100."""
    if len(answers) != 10:
        raise ValueError(f"SUS requires exactly 10 answers, got {len(answers)}")
    total = 0
    for i, x in enumerate(answers):
        if not 1 <= x <= 5:
            raise ValueError(f"SUS answer at position {i} out of range: {x}")
        total += (x - 1) if i % 2 == 0 else (5 - x)
    return round(total * 2.5, 2)


def axe_penalty(violations: list[dict[str, Any]]) -> float:
    """Sum weighted violations, cap at 30 so a11y can't dominate the composite."""
    penalty = 0.0
    for v in violations:
        weight = _IMPACT_WEIGHT.get(v.get("impact") or "", 0.5)
        nodes = max(1, len(v.get("nodes", [])))
        penalty += weight * nodes
    return min(penalty, 30.0)


def score(sus_answers: list[int], axe_violations: list[dict[str, Any]]) -> Score:
    """Legacy SUS-minus-axe score retained for old callers and stored runs."""
    base = sus_score(sus_answers)
    penalty = axe_penalty(axe_violations)
    composite = max(0.0, min(100.0, base - penalty))
    return Score(
        sus=base,
        axe_penalty=penalty,
        composite=round(composite, 2),
        breakdown={
            "sus_answers": sus_answers,
            "axe_violation_count": len(axe_violations),
        },
    )


def design_score(
    sus_answers: list[int],
    axe_violations: list[dict[str, Any]],
    *,
    task_score: float,
    task_results: list[dict[str, Any]],
    axe_error: str | None,
    console_errors: list[str],
    evaluator_errors: list[str],
    artifact_validation: dict[str, Any] | None = None,
) -> Score:
    """Score a design candidate and apply its promotion guardrails.

    The task completion rate is never blended with SUS or axe.  An iteration
    may have a high task score and still be ineligible for promotion, which is
    intentionally visible rather than hidden inside a weighted average.
    """
    diagnostic_sus = sus_score(sus_answers)
    diagnostic_axe = axe_penalty(axe_violations)
    blocking_impacts = {"critical", "serious"}
    blocking_violations = [
        {
            "id": violation.get("id"),
            "impact": violation.get("impact"),
            "nodes": len(violation.get("nodes", [])),
        }
        for violation in axe_violations
        if (violation.get("impact") or "").lower() in blocking_impacts
    ]
    accessibility_passed = axe_error is None and not blocking_violations
    correctness_errors = [*console_errors, *evaluator_errors]
    correctness_passed = not correctness_errors
    artifact_passed = (
        bool(artifact_validation.get("passed")) if artifact_validation is not None else True
    )
    guardrails = {
        "accessibility": {
            "passed": accessibility_passed,
            "axe_error": axe_error,
            "blocking_violations": blocking_violations,
        },
        "correctness": {
            "passed": correctness_passed,
            "errors": correctness_errors,
        },
        "artifact_boundary": artifact_validation
        or {"passed": True, "violations": [], "legacy": True},
    }
    primary = max(0.0, min(100.0, float(task_score)))
    return Score(
        sus=diagnostic_sus,
        axe_penalty=diagnostic_axe,
        composite=round(primary, 2),
        primary_metric="task_completion_rate",
        promotion_eligible=accessibility_passed and correctness_passed and artifact_passed,
        guardrails=guardrails,
        breakdown={
            "task_results": task_results,
            "sus_answers": sus_answers,
            "axe_violation_count": len(axe_violations),
        },
    )


def score_from_penalty(
    answers: list[int], penalty: float, breakdown: dict[str, Any] | None = None
) -> Score:
    """Shared path for SUS/CUS: takes answers + a pre-computed penalty.

    Conversation runs use this directly — their objective penalty is computed
    in transcript_renderer and doesn't look like axe violations.
    """
    base = sus_score(answers)
    capped_penalty = min(max(penalty, 0.0), 30.0)
    composite = max(0.0, min(100.0, base - capped_penalty))
    return Score(
        sus=base,
        axe_penalty=capped_penalty,
        composite=round(composite, 2),
        breakdown={"answers": list(answers), **(breakdown or {})},
    )
