"""Scoring: SUS answers -> 0-100 score; composite blends SUS with objective penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Score:
    sus: float  # 0-100, pure SUS
    axe_penalty: float  # 0-100, subtracted from composite
    composite: float  # 0-100, blended
    breakdown: dict[str, Any]


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
