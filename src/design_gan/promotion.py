"""Candidate promotion decisions for behavioral design evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    effect: float
    p_value: float | None
    comparable_trials: int
    wins: int
    losses: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def one_sided_sign_test(wins: int, losses: int) -> float:
    """Exact P(X >= wins) for X~Binomial(wins+losses, 0.5)."""
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    n = wins + losses
    if n == 0 or wins <= losses:
        return 1.0
    numerator = sum(math.comb(n, k) for k in range(wins, n + 1))
    return numerator / (2**n)


def _outcomes(results: list[dict[str, Any]] | None) -> dict[tuple[str, int], bool]:
    return {
        (str(item.get("task_id")), int(item.get("trial", 1))): bool(item.get("passed"))
        for item in (results or [])
    }


def decide(
    *,
    candidate_score: float,
    candidate_eligible: bool,
    candidate_results: list[dict[str, Any]] | None,
    baseline_score: float | None,
    baseline_results: list[dict[str, Any]] | None,
    minimum_effect: float,
    alpha: float,
) -> PromotionDecision:
    """Return an auditable promotion decision for one design candidate."""
    effect = candidate_score - baseline_score if baseline_score is not None else candidate_score
    if not candidate_eligible:
        return PromotionDecision(False, "blocked_by_guardrail", effect, None, 0, 0, 0)
    if baseline_score is None:
        return PromotionDecision(True, "initial_eligible_candidate", effect, None, 0, 0, 0)
    if effect < minimum_effect:
        return PromotionDecision(False, "effect_below_minimum", effect, None, 0, 0, 0)

    candidate = _outcomes(candidate_results)
    baseline = _outcomes(baseline_results)
    shared = sorted(candidate.keys() & baseline.keys())
    if not shared:
        # Legacy baselines have no per-trial evidence. Preserve upgradeability,
        # but make the fallback explicit in history.
        return PromotionDecision(True, "legacy_baseline_without_trials", effect, None, 0, 0, 0)

    wins = sum(candidate[key] and not baseline[key] for key in shared)
    losses = sum(baseline[key] and not candidate[key] for key in shared)
    p_value = one_sided_sign_test(wins, losses)
    promoted = wins > losses and p_value <= alpha
    return PromotionDecision(
        promoted=promoted,
        reason="significant_improvement" if promoted else "not_significant",
        effect=effect,
        p_value=p_value,
        comparable_trials=len(shared),
        wins=wins,
        losses=losses,
    )
