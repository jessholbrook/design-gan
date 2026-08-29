"""Cross-run incumbent contracts and final holdout challenge decisions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

from . import browser_evaluator, promotion

_KEY_WHITESPACE = re.compile(r"\s+")


def optimization_key(brief: str, explicit: str | None = None) -> str:
    """Return an explicit product key or a stable key for the normalized brief."""
    if explicit is not None:
        key = explicit.strip()
        if not key:
            raise ValueError("optimization_key cannot be blank")
        if len(key) > 160:
            raise ValueError("optimization_key must be at most 160 characters")
        return key
    normalized = _KEY_WHITESPACE.sub(" ", brief.strip().lower())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"brief:{digest}"


def artifact_hash(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LedgerContract:
    optimization_key: str
    domain: str
    domain_version: int
    evaluator_version: int
    artifact_policy_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChallengeDecision:
    outcome: str
    promotion: promotion.PromotionDecision | None

    @property
    def installs_candidate(self) -> bool:
        return self.outcome in {"established", "replaced"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "promotion": self.promotion.to_dict() if self.promotion else None,
        }


def decide_challenge(
    *,
    candidate: browser_evaluator.EvaluationResult,
    candidate_passed: bool,
    incumbent: browser_evaluator.EvaluationResult | None,
    minimum_effect: float,
    alpha: float,
) -> ChallengeDecision:
    """Decide a final-only challenge without exposing holdout evidence to search."""
    if incumbent is None:
        return ChallengeDecision(
            "established" if candidate_passed else "rejected_holdout",
            None,
        )
    if not candidate_passed:
        return ChallengeDecision("rejected_holdout", None)
    decision = promotion.decide(
        candidate_score=candidate.score,
        candidate_eligible=True,
        candidate_results=[result.to_dict() for result in candidate.tasks],
        baseline_score=incumbent.score,
        baseline_results=[result.to_dict() for result in incumbent.tasks],
        minimum_effect=minimum_effect,
        alpha=alpha,
    )
    return ChallengeDecision("replaced" if decision.promoted else "retained", decision)
