"""Versioned product-domain profiles and frozen evaluation plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .browser_evaluator import BrowserTask


@dataclass(frozen=True)
class ProductDomain:
    id: str
    name: str
    version: int
    tasks: tuple[BrowserTask, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    domain: str
    domain_version: int
    evaluator_version: int
    trials_per_task: int
    promotion_alpha: float
    minimum_effect: float
    tasks: tuple[BrowserTask, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = [task.to_dict() for task in self.tasks]
        return payload


LANDING_PAGE = ProductDomain(
    id="landing-page",
    name="Landing page",
    version=1,
    tasks=(
        BrowserTask(
            id="primary-action",
            name="Primary action works",
            instruction=(
                "Find the page's primary call to action, activate it, and observe a "
                "meaningful response such as navigation, scrolling, a dialog, a new "
                "window, or changed visible content."
            ),
        ),
    ),
)

LEAD_GENERATION = ProductDomain(
    id="lead-generation",
    name="Lead-generation form",
    version=1,
    tasks=(
        BrowserTask(
            id="form-completion",
            name="Lead form can be completed",
            instruction=(
                "Find the primary form, complete its required fields with valid sample "
                "data, submit it, and observe a success response without runtime errors."
            ),
        ),
    ),
)

DOMAINS: dict[str, ProductDomain] = {
    domain.id: domain for domain in (LANDING_PAGE, LEAD_GENERATION)
}


def get_domain(domain_id: str) -> ProductDomain:
    try:
        return DOMAINS[domain_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown product domain {domain_id!r}; choose one of {', '.join(DOMAINS)}"
        ) from exc


def make_plan(
    domain_id: str = "landing-page",
    *,
    trials_per_task: int = 6,
    promotion_alpha: float = 0.05,
    minimum_effect: float = 1.0,
) -> EvaluationPlan:
    if trials_per_task < 1 or trials_per_task > 50:
        raise ValueError("trials_per_task must be between 1 and 50")
    if not 0 < promotion_alpha <= 1:
        raise ValueError("promotion_alpha must be in (0, 1]")
    if not 0 <= minimum_effect <= 100:
        raise ValueError("minimum_effect must be between 0 and 100")
    domain = get_domain(domain_id)
    return EvaluationPlan(
        domain=domain.id,
        domain_version=domain.version,
        evaluator_version=2,
        trials_per_task=trials_per_task,
        promotion_alpha=promotion_alpha,
        minimum_effect=minimum_effect,
        tasks=domain.tasks,
    )
