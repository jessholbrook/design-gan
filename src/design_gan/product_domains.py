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

    @property
    def development_tasks(self) -> tuple[BrowserTask, ...]:
        return tuple(task for task in self.tasks if task.split == "development")

    @property
    def holdout_tasks(self) -> tuple[BrowserTask, ...]:
        return tuple(task for task in self.tasks if task.split == "holdout")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = [task.to_dict() for task in self.tasks]
        return payload


LANDING_PAGE = ProductDomain(
    id="landing-page",
    name="Landing page",
    version=2,
    tasks=(
        BrowserTask(
            id="landing-primary-desktop",
            name="Primary action works with a pointer on desktop",
            instruction=(
                "Find the page's primary call to action, activate it, and observe a "
                "meaningful response such as navigation, scrolling, a dialog, a new "
                "window, or changed visible content."
            ),
            behavior="primary-action",
            viewport=(1280, 800),
        ),
        BrowserTask(
            id="landing-primary-keyboard",
            name="Primary action works from the keyboard",
            instruction=(
                "Reach the primary call to action with keyboard focus, activate it with "
                "Enter, and observe a meaningful response without runtime errors."
            ),
            behavior="primary-action",
            viewport=(1280, 800),
            interaction="keyboard",
        ),
        BrowserTask(
            id="landing-primary-mobile-holdout",
            name="Primary action remains usable on mobile",
            instruction=(
                "At a compact mobile viewport, find and activate the primary call to action "
                "and observe a meaningful response without runtime errors."
            ),
            behavior="primary-action",
            split="holdout",
            viewport=(390, 844),
        ),
    ),
)

LEAD_GENERATION = ProductDomain(
    id="lead-generation",
    name="Lead-generation form",
    version=2,
    tasks=(
        BrowserTask(
            id="lead-form-desktop",
            name="Lead form can be completed on desktop",
            instruction=(
                "Find the primary form, complete its required fields with valid sample "
                "data, submit it, and observe a success response without runtime errors."
            ),
            behavior="form-completion",
            viewport=(1280, 800),
        ),
        BrowserTask(
            id="lead-form-mobile",
            name="Lead form can be completed on mobile",
            instruction=(
                "At a compact mobile viewport, complete and submit the primary lead form, "
                "then observe an offline success response without runtime errors."
            ),
            behavior="form-completion",
            viewport=(390, 844),
        ),
        BrowserTask(
            id="lead-form-keyboard-holdout",
            name="Lead form can be submitted from the keyboard",
            instruction=(
                "Complete the primary lead form and submit it from the keyboard, then "
                "observe an offline success response without runtime errors."
            ),
            behavior="form-completion",
            split="holdout",
            viewport=(1280, 800),
            interaction="keyboard",
        ),
    ),
)

STOREFRONT = ProductDomain(
    id="storefront",
    name="Single-product storefront",
    version=1,
    tasks=(
        BrowserTask(
            id="storefront-cart-desktop",
            name="A product can be added to the cart on desktop",
            instruction=(
                "Find the primary product purchase action, activate it, and observe a "
                "visible cart or bag state containing at least one item."
            ),
            behavior="cart-addition",
            viewport=(1280, 800),
        ),
        BrowserTask(
            id="storefront-cart-mobile",
            name="A product can be added to the cart on mobile",
            instruction=(
                "At a compact mobile viewport, add the primary product to the cart and "
                "observe a visible cart or bag state containing at least one item."
            ),
            behavior="cart-addition",
            viewport=(390, 844),
        ),
        BrowserTask(
            id="storefront-cart-keyboard-holdout",
            name="The cart action works from the keyboard",
            instruction=(
                "Activate the primary add-to-cart action from the keyboard and observe a "
                "visible cart or bag state containing at least one item."
            ),
            behavior="cart-addition",
            split="holdout",
            viewport=(1280, 800),
            interaction="keyboard",
        ),
    ),
)

DOMAINS: dict[str, ProductDomain] = {
    domain.id: domain for domain in (LANDING_PAGE, LEAD_GENERATION, STOREFRONT)
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
        evaluator_version=3,
        trials_per_task=trials_per_task,
        promotion_alpha=promotion_alpha,
        minimum_effect=minimum_effect,
        tasks=domain.tasks,
    )
