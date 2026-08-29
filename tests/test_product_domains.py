from __future__ import annotations

import pytest

from design_gan.product_domains import DOMAINS, get_domain, make_plan


def test_three_concrete_product_domains_are_registered():
    assert set(DOMAINS) == {"landing-page", "lead-generation", "storefront"}
    landing = get_domain("landing-page")
    lead = get_domain("lead-generation")
    assert {task.behavior for task in landing.tasks} == {"primary-action"}
    assert {task.behavior for task in lead.tasks} == {"form-completion"}
    assert {task.split for task in landing.tasks} == {"development", "holdout"}
    assert {task.split for task in lead.tasks} == {"development", "holdout"}
    storefront = get_domain("storefront")
    assert {task.behavior for task in storefront.tasks} == {"cart-addition"}
    assert len(storefront.tasks) == 4


def test_plan_freezes_versions_trials_and_promotion_policy():
    plan = make_plan(
        "lead-generation",
        trials_per_task=8,
        promotion_alpha=0.025,
        minimum_effect=5.0,
    )
    payload = plan.to_dict()
    assert payload["domain"] == "lead-generation"
    assert payload["domain_version"] == 3
    assert payload["evaluator_version"] == 4
    assert payload["trials_per_task"] == 8
    assert len(plan.development_tasks) == 2
    assert len(plan.holdout_tasks) == 2
    assert {task.viewport for task in plan.holdout_tasks} == {(1280, 800), (390, 844)}
    assert {task.interaction for task in plan.holdout_tasks} == {"keyboard"}


@pytest.mark.parametrize("domain", ["checkout", "", "conversation"])
def test_unknown_domain_is_rejected(domain: str):
    with pytest.raises(ValueError, match="unknown product domain"):
        get_domain(domain)
