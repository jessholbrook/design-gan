from __future__ import annotations

import pytest

from design_gan.product_domains import DOMAINS, get_domain, make_plan


def test_two_concrete_product_domains_are_registered():
    assert set(DOMAINS) == {"landing-page", "lead-generation"}
    assert get_domain("landing-page").tasks[0].id == "primary-action"
    assert get_domain("lead-generation").tasks[0].id == "form-completion"


def test_plan_freezes_versions_trials_and_promotion_policy():
    plan = make_plan(
        "lead-generation",
        trials_per_task=8,
        promotion_alpha=0.025,
        minimum_effect=5.0,
    )
    payload = plan.to_dict()
    assert payload["domain"] == "lead-generation"
    assert payload["domain_version"] == 1
    assert payload["evaluator_version"] == 2
    assert payload["trials_per_task"] == 8
    assert payload["tasks"][0]["id"] == "form-completion"


@pytest.mark.parametrize("domain", ["storefront", "", "conversation"])
def test_unknown_domain_is_rejected(domain: str):
    with pytest.raises(ValueError, match="unknown product domain"):
        get_domain(domain)
