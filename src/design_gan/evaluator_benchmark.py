"""Labeled validity benchmark for browser-task actors.

The benchmark is deliberately product-shaped rather than a generic web-agent
suite. It records whether an actor correctly accepts or rejects concrete
landing-page, lead-form, and storefront behaviors. Alternate actors can be passed to
``run_benchmark`` without connecting them to the optimization loop.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import browser_evaluator

Evaluator = Callable[..., Awaitable[browser_evaluator.EvaluationResult]]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    domain: str
    task: browser_evaluator.BrowserTask
    html: str
    expected_pass: bool


@dataclass(frozen=True)
class BenchmarkCaseResult:
    id: str
    domain: str
    expected_pass: bool
    actual_pass: bool
    correct: bool
    score: float
    errors: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkReport:
    corpus_version: int
    actor: str
    results: tuple[BenchmarkCaseResult, ...]

    @property
    def correct(self) -> int:
        return sum(result.correct for result in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "actor": self.actor,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "results": [asdict(result) for result in self.results],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _task(
    case_id: str,
    behavior: str,
    *,
    interaction: str = "pointer",
    viewport: tuple[int, int] = (1280, 800),
) -> browser_evaluator.BrowserTask:
    return browser_evaluator.BrowserTask(
        id=case_id,
        name=case_id.replace("-", " "),
        instruction=f"Exercise {behavior} and verify a meaningful response.",
        behavior=behavior,
        viewport=viewport,
        interaction=interaction,
    )


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "landing-marked-action",
        "landing-page",
        _task("landing-marked-action", "primary-action"),
        """<html><body><button data-primary-action
        onclick="document.querySelector('output').textContent='Started'">Start</button>
        <output></output></body></html>""",
        True,
    ),
    BenchmarkCase(
        "landing-keyboard-action",
        "landing-page",
        _task("landing-keyboard-action", "primary-action", interaction="keyboard"),
        """<html><body><button data-primary-action
        onclick="document.querySelector('output').textContent='Started'">Start</button>
        <output></output></body></html>""",
        True,
    ),
    BenchmarkCase(
        "landing-no-response",
        "landing-page",
        _task("landing-no-response", "primary-action"),
        "<html><body><button data-primary-action>Start</button></body></html>",
        False,
    ),
    BenchmarkCase(
        "landing-disabled-action",
        "landing-page",
        _task("landing-disabled-action", "primary-action"),
        "<html><body><button data-primary-action disabled>Start</button></body></html>",
        False,
    ),
    BenchmarkCase(
        "landing-mobile-hidden-action",
        "landing-page",
        _task(
            "landing-mobile-hidden-action",
            "primary-action",
            viewport=(390, 844),
        ),
        """<html><head><style>@media(max-width:500px){button{display:none}}</style></head>
        <body><button data-primary-action onclick="document.body.textContent='Done'">
        Start</button></body></html>""",
        False,
    ),
    BenchmarkCase(
        "landing-cookie-control-only",
        "landing-page",
        _task("landing-cookie-control-only", "primary-action"),
        """<html><body><div id="cookie"><p>Cookies</p>
        <button onclick="document.querySelector('#cookie').remove()">Accept</button>
        </div><main><h1>Product</h1></main></body></html>""",
        False,
    ),
    BenchmarkCase(
        "landing-primary-among-cookie-controls",
        "landing-page",
        _task("landing-primary-among-cookie-controls", "primary-action"),
        """<html><body><div id="cookie"><button
        onclick="document.querySelector('#cookie').remove()">Accept</button></div>
        <main><button class="hero-cta"
        onclick="document.querySelector('output').textContent='Booking started'">Book tour</button>
        <output></output></main></body></html>""",
        True,
    ),
    BenchmarkCase(
        "landing-action-runtime-error",
        "landing-page",
        _task("landing-action-runtime-error", "primary-action"),
        """<html><body><button data-primary-action
        onclick="document.querySelector('output').textContent='Started';throw new Error('broken')">
        Start</button><output></output></body></html>""",
        False,
    ),
    BenchmarkCase(
        "lead-marked-form",
        "lead-generation",
        _task("lead-marked-form", "form-completion"),
        """<html><body><form data-primary-action
        onsubmit="event.preventDefault();this.outerHTML='<p>Thanks</p>'">
        <label>Name<input required></label><label>Email<input type="email" required></label>
        <button type="submit">Request demo</button></form></body></html>""",
        True,
    ),
    BenchmarkCase(
        "lead-primary-among-distractors",
        "lead-generation",
        _task("lead-primary-among-distractors", "form-completion"),
        """<html><body><form role="search"><input type="search"><button>Search</button></form>
        <form data-primary-action onsubmit="event.preventDefault();this.outerHTML='<p>Thanks</p>'">
        <label>Work email<input type="email" required></label>
        <button type="submit">Contact sales</button></form></body></html>""",
        True,
    ),
    BenchmarkCase(
        "lead-keyboard-form",
        "lead-generation",
        _task("lead-keyboard-form", "form-completion", interaction="keyboard"),
        """<html><body><form data-primary-action
        onsubmit="event.preventDefault();this.outerHTML='<p>Thanks</p>'">
        <label>Email<input type="email" required></label>
        <button type="submit">Request demo</button></form></body></html>""",
        True,
    ),
    BenchmarkCase(
        "lead-no-success-state",
        "lead-generation",
        _task("lead-no-success-state", "form-completion"),
        """<html><body><form onsubmit="event.preventDefault()">
        <label>Email<input type="email" required></label>
        <button type="submit">Request demo</button></form></body></html>""",
        False,
    ),
    BenchmarkCase(
        "lead-missing-form",
        "lead-generation",
        _task("lead-missing-form", "form-completion"),
        '<html><body><a href="#contact">Contact us</a><section id="contact"></section></body></html>',
        False,
    ),
    BenchmarkCase(
        "lead-spinner-without-success",
        "lead-generation",
        _task("lead-spinner-without-success", "form-completion"),
        """<html><body><form data-primary-action onsubmit="event.preventDefault();
        this.querySelector('button').textContent='Sending…'">
        <label>Email<input type="email" required></label>
        <button type="submit">Request demo</button></form></body></html>""",
        False,
    ),
    BenchmarkCase(
        "lead-explicit-received-state",
        "lead-generation",
        _task("lead-explicit-received-state", "form-completion"),
        """<html><body><form data-primary-action onsubmit="event.preventDefault();
        document.querySelector('[role=status]').textContent='Request received'">
        <label>Email<input type="email" required></label>
        <button type="submit">Request demo</button></form><p role="status"></p></body></html>""",
        True,
    ),
    BenchmarkCase(
        "storefront-cart-state",
        "storefront",
        _task("storefront-cart-state", "cart-addition"),
        """<html><body><button data-primary-action
        onclick="this.textContent='Added to cart';document.querySelector('output').textContent='Cart (1)'">
        Add to cart</button><output>Cart (0)</output></body></html>""",
        True,
    ),
    BenchmarkCase(
        "storefront-keyboard-cart",
        "storefront",
        _task("storefront-keyboard-cart", "cart-addition", interaction="keyboard"),
        """<html><body><button data-primary-action
        onclick="this.textContent='Added to bag';document.querySelector('output').textContent='Bag (1)'">
        Add to bag</button><output>Bag (0)</output></body></html>""",
        True,
    ),
    BenchmarkCase(
        "storefront-generic-change",
        "storefront",
        _task("storefront-generic-change", "cart-addition"),
        """<html><body><button data-primary-action
        onclick="document.querySelector('output').textContent='Please wait'">Add to cart</button>
        <output></output></body></html>""",
        False,
    ),
    BenchmarkCase(
        "storefront-preexisting-cart-count",
        "storefront",
        _task("storefront-preexisting-cart-count", "cart-addition"),
        """<html><body><p>Cart (1)</p><button data-primary-action
        onclick="document.querySelector('output').textContent='Please wait'">Add to cart</button>
        <output></output></body></html>""",
        False,
    ),
)


async def run_benchmark(
    cases: Iterable[BenchmarkCase] = BENCHMARK_CASES,
    *,
    actor: str = "semantic-v4",
    evaluator: Evaluator = browser_evaluator.evaluate,
) -> BenchmarkReport:
    results: list[BenchmarkCaseResult] = []
    for case in cases:
        evaluation = await evaluator(case.html, tasks=(case.task,), trials_per_task=1)
        actual_pass = (
            evaluation.total == 1 and evaluation.passed == 1 and not evaluation.correctness_errors
        )
        results.append(
            BenchmarkCaseResult(
                id=case.id,
                domain=case.domain,
                expected_pass=case.expected_pass,
                actual_pass=actual_pass,
                correct=actual_pass == case.expected_pass,
                score=evaluation.score,
                errors=tuple(evaluation.correctness_errors),
            )
        )
    return BenchmarkReport(corpus_version=3, actor=actor, results=tuple(results))
