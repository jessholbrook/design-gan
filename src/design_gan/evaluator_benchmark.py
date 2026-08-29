"""Labeled validity benchmark for browser-task actors.

The benchmark is deliberately product-shaped rather than a generic web-agent
suite. It records whether an actor correctly accepts or rejects concrete
landing-page, lead-form, and storefront behaviors. Alternate actors can be passed to
``run_benchmark`` without connecting them to the optimization loop.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import artifact_policy, browser_evaluator, incumbent_ledger, storage

Evaluator = Callable[..., Awaitable[browser_evaluator.EvaluationResult]]
CORPUS_VERSION = 4


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    domain: str
    task: browser_evaluator.BrowserTask
    html: str
    expected_pass: bool
    provenance: dict[str, Any] = field(default_factory=lambda: {"source": "built-in"})

    def to_fixture(self) -> dict[str, Any]:
        return {
            "fixture_version": 1,
            "id": self.id,
            "domain": self.domain,
            "task": self.task.to_dict(),
            "html": self.html,
            "expected_pass": self.expected_pass,
            "provenance": self.provenance,
        }


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
        "landing-mobile-keyboard-action",
        "landing-page",
        _task(
            "landing-mobile-keyboard-action",
            "primary-action",
            interaction="keyboard",
            viewport=(390, 844),
        ),
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
        "lead-mobile-keyboard-form",
        "lead-generation",
        _task(
            "lead-mobile-keyboard-form",
            "form-completion",
            interaction="keyboard",
            viewport=(390, 844),
        ),
        """<html><body><form data-primary-action
        onsubmit="event.preventDefault();this.outerHTML='<p>Request received</p>'">
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
        "storefront-mobile-keyboard-cart",
        "storefront",
        _task(
            "storefront-mobile-keyboard-cart",
            "cart-addition",
            interaction="keyboard",
            viewport=(390, 844),
        ),
        """<html><body><button data-primary-action
        onclick="document.querySelector('output').textContent='Cart (1)'">Add to cart</button>
        <output>Cart (0)</output></body></html>""",
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

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")


def write_case_fixture(case: BenchmarkCase, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case.to_fixture(), indent=2), encoding="utf-8")
    return path


def load_case_fixture(path: Path) -> BenchmarkCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fixture_version") != 1:
        raise ValueError(f"unsupported evaluator fixture version in {path}")
    case_id = payload.get("id")
    if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
        raise ValueError(f"invalid evaluator case id in {path}")
    expected_pass = payload.get("expected_pass")
    if not isinstance(expected_pass, bool):
        raise ValueError(f"expected_pass must be boolean in {path}")
    html = payload.get("html")
    if not isinstance(html, str):
        raise ValueError(f"html must be a string in {path}")
    validation = artifact_policy.validate_html(html)
    if not validation.passed:
        raise ValueError(f"captured evaluator artifact violates policy in {path}")
    task_payload = payload.get("task")
    if not isinstance(task_payload, dict):
        raise ValueError(f"task must be an object in {path}")
    if isinstance(task_payload.get("viewport"), list):
        task_payload = {**task_payload, "viewport": tuple(task_payload["viewport"])}
    try:
        task = browser_evaluator.BrowserTask(**task_payload)
        browser_evaluator.frozen_suite((task,))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluator task in {path}: {exc}") from exc
    provenance = payload.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError(f"provenance must be an object in {path}")
    domain = payload.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError(f"domain must be a non-empty string in {path}")
    return BenchmarkCase(
        id=case_id,
        domain=domain,
        task=task,
        html=html,
        expected_pass=expected_pass,
        provenance=provenance,
    )


def load_case_directory(path: Path) -> tuple[BenchmarkCase, ...]:
    cases = tuple(load_case_fixture(item) for item in sorted(path.glob("*.json")))
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("captured evaluator case ids must be unique")
    return cases


def capture_run_case(
    store: storage.Storage,
    *,
    run_id: int,
    iteration: int,
    task_id: str,
    case_id: str,
    expected_pass: bool,
) -> BenchmarkCase:
    """Capture an operator-labeled evaluator case from immutable run history."""
    if not _CASE_ID.fullmatch(case_id):
        raise ValueError("case_id must be 3-80 lowercase letters, numbers, or hyphens")
    run = store.get_run(run_id)
    if run is None or (run.get("kind") or "design") != "design":
        raise ValueError(f"design run {run_id} not found")
    record = next(
        (item for item in store.iterations_for_run(run_id) if item["iter"] == iteration),
        None,
    )
    if record is None:
        raise ValueError(f"iteration {iteration} not found in run {run_id}")
    plan = run.get("evaluation_plan") or {}
    tasks = plan.get("tasks") or run.get("evaluation_suite") or []
    task_payload = next((task for task in tasks if task.get("id") == task_id), None)
    if task_payload is None:
        raise ValueError(f"task {task_id!r} not found in run {run_id}'s frozen suite")
    if isinstance(task_payload.get("viewport"), list):
        task_payload = {**task_payload, "viewport": tuple(task_payload["viewport"])}
    task = browser_evaluator.BrowserTask(**task_payload)
    html = record["html"]
    validation = artifact_policy.validate_html(html)
    if not validation.passed:
        raise ValueError("stored iteration violates the evaluator artifact policy")
    return BenchmarkCase(
        id=case_id,
        domain=run.get("domain") or plan.get("domain") or "custom",
        task=task,
        html=html,
        expected_pass=expected_pass,
        provenance={
            "source": "design-gan-run",
            "run_id": run_id,
            "iteration": iteration,
            "task_id": task_id,
            "artifact_sha256": incumbent_ledger.artifact_hash(html),
            "captured_at": time.time(),
        },
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
    return BenchmarkReport(corpus_version=CORPUS_VERSION, actor=actor, results=tuple(results))
