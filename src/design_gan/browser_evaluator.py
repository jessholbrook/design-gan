"""Concrete browser-task evaluation for design runs.

The v2 milestone intentionally starts with one stable product task instead of
an open-ended evaluator framework: can a user find and activate the page's
primary action, and does the page visibly respond?  The same immutable suite
is replayed from a clean browser page for every candidate in a run.

Task completion is the design loop's primary metric.  Runtime errors observed
while performing the task are reported separately so the scorer can use them
as a correctness promotion guardrail rather than blending them into the task
score.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BrowserTask:
    """A frozen behavioral scenario supported by the first v2 milestone."""

    id: str
    name: str
    instruction: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# Keep the initial scope deliberately concrete.  Later milestones can add
# product-specific suites and repeated trials without changing this result
# contract or the promotion logic built on top of it.
DEFAULT_DESIGN_TASKS: tuple[BrowserTask, ...] = (
    BrowserTask(
        id="primary-action",
        name="Primary action works",
        instruction=(
            "Find the page's primary call to action, activate it, and observe a "
            "meaningful response such as navigation, scrolling, a dialog, a new "
            "window, or changed visible content."
        ),
    ),
)


@dataclass
class TaskResult:
    task_id: str
    name: str
    instruction: str
    passed: bool
    target: str | None
    observed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    score: float
    tasks: list[TaskResult]
    correctness_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for task in self.tasks if task.passed)

    @property
    def total(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": "task_completion_rate",
            "score": self.score,
            "passed": self.passed,
            "total": self.total,
            "tasks": [task.to_dict() for task in self.tasks],
            "correctness_errors": list(self.correctness_errors),
        }

    def feedback(self) -> str:
        lines = [f"Behavioral task completion: {self.passed}/{self.total} ({self.score:.0f}/100)."]
        for task in self.tasks:
            status = "PASS" if task.passed else "FAIL"
            detail = "; ".join(task.observed or task.errors) or "no response observed"
            lines.append(f"- {status} {task.name}: {detail}")
        return "\n".join(lines)


_CTA_WORDS = re.compile(
    r"\b(book|reserve|get started|start|sign up|signup|join|buy|shop|contact|"
    r"request|try|subscribe|register|apply|schedule|learn more|explore)\b",
    re.IGNORECASE,
)


def frozen_suite(tasks: Iterable[BrowserTask] | None = None) -> tuple[BrowserTask, ...]:
    """Return an immutable, validated suite for the lifetime of a run."""
    suite = tuple(DEFAULT_DESIGN_TASKS if tasks is None else tasks)
    if not suite:
        raise ValueError("design evaluation requires at least one browser task")
    ids = [task.id for task in suite]
    if len(ids) != len(set(ids)):
        raise ValueError("browser task ids must be unique")
    unsupported = [task.id for task in suite if task.id != "primary-action"]
    if unsupported:
        raise ValueError(f"unsupported browser task(s): {', '.join(unsupported)}")
    return suite


def _candidate_score(candidate: dict[str, Any]) -> int:
    """Rank visible actions without coupling the task to generated copy."""
    attrs = " ".join(
        str(candidate.get(key) or "") for key in ("className", "id", "ariaLabel", "dataPrimary")
    ).lower()
    text = str(candidate.get("text") or "").strip()
    tag = str(candidate.get("tag") or "").lower()

    score = 0
    if candidate.get("dataPrimary") is not None:
        score += 100
    if any(word in attrs for word in ("primary", "cta", "hero-action", "main-action")):
        score += 50
    if _CTA_WORDS.search(text):
        score += 40
    if tag == "button" or candidate.get("role") == "button":
        score += 15
    if candidate.get("href") not in (None, "", "#"):
        score += 5
    # Prefer prominent controls when semantic signals tie.
    score += min(20, int((candidate.get("width", 0) * candidate.get("height", 0)) / 1000))
    return score


async def _primary_action(page: Any, task: BrowserTask) -> TaskResult:
    page_errors: list[str] = []
    console_errors: list[str] = []
    dialogs: list[str] = []
    popups: list[str] = []

    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on(
        "console",
        lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
    )

    def on_dialog(dialog: Any) -> None:
        dialogs.append(dialog.type)
        asyncio.create_task(dialog.dismiss())

    page.on("dialog", on_dialog)
    page.context.on("page", lambda popup: popups.append(popup.url))

    actions = page.locator("button, a[href], input[type=submit], input[type=button], [role=button]")
    candidates: list[dict[str, Any]] = await actions.evaluate_all(
        """els => els.map((el, index) => {
          const r = el.getBoundingClientRect();
          const style = getComputedStyle(el);
          return {
            index,
            tag: el.tagName.toLowerCase(),
            text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim(),
            href: el.getAttribute('href'),
            role: el.getAttribute('role'),
            className: typeof el.className === 'string' ? el.className : '',
            id: el.id || '',
            ariaLabel: el.getAttribute('aria-label'),
            dataPrimary: el.getAttribute('data-primary-action'),
            disabled: Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true',
            visible: r.width > 0 && r.height > 0 && style.visibility !== 'hidden' &&
                     style.display !== 'none' && Number(style.opacity) !== 0,
            width: r.width,
            height: r.height,
          };
        })"""
    )
    visible = [candidate for candidate in candidates if candidate["visible"]]
    if not visible:
        return TaskResult(
            task_id=task.id,
            name=task.name,
            instruction=task.instruction,
            passed=False,
            target=None,
            observed=["no visible interactive control was available"],
        )

    candidate = max(visible, key=_candidate_score)
    label = candidate.get("text") or candidate.get("ariaLabel") or candidate.get("tag")
    if candidate.get("disabled"):
        return TaskResult(
            task_id=task.id,
            name=task.name,
            instruction=task.instruction,
            passed=False,
            target=str(label),
            observed=["the most likely primary action was disabled"],
        )

    target = actions.nth(candidate["index"])
    before_url = page.url
    before_text = " ".join((await page.locator("body").inner_text()).split())
    before_scroll = await page.evaluate("() => window.scrollY")
    errors: list[str] = []
    try:
        await target.click(timeout=3000, no_wait_after=True)
        await page.wait_for_timeout(350)
    except Exception as exc:  # noqa: BLE001 - action failures are recorded as evidence
        errors.append(f"activation failed: {type(exc).__name__}: {exc}")

    observed: list[str] = []
    try:
        if not page.is_closed():
            after_url = page.url
            after_text = " ".join((await page.locator("body").inner_text()).split())
            after_scroll = await page.evaluate("() => window.scrollY")
            if after_url != before_url:
                observed.append("URL changed")
            if after_text != before_text:
                observed.append("visible content changed")
            if abs(after_scroll - before_scroll) >= 20:
                observed.append("page scrolled to new content")
    except Exception as exc:  # noqa: BLE001 - navigation failures are evidence
        errors.append(f"response observation failed: {type(exc).__name__}: {exc}")
    if dialogs:
        observed.append(f"opened {dialogs[0]} dialog")
    if popups:
        observed.append("opened a new window")

    runtime_errors = [*page_errors, *console_errors]
    errors.extend(runtime_errors)
    passed = bool(observed) and not errors
    if not observed and not errors:
        observed.append("activation produced no observable response")
    return TaskResult(
        task_id=task.id,
        name=task.name,
        instruction=task.instruction,
        passed=passed,
        target=str(label)[:160],
        observed=observed,
        errors=errors,
    )


async def evaluate(
    html: str,
    *,
    tasks: Iterable[BrowserTask] | None = None,
    viewport: tuple[int, int] = (1280, 800),
) -> EvaluationResult:
    """Replay the frozen suite in isolated Playwright pages."""
    from playwright.async_api import async_playwright

    suite = frozen_suite(tasks)
    results: list[TaskResult] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            for task in suite:
                context = await browser.new_context(
                    viewport={"width": viewport[0], "height": viewport[1]}
                )
                page = await context.new_page()
                try:
                    await page.set_content(html, wait_until="domcontentloaded")
                    await page.wait_for_timeout(300)
                    results.append(await _primary_action(page, task))
                except Exception as exc:  # noqa: BLE001 - isolate each frozen task
                    results.append(
                        TaskResult(
                            task_id=task.id,
                            name=task.name,
                            instruction=task.instruction,
                            passed=False,
                            target=None,
                            errors=[f"evaluator failed: {type(exc).__name__}: {exc}"],
                        )
                    )
                finally:
                    await context.close()
        finally:
            await browser.close()

    correctness_errors = [error for result in results for error in result.errors]
    score = round(100.0 * sum(result.passed for result in results) / len(results), 2)
    return EvaluationResult(
        score=score,
        tasks=results,
        correctness_errors=correctness_errors,
    )


def write_artifact(result: EvaluationResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evaluation.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
