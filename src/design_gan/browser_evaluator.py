"""Concrete browser-task evaluation for design runs.

The supported domains remain deliberately small: primary-action activation for
landing pages and successful form completion for lead-generation pages. The
same immutable suite is replayed from a clean browser page for every candidate
in a run.

Task completion is the design loop's primary metric. Runtime errors observed
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
    """A frozen behavioral scenario supported by the current evaluator."""

    id: str
    name: str
    instruction: str
    behavior: str | None = None
    split: str = "development"
    viewport: tuple[int, int] | None = None
    interaction: str = "pointer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Keep the default deliberately concrete. Product-domain profiles select among
# this evaluator's small set of supported semantic behaviors.
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
    trial: int = 1
    split: str = "development"
    interaction: str = "pointer"
    viewport: tuple[int, int] | None = None
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
        task_ids = list(dict.fromkeys(task.task_id for task in self.tasks))
        for task_id in task_ids:
            attempts = [task for task in self.tasks if task.task_id == task_id]
            passed = sum(task.passed for task in attempts)
            status = "PASS" if passed == len(attempts) else "FAIL"
            detail = ""
            if passed < len(attempts):
                failed = next(task for task in attempts if not task.passed)
                detail = "; ".join(failed.observed or failed.errors) or "no response observed"
                detail = f"; first failure: {detail}"
            lines.append(
                f"- {status} {attempts[0].name}: {passed}/{len(attempts)} trials passed{detail}"
            )
        return "\n".join(lines)


_CTA_WORDS = re.compile(
    r"\b(book|reserve|get started|start|sign up|signup|join|buy|shop|contact|"
    r"request|try|subscribe|register|apply|schedule|learn more|explore)\b",
    re.IGNORECASE,
)
_CART_ACTION_WORDS = re.compile(r"\b(add to (?:cart|bag)|buy now)\b", re.IGNORECASE)
_CART_EVIDENCE = re.compile(
    r"\b(added to (?:cart|bag)|in your (?:cart|bag)|(?:cart|bag)\s*[:(]?\s*[1-9])",
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
    supported = {"primary-action", "form-completion", "cart-addition"}
    unsupported = [task.id for task in suite if (task.behavior or task.id) not in supported]
    if unsupported:
        raise ValueError(f"unsupported browser task(s): {', '.join(unsupported)}")
    invalid_splits = [task.id for task in suite if task.split not in {"development", "holdout"}]
    if invalid_splits:
        raise ValueError(f"invalid browser task split(s): {', '.join(invalid_splits)}")
    invalid_interactions = [
        task.id for task in suite if task.interaction not in {"pointer", "keyboard"}
    ]
    if invalid_interactions:
        raise ValueError(f"invalid browser task interaction(s): {', '.join(invalid_interactions)}")
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


def _cart_candidate_score(candidate: dict[str, Any]) -> int:
    score = _candidate_score(candidate)
    text = str(candidate.get("text") or "")
    attrs = " ".join(
        str(candidate.get(key) or "") for key in ("className", "id", "ariaLabel")
    ).lower()
    if _CART_ACTION_WORDS.search(text):
        score += 100
    if "cart" in attrs or "bag" in attrs:
        score += 60
    return score


def _form_score(candidate: dict[str, Any]) -> int:
    """Prefer an explicitly marked or clearly primary lead form."""
    attrs = " ".join(
        str(candidate.get(key) or "") for key in ("className", "id", "ariaLabel", "dataPrimary")
    ).lower()
    submit_text = str(candidate.get("submitText") or "")
    score = min(30, int(candidate.get("fieldCount") or 0) * 5)
    if candidate.get("dataPrimary") is not None:
        score += 100
    if any(word in attrs for word in ("primary", "lead", "contact", "signup", "register")):
        score += 50
    if _CTA_WORDS.search(submit_text):
        score += 30
    if candidate.get("hasSearch"):
        score -= 100
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

    behavior = task.behavior or task.id
    score_candidate = _cart_candidate_score if behavior == "cart-addition" else _candidate_score
    candidate = max(visible, key=score_candidate)
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
        if task.interaction == "keyboard":
            await target.focus()
            await page.keyboard.press("Enter")
        else:
            await target.click(timeout=3000, no_wait_after=True)
        await page.wait_for_timeout(350)
    except Exception as exc:  # noqa: BLE001 - action failures are recorded as evidence
        errors.append(f"activation failed: {type(exc).__name__}: {exc}")

    observed: list[str] = []
    after_url = before_url
    after_text = before_text
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
    if behavior == "cart-addition":
        cart_evidence = []
        if _CART_EVIDENCE.search(after_text) and after_text != before_text:
            cart_evidence.append("visible cart state changed")
        if _CART_EVIDENCE.search(after_url) and after_url != before_url:
            cart_evidence.append("navigated to cart state")
        observed = cart_evidence
        passed = bool(cart_evidence) and not errors
        if not cart_evidence and not errors:
            observed.append("activation produced no observable cart state")
    else:
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


async def _form_completion(page: Any, task: BrowserTask) -> TaskResult:
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

    forms = page.locator("form")
    candidates: list[dict[str, Any]] = await forms.evaluate_all(
        """forms => forms.map((form, index) => {
          const r = form.getBoundingClientRect();
          const style = getComputedStyle(form);
          const submit = form.querySelector(
            'button[type=submit], input[type=submit], button:not([type])'
          );
          return {
            index,
            id: form.id || '',
            className: typeof form.className === 'string' ? form.className : '',
            ariaLabel: form.getAttribute('aria-label'),
            dataPrimary: form.getAttribute('data-primary-action'),
            fieldCount: form.querySelectorAll('input:not([type=hidden]), textarea, select').length,
            hasSearch: Boolean(form.querySelector('input[type=search]')) ||
                       form.getAttribute('role') === 'search',
            submitText: submit ?
              (submit.innerText || submit.value || submit.getAttribute('aria-label') || '') : '',
            visible: r.width > 0 && r.height > 0 && style.visibility !== 'hidden' &&
                     style.display !== 'none' && Number(style.opacity) !== 0,
          };
        })"""
    )
    visible_forms = [candidate for candidate in candidates if candidate["visible"]]
    if not visible_forms:
        return TaskResult(
            task_id=task.id,
            name=task.name,
            instruction=task.instruction,
            passed=False,
            target=None,
            observed=["no visible form was available"],
        )

    selected = max(visible_forms, key=_form_score)
    form = forms.nth(selected["index"])
    label = (await form.get_attribute("aria-label")) or (await form.get_attribute("id")) or "form"
    fields = form.locator("input, textarea, select")
    errors: list[str] = []
    for index in range(await fields.count()):
        field = fields.nth(index)
        if not await field.is_visible() or await field.is_disabled():
            continue
        tag = await field.evaluate("el => el.tagName.toLowerCase()")
        input_type = ((await field.get_attribute("type")) or "text").lower()
        if input_type in {"hidden", "submit", "button", "reset", "image", "file"}:
            continue
        try:
            if tag == "select":
                options = await field.locator("option:not([disabled])").all()
                chosen = None
                for option in options:
                    value = await option.get_attribute("value")
                    if value:
                        chosen = value
                        break
                if chosen is not None:
                    await field.select_option(chosen)
            elif input_type in {"checkbox", "radio"}:
                await field.check()
            else:
                values = {
                    "email": "alex@example.com",
                    "tel": "4155550123",
                    "number": "2",
                    "url": "https://example.com",
                    "date": "2030-01-15",
                }
                await field.fill(values.get(input_type, "Alex Example"))
        except Exception as exc:  # noqa: BLE001 - field failures are evidence
            errors.append(f"could not complete field {index + 1}: {type(exc).__name__}: {exc}")

    before_url = page.url
    before_text = " ".join((await page.locator("body").inner_text()).split())
    before_scroll = await page.evaluate("() => window.scrollY")
    submit = form.locator("button[type=submit], input[type=submit], button:not([type])").first
    try:
        if await submit.count() and await submit.is_visible():
            if task.interaction == "keyboard":
                await submit.focus()
                await page.keyboard.press("Enter")
            else:
                await submit.click(timeout=3000, no_wait_after=True)
        else:
            await form.evaluate("el => el.requestSubmit()")
        await page.wait_for_timeout(350)
    except Exception as exc:  # noqa: BLE001 - submission failures are evidence
        errors.append(f"submission failed: {type(exc).__name__}: {exc}")

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
    errors.extend([*page_errors, *console_errors])
    passed = bool(observed) and not errors
    if not observed and not errors:
        observed.append("submission produced no observable response")
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
    trials_per_task: int = 1,
) -> EvaluationResult:
    """Replay the frozen suite in isolated Playwright pages."""
    from playwright.async_api import async_playwright

    suite = frozen_suite(tasks)
    if trials_per_task < 1 or trials_per_task > 50:
        raise ValueError("trials_per_task must be between 1 and 50")
    results: list[TaskResult] = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            for task in suite:
                for trial in range(1, trials_per_task + 1):
                    task_viewport = task.viewport or viewport
                    context = await browser.new_context(
                        viewport={"width": task_viewport[0], "height": task_viewport[1]}
                    )
                    await context.route("**/*", lambda route: route.abort())
                    page = await context.new_page()
                    try:
                        await page.set_content(html, wait_until="domcontentloaded")
                        await page.wait_for_timeout(300)
                        if (task.behavior or task.id) in {"primary-action", "cart-addition"}:
                            result = await _primary_action(page, task)
                        else:
                            result = await _form_completion(page, task)
                        result.trial = trial
                        result.split = task.split
                        result.interaction = task.interaction
                        result.viewport = task_viewport
                        results.append(result)
                    except Exception as exc:  # noqa: BLE001 - isolate each frozen task
                        results.append(
                            TaskResult(
                                task_id=task.id,
                                name=task.name,
                                instruction=task.instruction,
                                passed=False,
                                target=None,
                                trial=trial,
                                split=task.split,
                                interaction=task.interaction,
                                viewport=task_viewport,
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


def write_artifact(
    result: EvaluationResult,
    out_dir: Path,
    filename: str = "evaluation.json",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
