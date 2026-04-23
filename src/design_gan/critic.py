"""Critic agent: scores a rendered site on the System Usability Scale."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from pydantic import BaseModel, Field, ValidationError

# SUS items. Odd items are positive, even items are negative — see scorer.py.
SUS_ITEMS = [
    "I think that I would like to use this website frequently.",
    "I found the website unnecessarily complex.",
    "I thought the website was easy to use.",
    "I think that I would need the support of a technical person to be able to use this website.",
    "I found the various functions in this website were well integrated.",
    "I thought there was too much inconsistency in this website.",
    "I would imagine that most people would learn to use this website very quickly.",
    "I found the website very cumbersome to use.",
    "I felt very confident using the website.",
    "I needed to learn a lot of things before I could get going with this website.",
]


Likert = Annotated[int, Field(ge=1, le=5)]


class SUSResponse(BaseModel):
    """Critic's answer: 10 Likert scores (1-5), prose feedback, actionable suggestions."""

    sus: Annotated[list[Likert], Field(min_length=10, max_length=10)] = Field(
        description="Answers to the 10 SUS items, in order, on a 1-5 Likert scale."
    )
    feedback: str = Field(
        description="2-4 sentence overall assessment of the site's usability."
    )
    suggestions: Annotated[list[str], Field(min_length=1, max_length=10)] = Field(
        description="3-6 concrete, prioritized improvements the generator should make next."
    )


_BASE_CRITIC_SYSTEM = """You are a __ROLE__ scoring a website on the System Usability Scale.

You will be given:
- The path to a rendered screenshot of the site. Use the Read tool to view it.
- A DOM snapshot (may be truncated).
- An axe-core accessibility report summary.

__LENS__

Score the site as an experienced user would, using the SUS questionnaire. Each item is rated 1-5:
  1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree.

Be honest and calibrated. Do not be generous. Most unimproved sites should score well below 70. A
score of 85+ is reserved for sites that would pass a real usability study.

Output contract:
- Return ONLY one fenced JSON code block: ```json { ... } ```
- No prose before or after the block.
- JSON schema:
  {
    "sus": [int,int,int,int,int,int,int,int,int,int],   // ten 1-5 Likert answers in order
    "feedback": "2-4 sentence overall assessment",
    "suggestions": ["concrete suggestion 1", "suggestion 2", ...]  // 3-6 items
  }
"""


@dataclass(frozen=True)
class CriticProfile:
    """A named critic persona with its own scoring lens.

    The prompt is rendered from _BASE_CRITIC_SYSTEM via simple token
    substitution (not .format()) so the literal curly braces in the output
    schema need no escaping. Every critic still scores the same 10-item SUS —
    they just weigh signals differently.
    """

    name: str  # short label for feedback prefix, e.g. "Usability"
    role: str  # what the critic *is*, e.g. "UX researcher"
    lens: str  # what this critic attends to

    def system_prompt(self) -> str:
        return (
            _BASE_CRITIC_SYSTEM
            .replace("__ROLE__", self.role)
            .replace("__LENS__", self.lens)
        )


# Default single critic — what every prior run used.
USABILITY_CRITIC = CriticProfile(
    name="Usability",
    role="UX researcher",
    lens=(
        "Weight navigation clarity, task completion, primary-action visibility, and "
        "information density. You care first about whether a user can accomplish what "
        "they came here to do without thinking too hard."
    ),
)

VISUAL_CRITIC = CriticProfile(
    name="Visual design",
    role="senior visual designer",
    lens=(
        "Weight typographic hierarchy, spacing and rhythm, colour harmony, composition, "
        "and overall visual polish. You care first about whether the site looks "
        "considered, professional, and aesthetically coherent — not just functional."
    ),
)

CONTENT_CRITIC = CriticProfile(
    name="Content & clarity",
    role="editorial lead",
    lens=(
        "Weight clarity of the core message, strength of the value proposition, and "
        "quality of the copy. You care first about whether the site communicates what "
        "it is and why a visitor should care within the first few seconds."
    ),
)

TRIO = [USABILITY_CRITIC, VISUAL_CRITIC, CONTENT_CRITIC]

# Backward-compat alias: callers that still import CRITIC_SYSTEM get the
# usability critic's prompt, i.e. identical to pre-ensemble behaviour.
CRITIC_SYSTEM = USABILITY_CRITIC.system_prompt()


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> str:
    match = _JSON_BLOCK.search(text)
    if match:
        return match.group(1)
    # Fallback: find the outermost braces.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise ValueError("No JSON object found in critic response.")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"


def _summarize_axe(violations: list[dict[str, Any]]) -> str:
    if not violations:
        return "axe-core: no violations detected (or axe unavailable)."
    lines = [f"axe-core found {len(violations)} violation types:"]
    for v in violations[:15]:
        impact = v.get("impact", "unknown")
        nodes = len(v.get("nodes", []))
        lines.append(f"- [{impact}] {v.get('id')}: {v.get('help')} ({nodes} node(s))")
    if len(violations) > 15:
        lines.append(f"... {len(violations) - 15} more")
    return "\n".join(lines)


def _build_user_message(
    screenshot_path: Path,
    dom_html: str,
    axe_violations: list[dict[str, Any]],
    brief: str,
) -> str:
    items_block = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(SUS_ITEMS))
    return (
        f"Brief the site was built for: {brief}\n\n"
        f"Screenshot: {screenshot_path}\n"
        f"Read that PNG with the Read tool before scoring.\n\n"
        f"SUS items (answer each 1-5 in order):\n{items_block}\n\n"
        f"{_summarize_axe(axe_violations)}\n\n"
        f"DOM snapshot:\n```html\n{_truncate(dom_html, 12000)}\n```"
    )


async def _run_once(
    model: str, system_prompt: str, user_message: str, screenshot_dir: Path
) -> tuple[str, float]:
    final: str | None = None
    cost_usd: float = 0.0
    async for msg in query(
        prompt=user_message,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            allowed_tools=["Read"],
            # Scope the Read tool so it can only see the screenshot's directory.
            # Combined with bypassPermissions this keeps the critic from
            # wandering the filesystem if a brief tries to coax it.
            add_dirs=[str(screenshot_dir)],
            permission_mode="bypassPermissions",
            # One Read + one final answer is all we need. Short-circuit runaway loops.
            max_turns=4,
        ),
    ):
        if isinstance(msg, ResultMessage):
            if msg.is_error:
                raise RuntimeError(f"Critic run failed: {msg.result!r}")
            final = msg.result
            cost_usd = msg.total_cost_usd or 0.0
    if not final:
        raise RuntimeError("Critic produced no result.")
    return final, cost_usd


async def _critique_one(
    model: str,
    profile: CriticProfile,
    screenshot_path: Path,
    dom_html: str,
    axe_violations: list[dict[str, Any]],
    brief: str,
) -> tuple[SUSResponse, float]:
    """Single-critic critique with retry-on-bad-JSON. Returns (response, cost_usd)."""
    user_message = _build_user_message(screenshot_path, dom_html, axe_violations, brief)
    screenshot_dir = screenshot_path.parent
    system_prompt = profile.system_prompt()

    total_cost: float = 0.0
    last_error: Exception | None = None
    for attempt in range(2):
        raw, cost = await _run_once(model, system_prompt, user_message, screenshot_dir)
        total_cost += cost
        try:
            payload = _extract_json(raw)
            return SUSResponse.model_validate_json(payload), total_cost
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_error = e
            user_message = (
                _build_user_message(screenshot_path, dom_html, axe_violations, brief)
                + "\n\nIMPORTANT: Your previous response was not valid JSON matching the schema. "
                "Return ONLY a single ```json ... ``` block now."
            )

    raise RuntimeError(
        f"Critic '{profile.name}' failed to return valid JSON after 2 attempts: {last_error}"
    )


async def critique(
    model: str,
    screenshot_path: Path,
    dom_html: str,
    axe_violations: list[dict[str, Any]],
    brief: str,
) -> tuple[SUSResponse, float]:
    """Single-critic critique (Usability lens). Returns (SUSResponse, cost_usd)."""
    return await _critique_one(
        model, USABILITY_CRITIC, screenshot_path, dom_html, axe_violations, brief
    )


def _dedupe_suggestions(all_suggestions: list[str], cap: int = 10) -> list[str]:
    """Combine suggestions across critics, drop near-duplicates by prefix."""
    seen: set[str] = set()
    out: list[str] = []
    for s in all_suggestions:
        key = re.sub(r"\s+", " ", s.strip().lower())[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _aggregate(
    profiles: list[CriticProfile], responses: list[SUSResponse]
) -> SUSResponse:
    """Mean SUS per item (rounded to int), labelled feedback, deduped suggestions."""
    n = len(responses)
    assert n >= 1
    # Per-item mean, rounded half-to-even. For n=3 this gives integers anyway.
    mean_sus = [
        max(1, min(5, round(sum(r.sus[i] for r in responses) / n)))
        for i in range(10)
    ]

    feedback_parts = [
        f"**{p.name}**: {r.feedback}" for p, r in zip(profiles, responses)
    ]
    feedback = "\n\n".join(feedback_parts)

    all_suggs: list[str] = []
    for r in responses:
        all_suggs.extend(r.suggestions)
    merged_suggs = _dedupe_suggestions(all_suggs)

    return SUSResponse(
        sus=mean_sus,
        feedback=feedback,
        suggestions=merged_suggs or ["(no new suggestions)"],
    )


async def critique_ensemble(
    model: str,
    profiles: list[CriticProfile],
    screenshot_path: Path,
    dom_html: str,
    axe_violations: list[dict[str, Any]],
    brief: str,
) -> tuple[SUSResponse, list[dict[str, Any]], float]:
    """Run multiple critics in parallel, aggregate their scores.

    Returns:
        aggregated SUSResponse (mean SUS, labelled feedback, merged suggestions)
        per-critic breakdown (list of dicts: name + raw sus + feedback + suggestions)
        total cost across all critics, including retries
    """
    assert profiles, "critique_ensemble requires at least one profile"

    results = await asyncio.gather(
        *(
            _critique_one(model, p, screenshot_path, dom_html, axe_violations, brief)
            for p in profiles
        )
    )
    responses = [r for r, _ in results]
    total_cost = sum(c for _, c in results)

    aggregated = _aggregate(profiles, responses)
    breakdown = [
        {
            "name": p.name,
            "sus": list(r.sus),
            "feedback": r.feedback,
            "suggestions": list(r.suggestions),
        }
        for p, r in zip(profiles, responses)
    ]
    return aggregated, breakdown, total_cost
