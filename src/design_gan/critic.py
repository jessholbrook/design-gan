"""Critic agent: scores a rendered site on the System Usability Scale."""

from __future__ import annotations

from typing import Annotated, Any

import anthropic
from pydantic import BaseModel, Field

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


CRITIC_SYSTEM = """You are a UX researcher scoring a website on the System Usability Scale.

You will be shown:
- A rendered screenshot of the site.
- A DOM snapshot (may be truncated).
- An axe-core accessibility report summary.

Score the site as an experienced user would, using the SUS questionnaire. Each item is rated 1-5:
  1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree.

Be honest and calibrated. Do not be generous. Most unimproved sites should score well below 70. A
score of 85+ is reserved for sites that would pass a real usability study.

After scoring, write 2-4 sentences of overall feedback and list 3-6 concrete, prioritized
suggestions the generator should act on in its next revision.
"""


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


def critique(
    client: anthropic.Anthropic,
    model: str,
    screenshot_png: bytes,
    dom_html: str,
    axe_violations: list[dict[str, Any]],
    brief: str,
) -> SUSResponse:
    """Run the critic against a rendered site; returns structured SUS response."""
    import base64

    screenshot_b64 = base64.standard_b64encode(screenshot_png).decode("ascii")
    items_block = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(SUS_ITEMS))

    user_text = (
        f"Brief the site was built for: {brief}\n\n"
        f"SUS items (answer each 1-5 in order):\n{items_block}\n\n"
        f"{_summarize_axe(axe_violations)}\n\n"
        f"DOM snapshot:\n```html\n{_truncate(dom_html, 12000)}\n```"
    )

    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        system=[
            {
                "type": "text",
                "text": CRITIC_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        output_format=SUSResponse,
    )
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("Critic failed to return a valid SUS response.")
    return parsed
