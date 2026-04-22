"""Generator agent: produces a single-page website as HTML/CSS/JS."""

from __future__ import annotations

import re
from dataclasses import dataclass

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

GENERATOR_SYSTEM = """You are a senior front-end designer generating single-page websites.

Output contract:
- Return ONE complete, standalone HTML document.
- Inline all CSS in a <style> tag and all JS in a <script> tag. No external requests.
- Wrap the entire document inside a fenced code block: ```html ... ```
- No commentary before or after the code block.
- The site must render meaningfully at 1280x800 in a headless browser with no network.

Design priorities (in order):
1. Clarity of purpose: a visitor should understand what the site is for within 2 seconds.
2. Usability: primary actions are obvious, reachable, and labeled.
3. Accessibility: semantic HTML, sufficient contrast, keyboard navigation, alt text.
4. Visual polish: coherent typography, spacing, and color system.
"""


@dataclass
class GenerationRequest:
    brief: str
    prior_html: str | None = None
    critic_feedback: str | None = None
    suggestions: list[str] | None = None


def _build_user_message(req: GenerationRequest) -> str:
    parts = [f"Brief: {req.brief}"]
    if req.prior_html:
        parts.append(
            "Here is the previous version of the site. Keep what works; fix the issues below.\n\n"
            f"```html\n{req.prior_html}\n```"
        )
    if req.critic_feedback:
        parts.append(f"Critic feedback:\n{req.critic_feedback}")
    if req.suggestions:
        bullets = "\n".join(f"- {s}" for s in req.suggestions)
        parts.append(f"Prioritize these specific suggestions:\n{bullets}")
    parts.append("Produce the next version of the site now.")
    return "\n\n".join(parts)


_HTML_BLOCK = re.compile(r"```html\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# A generated single-page site should never reasonably exceed this. Larger
# outputs are almost always runaway generations (repeated sections, hallucinated
# base64 blobs). Rejecting keeps SQLite + disk bounded.
MAX_HTML_BYTES = 512 * 1024  # 512 KiB


def _extract_html(text: str) -> str:
    match = _HTML_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def generate(model: str, req: GenerationRequest) -> tuple[str, float]:
    """Generate a single-page site. Returns (html, cost_usd)."""
    final: str | None = None
    cost_usd: float = 0.0
    async for msg in query(
        prompt=_build_user_message(req),
        options=ClaudeAgentOptions(
            system_prompt=GENERATOR_SYSTEM,
            model=model,
            tools=[],
        ),
    ):
        if isinstance(msg, ResultMessage):
            if msg.is_error:
                raise RuntimeError(f"Generator run failed: {msg.result!r}")
            final = msg.result
            cost_usd = msg.total_cost_usd or 0.0
    if not final:
        raise RuntimeError("Generator produced no result.")
    html = _extract_html(final)
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        raise RuntimeError(
            f"Generator output exceeded {MAX_HTML_BYTES} bytes "
            f"({len(html)} chars) — likely runaway generation."
        )
    return html, cost_usd
