"""Conversation generator: evolves an assistant *system prompt* across iterations.

This is the generator agent for conversation runs, parallel to generator.py for
design runs. The evolving artifact is the assistant's system prompt; the
transcript_renderer executes it against the user_simulator to produce the
rendered artifact (the transcript).

First iteration receives no prior prompt — it invents a starting system prompt
from the goal alone. Subsequent iterations receive the prior prompt + the
critic's feedback and suggestions, and return a refined version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

CONVERSATION_GENERATOR_SYSTEM = """You are a senior prompt engineer. Your job is \
to write the *system prompt* for an AI assistant so that the assistant will
perform well in a short (1-5 turn) conversation pursuing a user's goal.

You are NOT the assistant. You don't answer the user. You write the system
prompt that will shape the assistant's behaviour.

What makes a good system prompt for this context:
- Concrete instruction on *what to do first* when the user arrives.
- Clear stance on voice, length, and level of hedging.
- Explicit anti-patterns to avoid (boilerplate openers, filler, repetition).
- Guidance on how to wrap up when the user's goal looks met.
- Short enough that the assistant doesn't drown in instructions (≤ 400 words).

Output contract:
- Return ONE complete system prompt, plain text, inside a fenced ```text ... ```
  code block.
- No commentary before or after the block.
- The system prompt must be self-contained — no references to "the previous
  version" or "the critic" inside the prompt.
"""


@dataclass
class ConversationGenerationRequest:
    goal: str
    max_turns: int = 5
    prior_system_prompt: str | None = None
    critic_feedback: str | None = None
    suggestions: list[str] | None = None


MAX_PROMPT_BYTES = 8 * 1024  # 8 KiB — an assistant system prompt shouldn't be larger.


_TEXT_BLOCK = re.compile(r"```(?:text|plaintext)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_prompt(text: str) -> str:
    match = _TEXT_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _build_user_message(req: ConversationGenerationRequest) -> str:
    parts = [
        f"User's goal for the conversation:\n{req.goal.strip()}",
        f"Max turns available: {req.max_turns}",
    ]
    if req.prior_system_prompt:
        parts.append(
            "Previous version of the system prompt — keep what worked, "
            "fix the issues below:\n"
            f"```text\n{req.prior_system_prompt}\n```"
        )
    if req.critic_feedback:
        parts.append(f"Critic feedback on the prior conversation:\n{req.critic_feedback}")
    if req.suggestions:
        bullets = "\n".join(f"- {s}" for s in req.suggestions)
        parts.append(f"Prioritize these specific suggestions:\n{bullets}")
    parts.append(
        "Produce the next version of the assistant's system prompt now, "
        "inside a ```text ... ``` block."
    )
    return "\n\n".join(parts)


async def generate(
    model: str, req: ConversationGenerationRequest
) -> tuple[str, float]:
    """Generate an assistant system prompt. Returns (prompt_text, cost_usd)."""
    final: str | None = None
    cost_usd = 0.0
    async for msg in query(
        prompt=_build_user_message(req),
        options=ClaudeAgentOptions(
            system_prompt=CONVERSATION_GENERATOR_SYSTEM,
            model=model,
            tools=[],
        ),
    ):
        if isinstance(msg, ResultMessage):
            if msg.is_error:
                raise RuntimeError(f"Conversation generator run failed: {msg.result!r}")
            final = msg.result
            cost_usd = msg.total_cost_usd or 0.0
    if not final:
        raise RuntimeError("Conversation generator produced no result.")
    prompt = _extract_prompt(final)
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise RuntimeError(
            f"Generator output exceeded {MAX_PROMPT_BYTES} bytes "
            f"({len(prompt)} chars) — likely runaway generation."
        )
    return prompt, cost_usd
