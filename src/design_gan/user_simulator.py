"""User simulator agent: plays a user pursuing a specific goal.

Mirrors the generator/critic pattern — one Claude instance via the Agent SDK
with a constrained JSON output contract so the orchestrator can tell whether
the simulated user is satisfied or has another turn to offer.

Two entry points:

* ``opening_turn`` — asks the user sim to kick off a conversation from a cold
  goal. Produces only an opening message (no satisfaction flag — they
  haven't seen an assistant response yet).
* ``followup_turn`` — given the transcript so far, asks the user sim to either
  respond to the assistant or declare that their goal is met. Returns
  ``(reply, satisfied)`` where ``satisfied=True`` short-circuits the
  conversation loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from pydantic import BaseModel, Field, ValidationError

Turn = dict  # {"role": "user" | "assistant", "content": str}


_USER_SIM_SYSTEM = """You are simulating a realistic user who has a specific goal \
and is about to use (or is currently using) an AI assistant to pursue it.

You are NOT the assistant. You never answer as the assistant. Your job is to
speak naturally *as the user* — asking questions, following up, pushing back
if answers are vague or off-target, and declaring when your goal has been met.

Stay in character:
- Write in first person ("I need...", "Can you...", "Thanks, but...").
- Match the register of a real person — don't hedge excessively or use
  corporate phrasing. Don't role-play beyond what the goal requires.
- Keep turns to 1-3 sentences unless the goal explicitly calls for detail.
- If the assistant's last response satisfied your goal, say so briefly and
  set ``satisfied=true``. Don't keep the conversation alive out of politeness.
- If the assistant was vague, generic, or missed your intent, push back
  concretely ("I asked about X; you answered Y").

Output contract:
- Return ONLY one fenced JSON code block: ```json { ... } ```
- No prose before or after the block.
- JSON schema:
  {
    "message": "what you say to the assistant as the user",
    "satisfied": true | false
  }
"""

_OPENING_SYSTEM = """You are simulating a realistic user kicking off a conversation \
with an AI assistant to pursue a specific goal.

Write the first user turn — the opening message — as a real person would
phrase it. Be concrete and show your actual intent. Don't list every detail;
people usually start with one concrete ask and elaborate later if needed.

Output contract:
- Return ONLY one fenced JSON code block: ```json { ... } ```
- No prose before or after the block.
- JSON schema:
  {
    "message": "your opening message to the assistant, in first person"
  }
"""


class _OpeningResponse(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class _FollowupResponse(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    satisfied: bool


@dataclass
class SimulatedTurn:
    role: Literal["user"]
    content: str
    satisfied: bool  # only meaningful on non-opening turns
    cost_usd: float


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> str:
    m = _JSON_BLOCK.search(text)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    raise ValueError("No JSON object found in user-simulator response.")


async def _run_agent(model: str, system_prompt: str, user_message: str) -> tuple[str, float]:
    final: str | None = None
    cost = 0.0
    async for msg in query(
        prompt=user_message,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            tools=[],
            max_turns=2,
        ),
    ):
        if isinstance(msg, ResultMessage):
            if msg.is_error:
                raise RuntimeError(f"User-simulator run failed: {msg.result!r}")
            final = msg.result
            cost = msg.total_cost_usd or 0.0
    if not final:
        raise RuntimeError("User-simulator produced no result.")
    return final, cost


def _format_transcript(transcript: list[Turn]) -> str:
    if not transcript:
        return "(no turns yet)"
    lines = []
    for t in transcript:
        label = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{label}: {t['content']}")
    return "\n\n".join(lines)


async def opening_turn(model: str, goal: str) -> SimulatedTurn:
    """Generate the first user message for a conversation pursuing ``goal``."""
    prompt = (
        f"Your goal in this conversation:\n{goal.strip()}\n\n"
        "Produce your opening message to the assistant."
    )
    last_error: Exception | None = None
    for attempt in range(2):
        raw, cost = await _run_agent(model, _OPENING_SYSTEM, prompt)
        try:
            parsed = _OpeningResponse.model_validate_json(_extract_json(raw))
            return SimulatedTurn(
                role="user", content=parsed.message, satisfied=False, cost_usd=cost
            )
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_error = e
            prompt = (
                f"Your goal in this conversation:\n{goal.strip()}\n\n"
                "Produce your opening message.\n\n"
                "IMPORTANT: return ONLY a fenced ```json ... ``` block matching "
                'the schema {"message": "..."}.'
            )
    raise RuntimeError(f"User sim opening failed after 2 attempts: {last_error}")


async def followup_turn(
    model: str, goal: str, transcript: list[Turn]
) -> SimulatedTurn:
    """Generate the next user message given the conversation so far.

    The last entry in ``transcript`` must be an assistant turn.
    """
    if not transcript or transcript[-1]["role"] != "assistant":
        raise ValueError("followup_turn expects the transcript to end on an assistant turn.")

    prompt = (
        f"Your goal in this conversation:\n{goal.strip()}\n\n"
        "Conversation so far:\n\n"
        f"{_format_transcript(transcript)}\n\n"
        "Produce your next user turn. Set satisfied=true only if the assistant's "
        "most recent response actually met your goal."
    )
    last_error: Exception | None = None
    for attempt in range(2):
        raw, cost = await _run_agent(model, _USER_SIM_SYSTEM, prompt)
        try:
            parsed = _FollowupResponse.model_validate_json(_extract_json(raw))
            return SimulatedTurn(
                role="user",
                content=parsed.message,
                satisfied=parsed.satisfied,
                cost_usd=cost,
            )
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_error = e
            prompt = (
                f"Your goal in this conversation:\n{goal.strip()}\n\n"
                "Conversation so far:\n\n"
                f"{_format_transcript(transcript)}\n\n"
                "Produce your next user turn.\n\n"
                "IMPORTANT: return ONLY a fenced ```json ... ``` block matching "
                'the schema {"message": "...", "satisfied": true|false}.'
            )
    raise RuntimeError(f"User sim followup failed after 2 attempts: {last_error}")
