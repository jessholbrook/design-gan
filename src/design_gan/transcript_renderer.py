"""Transcript renderer: run the multi-turn conversation between assistant and user-sim.

Analogous to ``renderer.py`` for design runs — given the evolving artifact (the
assistant's system prompt) it executes the rendering (a real conversation) and
returns the produced artifact (the transcript) plus objective metrics that
anchor the subjective CUS score.

Objective metrics (the "axe-core" of conversations):

* **repetition** — consecutive assistant responses whose lexical overlap is
  high. Flags loops and padding.
* **boilerplate** — classic LLM tells ("Certainly!", "Great question!",
  "I'd be happy to help", etc.) in assistant responses.
* **length_bloat** — assistant responses that go well over a token budget
  appropriate to the conversation's complexity.
* **unresolved** — the conversation hit the turn cap without the user sim
  declaring satisfaction.

Each of the above contributes a weighted penalty, capped at 30 total (parallel
to scorer.axe_penalty).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from . import user_simulator

Turn = dict  # {"role": "user" | "assistant", "content": str, "cost_usd": float}


@dataclass
class TranscriptResult:
    transcript: list[Turn]
    assistant_system_prompt: str
    objective_metrics: dict[str, Any]
    objective_penalty: float  # 0-30, already weighted + capped
    total_cost_usd: float
    satisfied: bool  # did the user-sim declare the goal met?
    turns_taken: int  # assistant turns only (user turns == assistant turns or +1)


# --- Objective metric helpers ------------------------------------------------

# Common boilerplate phrases that signal low-quality LLM output. Case-insensitive.
_BOILERPLATE_PATTERNS = [
    r"\bcertainly!",
    r"\bof course!",
    r"\babsolutely!",
    r"\bgreat question!",
    r"\bthat'?s a great question",
    r"\bi'?d be (happy|glad) to help",
    r"\bi hope (this|that) helps",
    r"\blet me know if (you have|there('|)s) (any|anything)",
    r"\bi'?m just an? ai",
    r"\bas an ai",
    r"\bi don'?t have personal",
]

_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# A rough token estimator — 4 chars/token is close enough for budgeting.
_CHARS_PER_TOKEN = 4
_LENGTH_BLOAT_TOKEN_THRESHOLD = 500

# Per-violation penalty weights. Summed then capped at 30.
_WEIGHTS = {
    "repetition": 4.0,
    "boilerplate": 1.5,
    "length_bloat": 2.0,
    "unresolved": 5.0,
}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[\w']+\b", text.lower()) if len(w) > 2]


def _lexical_overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets, ignoring very short tokens."""
    wa, wb = set(_words(a)), set(_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _count_boilerplate(text: str) -> int:
    return len(_BOILERPLATE_RE.findall(text))


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def compute_objective_metrics(
    transcript: list[Turn], *, satisfied: bool
) -> tuple[dict[str, Any], float]:
    """Compute per-transcript objective metrics and a weighted penalty (0-30)."""
    assistant_turns = [t for t in transcript if t["role"] == "assistant"]

    # Repetition: adjacent assistant turns with overlap >= 0.6.
    repetition_hits: list[dict[str, Any]] = []
    for i in range(1, len(assistant_turns)):
        ov = _lexical_overlap(
            assistant_turns[i - 1]["content"], assistant_turns[i]["content"]
        )
        if ov >= 0.6:
            repetition_hits.append({"between_turns": (i - 1, i), "overlap": round(ov, 3)})

    # Boilerplate: count matches across all assistant turns.
    boilerplate_count = sum(
        _count_boilerplate(t["content"]) for t in assistant_turns
    )

    # Length bloat: assistant turns estimated >500 tokens.
    bloat_hits = [
        {"turn": idx, "est_tokens": _estimate_tokens(t["content"])}
        for idx, t in enumerate(assistant_turns)
        if _estimate_tokens(t["content"]) > _LENGTH_BLOAT_TOKEN_THRESHOLD
    ]

    penalty = 0.0
    penalty += _WEIGHTS["repetition"] * len(repetition_hits)
    penalty += _WEIGHTS["boilerplate"] * boilerplate_count
    penalty += _WEIGHTS["length_bloat"] * len(bloat_hits)
    if not satisfied:
        penalty += _WEIGHTS["unresolved"]
    penalty = min(penalty, 30.0)

    metrics = {
        "repetition_hits": repetition_hits,
        "boilerplate_count": boilerplate_count,
        "length_bloat_hits": bloat_hits,
        "unresolved": not satisfied,
        "assistant_turn_count": len(assistant_turns),
    }
    return metrics, penalty


# --- Assistant turn generator -------------------------------------------------


async def _run_assistant_turn(
    model: str, system_prompt: str, transcript: list[Turn]
) -> tuple[str, float]:
    """Ask the evolving-artifact assistant for its next reply given the transcript.

    We stream the whole conversation so the assistant sees its own prior
    replies naturally; the agent SDK's ``query`` is single-shot but we fake
    multi-turn context by including the transcript in the prompt.
    """
    lines = []
    for t in transcript:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
    rendered = "\n\n".join(lines)
    # Drop the final "Assistant:" slot; the model generates what comes next.
    prompt = (
        f"Conversation so far:\n\n{rendered}\n\n"
        "Produce the assistant's next reply. Respond directly as the assistant; "
        "do NOT prefix with 'Assistant:' and do NOT generate further user turns."
    )

    final: str | None = None
    cost = 0.0
    async for msg in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            tools=[],
            max_turns=2,
        ),
    ):
        if isinstance(msg, ResultMessage):
            if msg.is_error:
                raise RuntimeError(f"Assistant run failed: {msg.result!r}")
            final = msg.result
            cost = msg.total_cost_usd or 0.0
    if not final:
        raise RuntimeError("Assistant produced no result.")
    # Defensive: strip a leading "Assistant:" if the model decided to add it.
    stripped = final.strip()
    if stripped.lower().startswith("assistant:"):
        stripped = stripped.split(":", 1)[1].strip()
    return stripped, cost


# --- Conversation loop -------------------------------------------------------


async def run_conversation(
    *,
    model: str,
    assistant_system_prompt: str,
    goal: str,
    max_turns: int = 5,
) -> TranscriptResult:
    """Run a full user-sim ↔ assistant conversation, capped at max_turns rounds.

    A "round" is one user turn + one assistant reply. The conversation may end
    early when the user sim declares satisfaction. Always starts with an opening
    user turn.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")

    transcript: list[Turn] = []
    total_cost = 0.0
    satisfied = False

    # Opening user turn.
    opener = await user_simulator.opening_turn(model, goal)
    total_cost += opener.cost_usd
    transcript.append(
        {"role": "user", "content": opener.content, "cost_usd": opener.cost_usd}
    )

    for round_idx in range(1, max_turns + 1):
        # Assistant reply.
        reply, reply_cost = await _run_assistant_turn(
            model, assistant_system_prompt, transcript
        )
        total_cost += reply_cost
        transcript.append(
            {"role": "assistant", "content": reply, "cost_usd": reply_cost}
        )

        # Stop if we've hit the turn cap — don't burn another user-sim call
        # only to throw away its output.
        if round_idx == max_turns:
            break

        # User follow-up; may declare satisfaction.
        followup = await user_simulator.followup_turn(model, goal, transcript)
        total_cost += followup.cost_usd
        transcript.append(
            {"role": "user", "content": followup.content, "cost_usd": followup.cost_usd}
        )
        if followup.satisfied:
            satisfied = True
            break

    metrics, penalty = compute_objective_metrics(transcript, satisfied=satisfied)
    assistant_turn_count = sum(1 for t in transcript if t["role"] == "assistant")

    return TranscriptResult(
        transcript=transcript,
        assistant_system_prompt=assistant_system_prompt,
        objective_metrics=metrics,
        objective_penalty=penalty,
        total_cost_usd=total_cost,
        satisfied=satisfied,
        turns_taken=assistant_turn_count,
    )


def write_transcript_artifacts(result: TranscriptResult, out_dir: Path) -> dict[str, Path]:
    """Persist transcript + metrics to disk, mirroring renderer.write_artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.json"
    metrics_path = out_dir / "metrics.json"
    prompt_path = out_dir / "system_prompt.txt"

    transcript_path.write_text(
        json.dumps(
            {
                "transcript": result.transcript,
                "satisfied": result.satisfied,
                "turns_taken": result.turns_taken,
                "total_cost_usd": result.total_cost_usd,
                "captured_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": result.objective_metrics,
                "penalty": result.objective_penalty,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    prompt_path.write_text(result.assistant_system_prompt, encoding="utf-8")
    return {
        "transcript": transcript_path,
        "metrics": metrics_path,
        "system_prompt": prompt_path,
    }
