"""Tests for conversation_generator — prompt extraction + request building."""

from __future__ import annotations

import pytest

from design_gan.conversation_generator import (
    ConversationGenerationRequest,
    MAX_PROMPT_BYTES,
    _build_user_message,
    _extract_prompt,
)


class TestExtractPrompt:
    def test_extracts_text_fence(self):
        raw = "```text\nYou are a helpful assistant.\n```"
        assert _extract_prompt(raw) == "You are a helpful assistant."

    def test_extracts_plaintext_fence(self):
        raw = "```plaintext\nYou are a sharp editor.\n```"
        assert _extract_prompt(raw) == "You are a sharp editor."

    def test_extracts_unlabeled_fence(self):
        raw = "```\nJust be direct.\n```"
        assert _extract_prompt(raw) == "Just be direct."

    def test_no_fence_returns_stripped(self):
        assert _extract_prompt("  raw text  ") == "raw text"


class TestBuildUserMessage:
    def test_goal_only(self):
        msg = _build_user_message(ConversationGenerationRequest(goal="Help me with X"))
        assert "Help me with X" in msg
        assert "Produce the next version" in msg

    def test_prior_prompt_is_included(self):
        msg = _build_user_message(ConversationGenerationRequest(
            goal="g", prior_system_prompt="You are...",
        ))
        assert "Previous version of the system prompt" in msg
        assert "You are..." in msg

    def test_feedback_and_suggestions(self):
        msg = _build_user_message(ConversationGenerationRequest(
            goal="g", critic_feedback="too vague",
            suggestions=["be specific", "cut boilerplate"],
        ))
        assert "too vague" in msg
        assert "- be specific" in msg

    def test_max_turns_surfaced(self):
        msg = _build_user_message(ConversationGenerationRequest(goal="g", max_turns=3))
        assert "Max turns available: 3" in msg


class TestMaxPromptBytes:
    def test_cap_is_reasonable(self):
        assert MAX_PROMPT_BYTES >= 4000
