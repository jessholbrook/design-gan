"""Pytest fixtures and global setup.

We forcibly drop ANTHROPIC_API_KEY so no tests accidentally hit the live API
even if the developer has it exported."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
