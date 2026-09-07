"""Regression coverage for tool-free screenshot critique on hosted runtimes."""

import base64
import json

import pytest
from claude_agent_sdk import ResultMessage

from design_gan import critic


def result(text, *, is_error=False, cost=0.02):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="test",
        total_cost_usd=cost,
        result=text,
    )


@pytest.fixture
def screenshot(tmp_path):
    path = tmp_path / "screenshot.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    return path


@pytest.mark.asyncio
async def test_screenshot_is_attached_without_tools_or_permission_bypass(monkeypatch, screenshot):
    async def fake_query(*, prompt, options):
        assert options.tools == []
        assert options.allowed_tools == []
        assert options.add_dirs == []
        assert options.permission_mode is None
        assert options.max_turns == 4
        messages = [message async for message in prompt]
        assert len(messages) == 1
        assert messages[0]["type"] == "user"
        assert messages[0]["message"]["role"] == "user"
        image, text = messages[0]["message"]["content"]
        assert image["type"] == "image"
        assert image["source"]["type"] == "base64"
        assert image["source"]["media_type"] == "image/png"
        assert base64.b64decode(image["source"]["data"]) == screenshot.read_bytes()
        assert text == {"type": "text", "text": "Assess this page"}
        yield result("assessment")

    monkeypatch.setattr(critic, "query", fake_query)
    assert await critic._run_once("model", "system", "Assess this page", screenshot) == (
        "assessment", 0.02
    )


@pytest.mark.asyncio
async def test_retry_reattaches_screenshot_and_accumulates_cost(monkeypatch, screenshot):
    calls = []
    payload = {"sus": [3] * 10, "feedback": "Assessment", "suggestions": ["Improve contrast"]}

    async def fake_query(*, prompt, options):
        messages = [message async for message in prompt]
        calls.append(messages[0]["message"]["content"])
        yield result("invalid JSON" if len(calls) == 1 else json.dumps(payload))

    monkeypatch.setattr(critic, "query", fake_query)
    response, cost = await critic.critique(
        "model", screenshot, "<main>Page</main>", [], "A useful page"
    )
    assert response.sus == [3] * 10
    assert cost == pytest.approx(0.04)
    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]
    assert "previous response was not valid JSON" in calls[1][1]["text"]
    assert "Read tool" not in calls[0][1]["text"]
    assert str(screenshot.parent) not in calls[0][1]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["api_error", "no_result", "missing_screenshot"])
async def test_failures_do_not_become_successful_critiques(monkeypatch, screenshot, failure):
    async def fake_query(*, prompt, options):
        if failure == "missing_screenshot":
            pytest.fail("Missing screenshot should fail before an API call")
        if failure == "api_error":
            yield result("API unavailable", is_error=True)

    monkeypatch.setattr(critic, "query", fake_query)
    if failure == "missing_screenshot":
        screenshot = screenshot.with_name("missing.png")
    expected = FileNotFoundError if failure == "missing_screenshot" else RuntimeError
    with pytest.raises(expected):
        await critic._run_once("model", "system", "Assess", screenshot)
