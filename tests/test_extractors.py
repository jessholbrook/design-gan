"""Tests for the HTML and JSON extraction helpers inside generator/critic."""

from __future__ import annotations

import pytest

from design_gan.critic import _extract_json, _summarize_axe, _truncate
from design_gan.generator import _build_user_message, _extract_html, GenerationRequest


class TestHtmlSizeCap:
    def test_html_under_cap_ok(self):
        from design_gan.generator import MAX_HTML_BYTES
        # Sanity: the cap is non-trivial.
        assert MAX_HTML_BYTES > 10_000


class TestExtractHtml:
    def test_extracts_from_fenced_block(self):
        raw = "here you go:\n```html\n<!doctype html><html></html>\n```\nlet me know!"
        assert _extract_html(raw) == "<!doctype html><html></html>"

    def test_extracts_case_insensitive_fence(self):
        raw = "```HTML\n<p>hi</p>\n```"
        assert _extract_html(raw) == "<p>hi</p>"

    def test_extracts_multiline(self):
        raw = "```html\n<html>\n<body>\nx\n</body>\n</html>\n```"
        assert "<body>" in _extract_html(raw)

    def test_fallback_returns_whole_text_when_no_fence(self):
        raw = "<!doctype html><html></html>"
        assert _extract_html(raw) == "<!doctype html><html></html>"

    def test_strips_whitespace(self):
        assert _extract_html("   <html></html>   ") == "<html></html>"


class TestBuildUserMessage:
    def test_brief_only(self):
        msg = _build_user_message(GenerationRequest(brief="a cycling site"))
        assert "Brief: a cycling site" in msg
        assert "Produce the next version" in msg

    def test_with_prior_html(self):
        msg = _build_user_message(
            GenerationRequest(brief="b", prior_html="<p>old</p>")
        )
        assert "<p>old</p>" in msg
        assert "Keep what works" in msg

    def test_with_feedback(self):
        msg = _build_user_message(
            GenerationRequest(brief="b", critic_feedback="too cramped")
        )
        assert "Critic feedback:" in msg
        assert "too cramped" in msg

    def test_with_suggestions(self):
        msg = _build_user_message(
            GenerationRequest(
                brief="b", suggestions=["add CTA", "increase contrast"]
            )
        )
        assert "- add CTA" in msg
        assert "- increase contrast" in msg


class TestExtractJson:
    def test_extracts_from_fenced_json_block(self):
        raw = '```json\n{"sus": [1,2,3]}\n```'
        assert _extract_json(raw) == '{"sus": [1,2,3]}'

    def test_extracts_from_unlabeled_fence(self):
        raw = '```\n{"x": 1}\n```'
        assert _extract_json(raw) == '{"x": 1}'

    def test_fallback_on_braces(self):
        raw = 'blah {"x": 1} trailing'
        assert _extract_json(raw) == '{"x": 1}'

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here")

    def test_handles_nested_objects(self):
        raw = '```json\n{"a": {"b": 2}}\n```'
        assert _extract_json(raw) == '{"a": {"b": 2}}'


class TestSummarizeAxe:
    def test_empty_reports_no_violations(self):
        assert "no violations" in _summarize_axe([])

    def test_includes_impact_and_count(self):
        v = [{"impact": "critical", "id": "color-contrast", "help": "h",
              "nodes": [{}, {}]}]
        s = _summarize_axe(v)
        assert "critical" in s
        assert "color-contrast" in s
        assert "2 node" in s

    def test_truncates_long_lists(self):
        many = [
            {"impact": "minor", "id": f"rule-{i}", "help": "h", "nodes": [{}]}
            for i in range(25)
        ]
        s = _summarize_axe(many)
        assert "10 more" in s
        assert "rule-0" in s
        assert "rule-24" not in s  # beyond cap


class TestTruncate:
    def test_shorter_than_limit_unchanged(self):
        assert _truncate("hi", 100) == "hi"

    def test_longer_is_cut_with_notice(self):
        out = _truncate("x" * 100, 10)
        assert out.startswith("x" * 10)
        assert "truncated" in out
