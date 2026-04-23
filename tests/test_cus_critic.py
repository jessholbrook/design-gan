"""Tests for the CUS critic path and scorer variant."""

from __future__ import annotations

import pytest

from design_gan.critic import (
    CUS_ITEMS,
    CUS_TRIO,
    CUS_TONE_CRITIC,
    CUS_TRUST_CRITIC,
    CUS_USABILITY_CRITIC,
    DEFAULT_CUS_CRITIC,
    _build_cus_user_message,
    _summarize_objective,
)
from design_gan.scorer import score, score_from_penalty, sus_score


class TestCusItems:
    def test_ten_items(self):
        assert len(CUS_ITEMS) == 10

    def test_alternating_pattern_matches_sus_math(self):
        # Odd (1-indexed) items positive, even negative. Sanity-check the
        # framing holds against the scorer math.
        # All-positive answers (5,1,5,1,...) should give SUS=100.
        assert sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]) == 100.0


class TestCusTrio:
    def test_three_distinct_profiles(self):
        names = {c.name for c in CUS_TRIO}
        assert names == {
            "Conversation usability", "Tone & register", "Trust & specificity"
        }

    def test_default_is_usability(self):
        assert DEFAULT_CUS_CRITIC is CUS_USABILITY_CRITIC


class TestSummarizeObjective:
    def test_reports_key_fields(self):
        m = {
            "assistant_turn_count": 3,
            "unresolved": False,
            "boilerplate_count": 2,
            "repetition_hits": [{"between_turns": (0, 1), "overlap": 0.7}],
            "length_bloat_hits": [],
        }
        s = _summarize_objective(m)
        assert "assistant turns: 3" in s
        assert "boilerplate hits: 2" in s
        assert "repetition hits: 1" in s
        assert "unresolved: False" in s


class TestBuildCusUserMessage:
    def test_includes_goal_transcript_prompt_and_metrics(self):
        msg = _build_cus_user_message(
            goal="Make me cold brew",
            transcript=[
                {"role": "user", "content": "how?"},
                {"role": "assistant", "content": "Steep 1:4 for 18h."},
            ],
            objective_metrics={
                "assistant_turn_count": 1, "unresolved": False,
                "boilerplate_count": 0, "repetition_hits": [],
                "length_bloat_hits": [],
            },
            assistant_system_prompt="Be concrete.",
        )
        assert "Make me cold brew" in msg
        assert "User: how?" in msg
        assert "Assistant: Steep 1:4 for 18h." in msg
        assert "Be concrete." in msg
        assert "CUS items" in msg


class TestScoreFromPenalty:
    def test_composite_is_sus_minus_penalty(self):
        # Neutral answers: SUS = 50. Penalty 10 -> composite 40.
        s = score_from_penalty([3] * 10, 10.0)
        assert s.sus == 50.0
        assert s.axe_penalty == 10.0
        assert s.composite == 40.0

    def test_negative_penalty_clamped_to_zero(self):
        s = score_from_penalty([3] * 10, -5.0)
        assert s.axe_penalty == 0.0

    def test_penalty_capped_at_thirty(self):
        s = score_from_penalty([3] * 10, 1000.0)
        assert s.axe_penalty == 30.0

    def test_composite_floor_is_zero(self):
        s = score_from_penalty([1, 5] * 5, 30.0)  # SUS 0 - 30 penalty -> floor 0
        assert s.composite == 0.0

    def test_composite_ceiling_is_one_hundred(self):
        s = score_from_penalty([5, 1] * 5, 0.0)
        assert s.composite == 100.0

    def test_breakdown_carries_answers(self):
        ans = [3] * 10
        s = score_from_penalty(ans, 5.0, breakdown={"note": "cus"})
        assert s.breakdown["answers"] == ans
        assert s.breakdown["note"] == "cus"

    def test_symmetry_with_score(self):
        # The old score() path should produce the same composite as score_from_penalty
        # when given equivalent inputs.
        violations = [{"impact": "serious", "nodes": [{}, {}]}]  # penalty = 6
        s1 = score([3] * 10, violations)
        s2 = score_from_penalty([3] * 10, s1.axe_penalty)
        assert s1.composite == s2.composite
