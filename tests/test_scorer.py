"""Unit tests for scorer.py — SUS math, axe weighting, composite blending."""

from __future__ import annotations

import pytest

from design_gan.scorer import axe_penalty, score, sus_score


class TestSusScore:
    def test_all_fives_on_positives_and_ones_on_negatives_is_max(self):
        # Positive (odd, 1-indexed) items score (x-1); negative (even) items score (5-x).
        # Best possible: 5 on odd items, 1 on even items -> all 4s -> 40 * 2.5 = 100.
        answers = [5, 1, 5, 1, 5, 1, 5, 1, 5, 1]
        assert sus_score(answers) == 100.0

    def test_all_ones_on_positives_and_fives_on_negatives_is_zero(self):
        answers = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
        assert sus_score(answers) == 0.0

    def test_all_threes_is_fifty(self):
        assert sus_score([3] * 10) == 50.0

    def test_alternating_pattern_demo(self):
        # Matches demo.py iter 3: SUS = 75
        assert sus_score([4, 2, 4, 2, 4, 2, 4, 2, 4, 2]) == 75.0

    def test_demo_iter4_is_95(self):
        assert sus_score([5, 1, 5, 1, 4, 2, 5, 1, 5, 1]) == 95.0

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 10"):
            sus_score([3] * 9)
        with pytest.raises(ValueError, match="exactly 10"):
            sus_score([3] * 11)

    def test_out_of_range_raises(self):
        bad = [3] * 10
        bad[4] = 6
        with pytest.raises(ValueError, match="out of range"):
            sus_score(bad)
        bad[4] = 0
        with pytest.raises(ValueError, match="out of range"):
            sus_score(bad)


class TestAxePenalty:
    def test_empty_is_zero(self):
        assert axe_penalty([]) == 0.0

    def test_impact_weights(self):
        cases = [
            ({"impact": "critical", "nodes": [{}]}, 5.0),
            ({"impact": "serious", "nodes": [{}]}, 3.0),
            ({"impact": "moderate", "nodes": [{}]}, 1.5),
            ({"impact": "minor", "nodes": [{}]}, 0.5),
            ({"impact": "", "nodes": [{}]}, 0.5),  # unknown falls back to minor weight
            ({"nodes": [{}]}, 0.5),  # missing impact also falls back
        ]
        for v, expected in cases:
            assert axe_penalty([v]) == expected

    def test_node_count_multiplies_weight(self):
        violation = {"impact": "critical", "nodes": [{}, {}, {}]}
        assert axe_penalty([violation]) == 15.0

    def test_missing_nodes_treated_as_one(self):
        assert axe_penalty([{"impact": "critical"}]) == 5.0

    def test_penalty_capped_at_thirty(self):
        # 10 critical violations with 10 nodes each = 500 raw, but cap at 30.
        many = [{"impact": "critical", "nodes": [{}] * 10}] * 10
        assert axe_penalty(many) == 30.0


class TestScore:
    def test_composite_is_sus_minus_penalty(self):
        result = score([3] * 10, [{"impact": "serious", "nodes": [{}]}])
        assert result.sus == 50.0
        assert result.axe_penalty == 3.0
        assert result.composite == 47.0

    def test_composite_floor_is_zero(self):
        # SUS 0 + large penalty -> composite clamped to 0, not negative.
        result = score([1, 5] * 5, [{"impact": "critical", "nodes": [{}] * 10}])
        assert result.sus == 0.0
        assert result.composite == 0.0

    def test_composite_ceiling_is_one_hundred(self):
        result = score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1], [])
        assert result.sus == 100.0
        assert result.composite == 100.0

    def test_breakdown_included(self):
        result = score([3] * 10, [{"impact": "minor"}])
        assert result.breakdown["sus_answers"] == [3] * 10
        assert result.breakdown["axe_violation_count"] == 1
