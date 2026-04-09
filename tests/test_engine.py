"""Unit tests for the pure game engine logic."""

import pytest

from negotiate.engine import (
    GUARANTEED_ROUNDS,
    NO_DEAL_PENALTY,
    RESOURCE_TYPES,
    TARGET_VALUE,
    build_system_prompt,
    build_turn_prompt,
    generate_scenario,
    max_possible,
    score_split,
    validate_offer,
)


# -- generate_scenario ---------------------------------------------------------


class TestGenerateScenario:
    def test_deterministic(self):
        s1 = generate_scenario(42)
        s2 = generate_scenario(42)
        assert s1 == s2

    def test_different_seeds_differ(self):
        s1 = generate_scenario(100)
        s2 = generate_scenario(200)
        assert s1 != s2

    def test_pool_in_range(self):
        s = generate_scenario(7)
        for r in RESOURCE_TYPES:
            assert 1 <= s["pool"][r] <= 10

    def test_valuations_sum_to_target(self):
        for seed in range(20):
            s = generate_scenario(seed)
            total_a = sum(s["valuations_a"][r] * s["pool"][r] for r in RESOURCE_TYPES)
            total_b = sum(s["valuations_b"][r] * s["pool"][r] for r in RESOURCE_TYPES)
            assert total_a == TARGET_VALUE
            assert total_b == TARGET_VALUE

    def test_valuations_at_least_one(self):
        for seed in range(20):
            s = generate_scenario(seed)
            for r in RESOURCE_TYPES:
                assert s["valuations_a"][r] >= 1
                assert s["valuations_b"][r] >= 1

    def test_valuations_asymmetric(self):
        for seed in range(20):
            s = generate_scenario(seed)
            assert s["valuations_a"] != s["valuations_b"]

    def test_has_expected_keys(self):
        s = generate_scenario(0)
        assert set(s.keys()) == {"pool", "valuations_a", "valuations_b", "seed"}
        assert s["seed"] == 0


# -- validate_offer ------------------------------------------------------------


class TestValidateOffer:
    def test_valid_offer(self):
        pool = {"books": 5, "hats": 3, "balls": 4}
        offer = {
            "my_share": {"books": 3, "hats": 1, "balls": 2},
            "their_share": {"books": 2, "hats": 2, "balls": 2},
        }
        assert validate_offer(offer, pool) is True

    def test_shares_dont_sum_to_pool(self):
        pool = {"books": 5, "hats": 3, "balls": 4}
        offer = {
            "my_share": {"books": 3, "hats": 1, "balls": 2},
            "their_share": {"books": 3, "hats": 2, "balls": 2},  # books: 3+3=6 != 5
        }
        assert validate_offer(offer, pool) is False

    def test_negative_quantity(self):
        pool = {"books": 5, "hats": 3, "balls": 4}
        offer = {
            "my_share": {"books": 6, "hats": 1, "balls": 2},
            "their_share": {"books": -1, "hats": 2, "balls": 2},
        }
        assert validate_offer(offer, pool) is False

    def test_all_to_one_player(self):
        pool = {"books": 5, "hats": 3, "balls": 4}
        offer = {
            "my_share": {"books": 5, "hats": 3, "balls": 4},
            "their_share": {"books": 0, "hats": 0, "balls": 0},
        }
        assert validate_offer(offer, pool) is True

    def test_missing_resource_defaults_zero(self):
        pool = {"books": 5, "hats": 3, "balls": 4}
        offer = {
            "my_share": {"books": 5, "hats": 3, "balls": 4},
            "their_share": {},  # all default to 0
        }
        assert validate_offer(offer, pool) is True

    def test_empty_offer(self):
        pool = {"books": 5, "hats": 3, "balls": 4}
        offer = {"my_share": {}, "their_share": {}}
        # 0 + 0 != 5, should fail
        assert validate_offer(offer, pool) is False


# -- score_split / max_possible ------------------------------------------------


class TestScoring:
    def test_score_split_basic(self):
        vals = {"books": 10, "hats": 5, "balls": 2}
        split = {"books": 3, "hats": 1, "balls": 4}
        assert score_split(vals, split) == 10 * 3 + 5 * 1 + 2 * 4  # 43

    def test_score_split_empty(self):
        vals = {"books": 10, "hats": 5, "balls": 2}
        assert score_split(vals, {}) == 0

    def test_max_possible(self):
        vals = {"books": 10, "hats": 5, "balls": 2}
        pool = {"books": 5, "hats": 3, "balls": 4}
        assert max_possible(vals, pool) == 10 * 5 + 5 * 3 + 2 * 4  # 73

    def test_normalized_score_range(self):
        """A valid deal should produce scores in [0.0, 1.0]."""
        s = generate_scenario(42)
        pool = s["pool"]
        vals = s["valuations_a"]
        mx = max_possible(vals, pool)
        # Give player everything
        full_score = score_split(vals, pool) / mx
        assert full_score == pytest.approx(1.0)
        # Give player nothing
        nothing = {r: 0 for r in RESOURCE_TYPES}
        zero_score = score_split(vals, nothing) / mx
        assert zero_score == pytest.approx(0.0)


# -- build_system_prompt -------------------------------------------------------


class TestBuildSystemPrompt:
    def test_includes_role(self):
        s = generate_scenario(1)
        prompt = build_system_prompt("A", s["pool"], s["valuations_a"])
        assert "Player A" in prompt

    def test_includes_valuations(self):
        pool = {"books": 3, "hats": 2, "balls": 5}
        vals = {"books": 10, "hats": 20, "balls": 6}
        prompt = build_system_prompt("B", pool, vals)
        assert "books: 10 points each" in prompt
        assert "3 books" in prompt

    def test_includes_custom_strategy(self):
        s = generate_scenario(1)
        prompt = build_system_prompt("A", s["pool"], s["valuations_a"], custom_strategy="Be greedy.")
        assert "ADDITIONAL STRATEGY INSTRUCTIONS" in prompt
        assert "Be greedy." in prompt

    def test_no_strategy_section_when_none(self):
        s = generate_scenario(1)
        prompt = build_system_prompt("A", s["pool"], s["valuations_a"])
        assert "ADDITIONAL STRATEGY INSTRUCTIONS" not in prompt


# -- build_turn_prompt ---------------------------------------------------------


class TestBuildTurnPrompt:
    def test_empty_history(self):
        pool = {"books": 3, "hats": 2, "balls": 5}
        prompt = build_turn_prompt([], "A", 1, pool)
        assert "Round 1 —" in prompt
        assert "Player A" in prompt
        assert "NEGOTIATION HISTORY" not in prompt

    def test_overtime_warning(self):
        pool = {"books": 3, "hats": 2, "balls": 5}
        prompt = build_turn_prompt([], "A", GUARANTEED_ROUNDS + 1, pool)
        assert "OVERTIME" in prompt
        assert "-0.5" in prompt

    def test_no_overtime_warning_in_guaranteed_rounds(self):
        pool = {"books": 3, "hats": 2, "balls": 5}
        prompt = build_turn_prompt([], "A", GUARANTEED_ROUNDS, pool)
        assert "OVERTIME" not in prompt

    def test_history_shows_proposals(self):
        history = [
            {
                "player": "A",
                "action": "propose",
                "message": "Fair split?",
                "offer": {
                    "my_share": {"books": 2, "hats": 1, "balls": 3},
                    "their_share": {"books": 1, "hats": 1, "balls": 2},
                },
            }
        ]
        pool = {"books": 3, "hats": 2, "balls": 5}
        # From B's perspective, A's proposal should show "they keep" / "you get"
        prompt = build_turn_prompt(history, "B", 1, pool)
        assert "Player A proposed" in prompt
        assert "they keep" in prompt

    def test_history_self_perspective(self):
        history = [
            {
                "player": "A",
                "action": "propose",
                "message": "My offer",
                "offer": {
                    "my_share": {"books": 2, "hats": 1, "balls": 3},
                    "their_share": {"books": 1, "hats": 1, "balls": 2},
                },
            }
        ]
        pool = {"books": 3, "hats": 2, "balls": 5}
        # From A's own perspective
        prompt = build_turn_prompt(history, "A", 2, pool)
        assert "Player A (you) proposed" in prompt
        assert "you keep" in prompt
