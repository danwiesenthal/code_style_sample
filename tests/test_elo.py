"""Tests for Elo rating calculations.

Pure unit tests - no external dependencies.
"""

import pytest

from elo import (
    PointsPair,
    RatingPair,
    _points_from_outcome,
    _probability_a_beats_b,
    update_ratings,
)
from match_types import Outcome


class TestProbabilityABeatsB:
    """Tests for the probability calculation."""

    def test_should_give_50_percent_for_equal_ratings(self):
        """Should return 0.5 when both players have equal ratings."""
        assert _probability_a_beats_b(RatingPair(100.0, 100.0)) == 0.5

    def test_should_give_higher_probability_for_higher_rated_player(self):
        """Should return >0.5 when player A has higher rating."""
        prob = _probability_a_beats_b(RatingPair(200.0, 100.0))
        assert prob > 0.5


class TestPointsFromOutcome:
    """Tests for converting an outcome to the points each player earns."""

    def test_should_give_winner_1_and_loser_0(self):
        """Should give 1 point to the winner and none to the loser."""
        assert _points_from_outcome(Outcome.WIN_A) == PointsPair(1.0, 0.0)
        assert _points_from_outcome(Outcome.WIN_B) == PointsPair(0.0, 1.0)

    def test_should_split_points_for_tie(self):
        """Should give each player half a point for a tie."""
        assert _points_from_outcome(Outcome.TIE) == PointsPair(0.5, 0.5)


class TestUpdateRatings:
    """Tests for a single Elo update step."""

    def test_should_reward_upset_more_than_expected_win(self):
        """Should give larger rating change for upsets vs expected results."""
        # Upset: underdog A (100) beats favorite B (200)
        upset = update_ratings(RatingPair(100.0, 200.0), Outcome.WIN_A, k=32)
        upset_gain = upset.a - 100.0

        # Expected: favorite A (200) beats underdog B (100)
        expected = update_ratings(RatingPair(200.0, 100.0), Outcome.WIN_A, k=32)
        expected_gain = expected.a - 200.0

        assert upset_gain > expected_gain  # Upsets rewarded more

    def test_should_not_change_ratings_for_tie_between_equals(self):
        """Should leave ratings unchanged when equals tie (expected result)."""
        new = update_ratings(RatingPair(100.0, 100.0), Outcome.TIE, k=32)
        assert new.a == 100.0
        assert new.b == 100.0

    def test_should_be_zero_sum(self):
        """Should be zero-sum: A's rating gain equals B's rating loss."""
        old_a, old_b = 150.0, 120.0
        new = update_ratings(RatingPair(old_a, old_b), Outcome.WIN_A, k=32)

        gain = new.a - old_a
        loss = old_b - new.b
        assert gain == pytest.approx(loss)
