"""Tests for Elo rating calculations.

Pure unit tests - no external dependencies.
"""

from elo import RatingPair, _probability_a_beats_b, update_ratings
from match_types import Outcome


class TestProbabilityABeatsB:
    """Tests for the probability calculation (private, but worth testing directly)."""

    def test_should_give_50_percent_for_equal_ratings(self):
        """Should return 0.5 when both players have equal ratings."""
        assert _probability_a_beats_b(RatingPair(100.0, 100.0)) == 0.5

    def test_should_give_higher_probability_for_higher_rated_player(self):
        """Should return >0.5 when player A has higher rating."""
        prob = _probability_a_beats_b(RatingPair(200.0, 100.0))
        assert prob > 0.5


class TestUpdateRatings:
    """Tests for a single Elo update step."""

    def test_should_reward_upset_more_than_expected_win(self):
        """Should give larger rating change for upsets vs expected results.

        Sanity check: underdog winning should gain more than favorite winning.
        """
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
        """Should transfer points symmetrically (winner gains = loser loses)."""
        old_a, old_b = 150.0, 120.0
        new = update_ratings(RatingPair(old_a, old_b), Outcome.WIN_A, k=32)

        gain = new.a - old_a
        loss = old_b - new.b
        assert abs(gain - loss) < 1e-10
