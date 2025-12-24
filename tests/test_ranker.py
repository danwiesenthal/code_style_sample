"""Tests for ranker module.

Pure unit tests - no external dependencies.
"""

from elo import RatingPair
from ranker import compute_ratings
from tests.helpers import make_match


class TestComputeRatings:
    """Tests for computing ratings over a sequence of matches."""

    def test_should_return_init_ratings_for_empty_matches(self):
        """Should return initial ratings unchanged if no matches."""
        init = RatingPair(100.0, 100.0)
        result = compute_ratings([], init=init)
        assert result == init

    def test_should_increase_winner_rating(self):
        """Should increase rating for model that wins."""
        matches = [make_match(score_a=1.0, score_b=0.0)]  # A wins
        result = compute_ratings(matches, init=RatingPair(100.0, 100.0))

        assert result.a > 100.0  # Winner gains
        assert result.b < 100.0  # Loser drops

    def test_should_accumulate_across_multiple_matches(self):
        """Should process all matches, accumulating rating changes."""
        # A wins twice
        matches = [
            make_match(score_a=1.0, score_b=0.0),
            make_match(score_a=1.0, score_b=0.0),
        ]
        result = compute_ratings(matches, init=RatingPair(100.0, 100.0))

        # After two wins, A should be even higher
        single_win = compute_ratings(matches[:1], init=RatingPair(100.0, 100.0))
        assert result.a > single_win.a

    def test_should_weight_later_matches_more(self):
        """Should let the later of two opposite results count more."""
        # A wins, then B wins
        matches = [
            make_match(score_a=1.0, score_b=0.0),
            make_match(score_a=0.0, score_b=1.0),
        ]
        result = compute_ratings(matches, init=RatingPair(100.0, 100.0))

        assert result.b > result.a
