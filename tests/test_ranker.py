"""Tests for ranker module.

Pure unit tests - no external dependencies.
"""

from elo import RatingPair
from match_types import Outcome
from ranker import compute_ratings
from tests.helpers import make_match_row


class TestComputeRatings:
    """Tests for computing ratings over a sequence of match rows."""

    def test_should_return_init_ratings_for_no_match_rows(self):
        """Should return initial ratings unchanged if there are no match rows."""
        init = RatingPair(100.0, 100.0)
        result = compute_ratings([], init=init)
        assert result == init

    def test_should_raise_winner_and_lower_loser(self):
        """Should increase the winner's rating and decrease the loser's."""
        match_rows = [make_match_row(Outcome.WIN_A)]
        result = compute_ratings(match_rows, init=RatingPair(100.0, 100.0))

        assert result.a > 100.0
        assert result.b < 100.0

    def test_should_accumulate_across_multiple_match_rows(self):
        """Should process all match rows, accumulating rating changes."""
        match_rows = [make_match_row(Outcome.WIN_A), make_match_row(Outcome.WIN_A)]
        result = compute_ratings(match_rows, init=RatingPair(100.0, 100.0))

        # After two wins, A should be even higher
        single_win = compute_ratings(match_rows[:1], init=RatingPair(100.0, 100.0))
        assert result.a > single_win.a

    def test_should_weight_later_match_rows_more(self):
        """Should let the later of two opposite results count more."""
        match_rows = [make_match_row(Outcome.WIN_A), make_match_row(Outcome.WIN_B)]
        result = compute_ratings(match_rows, init=RatingPair(100.0, 100.0))

        assert result.b > result.a
