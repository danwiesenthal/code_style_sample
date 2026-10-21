"""Tests for match types.

Pure unit tests - no external dependencies.
"""

import dataclasses

import pytest

from match_types import MatchRow, Outcome


def _match_row_with_scores(score_a: float, score_b: float) -> MatchRow:
    """Create a match row with a placeholder prompt and the given judge scores."""
    return MatchRow(row_id="test", prompt="test", score_a=score_a, score_b=score_b)


class TestMatchRow:
    """Tests for deciding a match row's outcome from the two judge scores."""

    def test_should_be_win_a_when_a_scores_higher(self):
        """Should give WIN_A when model A's response scores higher."""
        assert _match_row_with_scores(0.8, 0.3).outcome == Outcome.WIN_A

    def test_should_be_win_b_when_b_scores_higher(self):
        """Should give WIN_B when model B's response scores higher."""
        assert _match_row_with_scores(0.3, 0.8).outcome == Outcome.WIN_B

    def test_should_be_tie_when_scores_are_equal(self):
        """Should give TIE when both responses get the same score."""
        assert _match_row_with_scores(0.5, 0.5).outcome == Outcome.TIE

    def test_should_not_treat_close_scores_as_tie(self):
        """Should count any difference in scores as a win, however small."""
        assert _match_row_with_scores(0.51, 0.5).outcome == Outcome.WIN_A

    def test_should_be_immutable(self):
        """Should raise when a field is changed after the match row is created."""
        match_row = _match_row_with_scores(1.0, 0.0)

        # ty flags this assignment too; the test checks it also fails at runtime
        with pytest.raises(dataclasses.FrozenInstanceError):
            match_row.score_a = 0.0  # ty: ignore[invalid-assignment]
