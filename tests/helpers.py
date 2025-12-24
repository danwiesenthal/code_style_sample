"""Shared helpers for tests."""

from match_types import MatchRow


def make_match(
    row_id: str = "test",
    score_a: float = 1.0,
    score_b: float = 0.0,
) -> MatchRow:
    """Create a MatchRow with a placeholder prompt. Defaults to an A win."""
    return MatchRow(row_id=row_id, prompt="test", score_a=score_a, score_b=score_b)
