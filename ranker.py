"""Ranker module for computing Elo ratings from match sequences."""

from collections.abc import Sequence

from elo import RatingPair, update_ratings
from match_types import MatchRow

# K-factor from paper (Appendix A)
DEFAULT_K = 32

# Conventional Elo starting rating, shared by both models. Only the gap between
# the two ratings affects predictions, so this value doesn't (unlike K).
DEFAULT_INIT_RATING = 1500.0


def compute_ratings(
    matches: Sequence[MatchRow],
    k: float = DEFAULT_K,
    init: RatingPair = RatingPair(DEFAULT_INIT_RATING, DEFAULT_INIT_RATING),
) -> RatingPair:
    """Compute Elo ratings by processing matches sequentially.

    Used for both global (all training rows) and local (neighbor subset) ranking.
    Order matters: each update shrinks the effect of earlier matches, so later
    matches count more.

    Args:
        matches: Sequence of matches to process in order
        k: Elo K-factor (paper uses K=32)
        init: Starting ratings for models A and B

    Returns:
        Final RatingPair after processing all matches
    """
    ratings = init

    for match in matches:
        ratings = update_ratings(ratings, match.outcome, k)

    return ratings
