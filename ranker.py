"""Ranker module for computing Elo ratings from match row sequences."""

from collections.abc import Sequence

from elo import RatingPair, update_ratings
from match_types import MatchRow
from parameters import DEFAULT_INIT_RATING, DEFAULT_K


def compute_ratings(
    match_rows: Sequence[MatchRow],
    k: float = DEFAULT_K,
    init: RatingPair = RatingPair(DEFAULT_INIT_RATING, DEFAULT_INIT_RATING),
) -> RatingPair:
    """Compute Elo ratings by processing match rows sequentially.

    Used for both the global rating (all training match rows) and the local
    rating (a query's nearest neighbors). Order matters: each update shrinks
    the effect of earlier match rows, so later ones count more.

    Args:
        match_rows: Match rows to process, in order
        k: Elo K-factor (paper uses K=32)
        init: Starting ratings for models A and B

    Returns:
        Final RatingPair after processing all match rows
    """
    ratings = init

    for match_row in match_rows:
        ratings = update_ratings(ratings, match_row.outcome, k)

    return ratings
