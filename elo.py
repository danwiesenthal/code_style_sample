"""Elo rating system - pure math, no dependencies on dataset or embeddings.

Standard Elo formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    R_A_new = R_A + K * (S_A - E_A)

Where:
    E_A = expected points (probability A wins)
    S_A = points A actually earned (1=win, 0.5=tie, 0=loss)
    K = update sensitivity (higher = faster rating changes)
    400 = standard Elo scale factor

The paper calls S the "actual score". This repo says "points" because
"score" here only means a judge's score for a model's response.
"""

from typing import NamedTuple

from match_types import Outcome

# Standard Elo scaling constant
SCALE = 400


class RatingPair(NamedTuple):
    """Pair of ratings for players A and B."""

    a: float
    b: float


def _probability_a_beats_b(ratings: RatingPair) -> float:
    """Probability that player A beats player B given their ratings."""
    return 1.0 / (1.0 + 10.0 ** ((ratings.b - ratings.a) / SCALE))


class PointsPair(NamedTuple):
    """Points players A and B earn from one match."""

    a: float
    b: float


def _points_from_outcome(outcome: Outcome) -> PointsPair:
    """Convert outcome to the points each player earns.

    win_a -> (1.0, 0.0)
    win_b -> (0.0, 1.0)
    tie   -> (0.5, 0.5)
    """
    if outcome == Outcome.WIN_A:
        return PointsPair(1.0, 0.0)
    elif outcome == Outcome.WIN_B:
        return PointsPair(0.0, 1.0)
    return PointsPair(0.5, 0.5)


def update_ratings(
    ratings: RatingPair,
    outcome: Outcome,
    k: float,
) -> RatingPair:
    """Apply one Elo update step after a match.

    Args:
        ratings: Current ratings of players A and B
        outcome: Match result (WIN_A, WIN_B, or TIE)
        k: Elo K-factor (sensitivity, paper uses K=32)

    Returns:
        RatingPair with new ratings for A and B
    """
    e_a = _probability_a_beats_b(ratings)
    e_b = 1.0 - e_a

    points = _points_from_outcome(outcome)

    new_a = ratings.a + k * (points.a - e_a)
    new_b = ratings.b + k * (points.b - e_b)

    return RatingPair(new_a, new_b)
