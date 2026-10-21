"""EAGLE router - combines global and local Elo ratings for prediction."""

import numpy as np

from elo import RatingPair
from match_types import MatchRow, Outcome
from parameters import DEFAULT_K, DEFAULT_N_NEIGHBORS, DEFAULT_P, DEFAULT_TIE_THRESHOLD
from ranker import compute_ratings
from similarity import TrainIndex


def build_train_index(
    train_match_rows: list[MatchRow],
    train_embeddings: np.ndarray,
) -> TrainIndex:
    """Build the similarity index over the training match rows.

    Args:
        train_match_rows: Training match rows
        train_embeddings: Prompt embedding for each training match row,
            shape (n_train, dim)

    Returns:
        TrainIndex for finding a query's nearest training match rows
    """
    return TrainIndex(match_rows=train_match_rows, embeddings=train_embeddings)


def compute_global_ratings(
    train_match_rows: list[MatchRow],
    k: float = DEFAULT_K,
) -> RatingPair:
    """Compute global Elo ratings from all training match rows.

    Args:
        train_match_rows: Training match rows, processed in order
        k: Elo K-factor

    Returns:
        Global ratings for models A and B
    """
    return compute_ratings(train_match_rows, k=k)


class EagleRouter:
    """EAGLE router combining global and local Elo ratings to predict outcomes.

    Takes pre-computed global ratings and a train index, built with
    compute_global_ratings() and build_train_index(). They're explicit inputs,
    rather than built inside __init__, so the caller controls when that work
    happens: main.py builds them once and shares them across every router it
    tries, instead of rebuilding them in each constructor.

    Note: tie_threshold is an extension, not part of the original paper.
    The paper simply picks the higher-rated model (no ties); the default
    tie_threshold=0.0 matches that behavior.

    Args:
        global_ratings: Elo ratings from every training match row
        train_index: Training match rows indexed for similarity search
        k: Elo K-factor for the local ratings
        n_neighbors: Number of nearest training prompts used for the local
            rating (the paper's N)
        p: Weight on the global rating in the combined rating; the local rating
            gets 1 - p
        tie_threshold: Largest gap between the two combined ratings that's
            predicted as a tie
    """

    def __init__(
        self,
        global_ratings: RatingPair,
        train_index: TrainIndex,
        k: float = DEFAULT_K,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        p: float = DEFAULT_P,
        tie_threshold: float = DEFAULT_TIE_THRESHOLD,
    ) -> None:
        self.global_ratings = global_ratings
        self.train_index = train_index
        self.k = k
        self.n_neighbors = n_neighbors
        self.p = p
        self.tie_threshold = tie_threshold

    def predict(self, query_embedding: np.ndarray) -> Outcome:
        """Predict which model wins for a query.

        Args:
            query_embedding: Embedding of the query prompt, shape (dim,)

        Returns:
            Predicted outcome: WIN_A, WIN_B, or TIE
        """
        ratings = self._combined_ratings(query_embedding)
        diff = ratings.a - ratings.b

        if abs(diff) <= self.tie_threshold:
            return Outcome.TIE
        elif diff > 0:
            return Outcome.WIN_A
        return Outcome.WIN_B

    def _combined_ratings(self, query_embedding: np.ndarray) -> RatingPair:
        """Compute combined global + local ratings for a query."""
        # Find similar prompts from training set
        neighbors = self.train_index.get_nearest_neighbors(
            query_embedding, self.n_neighbors
        )

        # Compute local ratings from neighbor match rows, starting from global.
        # Elo weights later match rows more, so apply the most similar one last.
        local_ratings = compute_ratings(
            neighbors[::-1], k=self.k, init=self.global_ratings
        )

        # Weighted combination: P * global + (1-P) * local
        combined_a = self.p * self.global_ratings.a + (1 - self.p) * local_ratings.a
        combined_b = self.p * self.global_ratings.b + (1 - self.p) * local_ratings.b

        return RatingPair(combined_a, combined_b)
