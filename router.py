"""EAGLE router - combines global and local Elo ratings for prediction."""

import numpy as np

from match_types import MatchRow, Outcome
from elo import RatingPair
from ranker import DEFAULT_INIT_RATING, DEFAULT_K, compute_ratings
from similarity import TrainIndex

# Paper hyperparameters (Appendix A)
DEFAULT_P = 0.5
DEFAULT_N = 20


def build_train_index(
    train_rows: list[MatchRow],
    train_embeddings: np.ndarray,
) -> TrainIndex:
    """Build similarity index from training data."""
    return TrainIndex(rows=train_rows, embeddings=train_embeddings)


def compute_global_ratings(
    train_rows: list[MatchRow],
    k: float = DEFAULT_K,
) -> RatingPair:
    """Compute global Elo ratings from all training matches."""
    init = RatingPair(DEFAULT_INIT_RATING, DEFAULT_INIT_RATING)
    return compute_ratings(train_rows, k=k, init=init)


class EagleRouter:
    """EAGLE router combining global and local Elo for model selection.

    Takes pre-computed global ratings and a train index, built with
    compute_global_ratings() and build_train_index() (in either order).
    Building the ratings and index is the expensive part, so it happens in
    explicit calls rather than implicitly inside __init__.

    Note: tie_threshold is an extension, not part of the original paper.
    The paper simply picks the higher-rated model (no ties); the default
    tie_threshold=0.0 matches that behavior.
    """

    def __init__(
        self,
        global_ratings: RatingPair,
        train_index: TrainIndex,
        k: float = DEFAULT_K,
        n_neighbors: int = DEFAULT_N,
        p: float = DEFAULT_P,
        tie_threshold: float = 0.0,
    ) -> None:
        self.global_ratings = global_ratings
        self.train_index = train_index
        self.k = k
        self.n_neighbors = n_neighbors
        self.p = p
        self.tie_threshold = tie_threshold

    def predict(self, query_embedding: np.ndarray) -> Outcome:
        """Predict which model wins for a query."""
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

        # Compute local ratings from neighbor matches, starting from global.
        # Elo weights later matches more, so apply the most similar one last.
        local_ratings = compute_ratings(
            neighbors[::-1], k=self.k, init=self.global_ratings
        )

        # Weighted combination: P * global + (1-P) * local
        combined_a = (
            self.p * self.global_ratings.a + (1 - self.p) * local_ratings.a
        )
        combined_b = (
            self.p * self.global_ratings.b + (1 - self.p) * local_ratings.b
        )

        return RatingPair(combined_a, combined_b)
