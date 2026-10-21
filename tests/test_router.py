"""Tests for EAGLE router.

Pure unit tests using synthetic embeddings - no external dependencies.
"""

from typing import Any

import numpy as np

from elo import RatingPair
from match_types import Outcome
from parameters import DEFAULT_INIT_RATING
from router import EagleRouter, build_train_index, compute_global_ratings
from tests.helpers import make_match_row, router_from_match_rows

# Two opposite points in a 2-d embedding space, as far apart as cosine
# similarity allows
LEFT = np.array([-1.0, 0.0])  # (dim,)
RIGHT = np.array([1.0, 0.0])  # (dim,)


def _predict_after_training_on(outcome: Outcome, **router_kwargs: Any) -> Outcome:
    """Train on 10 match rows that all have outcome, then predict for a query.

    The embeddings are random, since every neighbor has the same outcome.
    """
    rng = np.random.default_rng(seed=0)
    train_match_rows = [make_match_row(outcome, row_id=str(i)) for i in range(10)]
    train_embeddings = rng.standard_normal((10, 2))  # (n_train, dim)
    router = router_from_match_rows(
        train_match_rows, train_embeddings, n_neighbors=3, **router_kwargs
    )

    query = rng.standard_normal(2)  # (dim,)
    return router.predict(query)


def _router_with_a_wins_left_and_b_wins_right(
    n_per_side: int, **router_kwargs: Any
) -> EagleRouter:
    """Train on n_per_side A wins at LEFT, then n_per_side B wins at RIGHT."""
    a_wins = [make_match_row(Outcome.WIN_A, row_id=f"a{i}") for i in range(n_per_side)]
    b_wins = [make_match_row(Outcome.WIN_B, row_id=f"b{i}") for i in range(n_per_side)]
    train_embeddings = np.vstack(
        [np.tile(LEFT, (n_per_side, 1)), np.tile(RIGHT, (n_per_side, 1))]
    )  # (2 * n_per_side, dim)
    return router_from_match_rows(a_wins + b_wins, train_embeddings, **router_kwargs)


class TestComputeGlobalRatings:
    """Tests for the global ratings over all training match rows."""

    def test_should_start_both_models_at_default_rating(self):
        """Should return the default starting rating when there's no training data."""
        assert compute_global_ratings([]) == RatingPair(
            DEFAULT_INIT_RATING, DEFAULT_INIT_RATING
        )

    def test_should_use_given_k(self):
        """Should move each rating by k/2 for one win from equal ratings."""
        ratings = compute_global_ratings([make_match_row(Outcome.WIN_A)], k=16)

        assert ratings == RatingPair(DEFAULT_INIT_RATING + 8, DEFAULT_INIT_RATING - 8)


class TestBuildTrainIndex:
    """Tests for indexing the training match rows."""

    def test_should_pair_each_match_row_with_its_embedding(self):
        """Should return the match row whose embedding is nearest the query."""
        match_rows = [make_match_row(row_id="left"), make_match_row(row_id="right")]
        train_embeddings = np.array([LEFT, RIGHT])  # (n_train, dim)

        index = build_train_index(match_rows, train_embeddings)
        neighbors = index.get_nearest_neighbors(RIGHT, n_neighbors=1)

        assert neighbors[0].row_id == "right"


class TestPredict:
    """Tests for predicting which model wins for a query."""

    def test_should_predict_a_when_trained_on_a_wins(self):
        """Should predict A when every training match row is an A win."""
        assert _predict_after_training_on(Outcome.WIN_A) == Outcome.WIN_A

    def test_should_predict_b_when_trained_on_b_wins(self):
        """Should predict B when every training match row is a B win."""
        assert _predict_after_training_on(Outcome.WIN_B) == Outcome.WIN_B

    def test_should_predict_tie_for_equal_ratings(self):
        """Should predict TIE when both models end up with equal ratings."""
        assert _predict_after_training_on(Outcome.TIE) == Outcome.TIE

    def test_should_ignore_neighbors_when_p_equals_1(self):
        """Should ignore training match rows similar to the query when P=1.0."""
        # P=1.0 means 100% global, 0% local, so the query's nearest neighbors
        # don't affect the prediction
        router = _router_with_a_wins_left_and_b_wins_right(
            n_per_side=5, n_neighbors=3, p=1.0
        )

        assert router.predict(LEFT) == router.predict(RIGHT)

    def test_should_use_local_context_when_p_equals_0(self):
        """Should base the prediction on nearest neighbors when P=0.0."""
        # A query among the A wins should predict A, and among the B wins, B
        router = _router_with_a_wins_left_and_b_wins_right(
            n_per_side=10, n_neighbors=5, p=0.0
        )

        assert router.predict(LEFT) == Outcome.WIN_A
        assert router.predict(RIGHT) == Outcome.WIN_B

    def test_should_weight_most_similar_neighbor_most(self):
        """Should let the most similar neighbor outweigh a less similar one."""
        # One A win right next to the query and one B win further away, with
        # P=0.0 so only the local rating counts: A should come out ahead
        train_match_rows = [
            make_match_row(Outcome.WIN_A, row_id="a"),
            make_match_row(Outcome.WIN_B, row_id="b"),
        ]
        train_embeddings = np.array([[1.0, 0.0], [0.6, 0.8]])  # (n_train, dim)
        router = router_from_match_rows(
            train_match_rows, train_embeddings, n_neighbors=2, p=0.0
        )

        query = np.array([1.0, 0.05])  # (dim,)
        assert router.predict(query) == Outcome.WIN_A

    def test_should_predict_tie_when_gap_is_within_tie_threshold(self):
        """Should predict TIE when the rating gap is smaller than tie_threshold."""
        prediction = _predict_after_training_on(Outcome.WIN_A, tie_threshold=1000.0)
        assert prediction == Outcome.TIE

    def test_should_predict_tie_when_gap_equals_tie_threshold(self):
        """Should predict TIE when the rating gap is exactly tie_threshold."""
        # With P=1.0 only the global rating counts. One A win from equal
        # ratings moves each rating by K/2 = 16, so the gap is exactly 32.
        router = router_from_match_rows(
            [make_match_row(Outcome.WIN_A)],
            np.array([RIGHT]),  # (n_train, dim)
            k=32,
            n_neighbors=1,
            p=1.0,
            tie_threshold=32.0,
        )

        assert router.predict(RIGHT) == Outcome.TIE

    def test_should_predict_winner_when_gap_exceeds_tie_threshold(self):
        """Should still predict the winner when the gap is bigger than tie_threshold."""
        prediction = _predict_after_training_on(Outcome.WIN_A, tie_threshold=1.0)
        assert prediction == Outcome.WIN_A
