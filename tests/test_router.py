"""Tests for EAGLE router.

Pure unit tests using synthetic embeddings - no external dependencies.
"""

import numpy as np

from match_types import Outcome
from tests.helpers import make_match, make_router


class TestPredict:
    """Tests for predicting which model wins for a query."""

    def test_should_predict_a_when_trained_on_a_wins(self):
        """Should predict A when every training match is an A win."""
        rng = np.random.default_rng(seed=0)
        train_rows = [make_match(str(i), 1.0, 0.0) for i in range(10)]
        router = make_router(train_rows, rng.standard_normal((10, 8)), n_neighbors=3)

        query = rng.standard_normal(8)
        assert router.predict(query) == Outcome.WIN_A

    def test_should_predict_b_when_trained_on_b_wins(self):
        """Should predict B when every training match is a B win."""
        rng = np.random.default_rng(seed=0)
        train_rows = [make_match(str(i), 0.0, 1.0) for i in range(10)]
        router = make_router(train_rows, rng.standard_normal((10, 8)), n_neighbors=3)

        query = rng.standard_normal(8)
        assert router.predict(query) == Outcome.WIN_B

    def test_should_predict_tie_for_equal_ratings(self):
        """Should predict TIE when both models end up with equal ratings."""
        rng = np.random.default_rng(seed=0)
        train_rows = [make_match(str(i), 0.5, 0.5) for i in range(10)]
        router = make_router(train_rows, rng.standard_normal((10, 8)), n_neighbors=3)

        query = rng.standard_normal(8)
        assert router.predict(query) == Outcome.TIE

    def test_should_ignore_neighbors_when_p_equals_1(self):
        """Should ignore training examples similar to the query when P=1.0.

        P=1.0 means 100% global, 0% local - so the query's nearest neighbors
        don't affect the prediction.
        """
        # Global: 5 A wins, 5 B wins
        a_wins = [make_match(f"a{i}", 1.0, 0.0) for i in range(5)]
        b_wins = [make_match(f"b{i}", 0.0, 1.0) for i in range(5)]

        # Embeddings: A wins at [-1, ...], B wins at [1, ...]
        a_embeddings = np.tile([-1, 0, 0, 0, 0, 0, 0, 0], (5, 1)).astype(float)
        b_embeddings = np.tile([1, 0, 0, 0, 0, 0, 0, 0], (5, 1)).astype(float)

        router = make_router(
            a_wins + b_wins,
            np.vstack([a_embeddings, b_embeddings]),
            n_neighbors=3,
            p=1.0,
        )

        # Query near A wins vs query near B wins should give same result
        query_near_a = np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        query_near_b = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        assert router.predict(query_near_a) == router.predict(query_near_b)

    def test_should_use_local_context_when_p_equals_0(self):
        """Should base the prediction on nearest neighbors when P=0.0.

        Query near A-winning examples should predict A.
        Query near B-winning examples should predict B.
        """
        # All A wins in one region, all B wins in another
        a_wins = [make_match(f"a{i}", 1.0, 0.0) for i in range(10)]
        b_wins = [make_match(f"b{i}", 0.0, 1.0) for i in range(10)]

        # Embeddings: A wins at [-1, 0, 0, ...], B wins at [1, 0, 0, ...]
        a_embeddings = np.tile([-1, 0, 0, 0, 0, 0, 0, 0], (10, 1)).astype(float)
        b_embeddings = np.tile([1, 0, 0, 0, 0, 0, 0, 0], (10, 1)).astype(float)

        router = make_router(
            a_wins + b_wins,
            np.vstack([a_embeddings, b_embeddings]),
            n_neighbors=5,
            p=0.0,
        )

        query_near_a = np.array([-1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        query_near_b = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        assert router.predict(query_near_a) == Outcome.WIN_A
        assert router.predict(query_near_b) == Outcome.WIN_B

    def test_should_weight_most_similar_neighbor_most(self):
        """Should let the most similar neighbor outweigh a less similar one.

        One A win right next to the query and one B win further away, with
        P=0.0 so only the local rating counts: A should come out ahead.
        """
        train_rows = [make_match("a", 1.0, 0.0), make_match("b", 0.0, 1.0)]
        train_embeddings = np.array([[1.0, 0.0], [0.6, 0.8]])
        router = make_router(train_rows, train_embeddings, n_neighbors=2, p=0.0)

        query = np.array([1.0, 0.05])
        assert router.predict(query) == Outcome.WIN_A
