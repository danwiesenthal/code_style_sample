"""Tests for similarity search.

Pure unit tests using synthetic embeddings - no external dependencies.
"""

import numpy as np
import pytest

from similarity import TrainIndex
from tests.helpers import make_match_row


class TestTrainIndex:
    """Tests for building the index and retrieving nearest neighbors."""

    def test_should_reject_mismatched_lengths(self):
        """Should raise ValueError if match rows and embeddings differ in length."""
        match_rows = [make_match_row(row_id="1"), make_match_row(row_id="2")]
        embeddings = np.array([[1.0, 0.0]])  # (1, dim): one embedding for two rows

        with pytest.raises(ValueError, match="must have same length"):
            TrainIndex(match_rows=match_rows, embeddings=embeddings)

    def test_should_return_n_neighbors(self):
        """Should return exactly n_neighbors neighbors."""
        match_rows = [make_match_row(row_id=str(i)) for i in range(10)]
        rng = np.random.default_rng(seed=0)
        embeddings = rng.standard_normal((10, 2))  # (n, dim)

        index = TrainIndex(match_rows=match_rows, embeddings=embeddings)
        neighbors = index.get_nearest_neighbors(embeddings[0], n_neighbors=3)

        assert len(neighbors) == 3

    def test_should_return_most_similar_first(self):
        """Should return neighbors ordered by descending similarity."""
        match_rows = [
            make_match_row(row_id="a"),
            make_match_row(row_id="b"),
            make_match_row(row_id="c"),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],  # a
                [0.5, 0.5],  # b
                [0.0, 1.0],  # c
            ]
        )  # (n, dim)

        index = TrainIndex(match_rows=match_rows, embeddings=embeddings)

        # Query is closest to 'a', then 'b', then 'c'
        query = np.array([0.9, 0.1])  # (dim,)
        neighbors = index.get_nearest_neighbors(query, n_neighbors=3)

        assert [match_row.row_id for match_row in neighbors] == ["a", "b", "c"]

    def test_should_rank_by_direction_not_length(self):
        """Should rank by cosine similarity, so an embedding's length doesn't count."""
        match_rows = [make_match_row(row_id="short"), make_match_row(row_id="long")]
        embeddings = np.array(
            [
                [0.1, 0.0],  # short, pointing the same way as the query
                [10.0, 10.0],  # long, at 45 degrees to the query
            ]
        )  # (n, dim)

        index = TrainIndex(match_rows=match_rows, embeddings=embeddings)
        neighbors = index.get_nearest_neighbors(np.array([1.0, 0.0]), n_neighbors=1)

        assert neighbors[0].row_id == "short"
