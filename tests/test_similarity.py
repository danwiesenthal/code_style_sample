"""Tests for similarity search.

Pure unit tests using synthetic embeddings - no external dependencies.
"""

import numpy as np
import pytest

from similarity import TrainIndex
from tests.helpers import make_match


class TestTrainIndex:
    """Tests for nearest neighbor retrieval."""

    def test_should_reject_mismatched_lengths(self):
        """Should raise ValueError if rows and embeddings have different lengths."""
        rows = [make_match("1"), make_match("2")]
        embeddings = np.array([[1.0, 0.0]])  # Only 1 embedding for 2 rows

        with pytest.raises(ValueError):
            TrainIndex(rows=rows, embeddings=embeddings)

    def test_should_return_n_neighbors(self):
        """Should return exactly n_neighbors neighbors."""
        rows = [make_match(str(i)) for i in range(10)]
        rng = np.random.default_rng(seed=0)
        embeddings = rng.standard_normal((10, 8))

        index = TrainIndex(rows=rows, embeddings=embeddings)
        neighbors = index.get_nearest_neighbors(embeddings[0], n_neighbors=3)

        assert len(neighbors) == 3

    def test_should_return_most_similar_first(self):
        """Should return neighbors ordered by descending similarity."""
        rows = [make_match("a"), make_match("b"), make_match("c")]
        # Embeddings: a=[1,0], b=[0.5,0.5], c=[0,1]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        )

        index = TrainIndex(rows=rows, embeddings=embeddings)

        # Query is close to 'a' - should match 'a' best, then 'b', then 'c'
        query = np.array([0.9, 0.1])
        neighbors = index.get_nearest_neighbors(query, n_neighbors=3)

        assert neighbors[0].row_id == "a"  # Closest
        assert neighbors[1].row_id == "b"  # Middle
        assert neighbors[2].row_id == "c"  # Furthest
