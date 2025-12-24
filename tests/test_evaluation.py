"""Tests for evaluation utilities.

Pure unit tests using synthetic embeddings - no external dependencies.
"""

import numpy as np

from evaluation import compute_accuracy, compute_baseline_accuracy
from match_types import Outcome
from tests.helpers import make_match, make_router


class TestComputeAccuracy:
    """Tests for router accuracy on labeled examples."""

    def test_should_return_fraction_of_correct_predictions(self):
        """Should return 2/3 when router correctly predicts 2 of 3 test examples."""
        # Train on all A wins -> router will always predict A
        train_rows = [make_match(str(i), 1.0, 0.0) for i in range(10)]
        router = make_router(train_rows, np.ones((10, 8)), n_neighbors=3)

        # Test set: 2 A wins, 1 B win -> router gets 2 right, 1 wrong
        test_rows = [
            make_match("t1", 1.0, 0.0),  # A wins - router predicts A ✓
            make_match("t2", 1.0, 0.0),  # A wins - router predicts A ✓
            make_match("t3", 0.0, 1.0),  # B wins - router predicts A ✗
        ]

        accuracy = compute_accuracy(router, test_rows, np.ones((3, 8)))
        assert accuracy == 2 / 3

    def test_should_return_0_for_empty_input(self):
        """Should return 0 when given an empty test set."""
        train_rows = [make_match(str(i), 1.0, 0.0) for i in range(5)]
        router = make_router(train_rows, np.ones((5, 8)), n_neighbors=3)

        accuracy = compute_accuracy(router, [], np.array([]))
        assert accuracy == 0.0


class TestComputeBaselineAccuracy:
    """Tests for the accuracy of always predicting one outcome."""

    def test_should_return_fraction_with_that_outcome(self):
        """Should return 2/3 when 2 of 3 examples have the predicted outcome."""
        rows = [
            make_match("t1", 0.5, 0.5),  # Tie
            make_match("t2", 0.5, 0.5),  # Tie
            make_match("t3", 1.0, 0.0),  # A wins
        ]

        assert compute_baseline_accuracy(rows, Outcome.TIE) == 2 / 3

    def test_should_return_0_for_empty_input(self):
        """Should return 0 when given an empty test set."""
        assert compute_baseline_accuracy([], Outcome.TIE) == 0.0
