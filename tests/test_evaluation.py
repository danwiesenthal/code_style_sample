"""Tests for evaluation utilities.

Pure unit tests using synthetic embeddings - no external dependencies.
"""

import numpy as np
import pytest

from evaluation import compute_accuracy, compute_baseline_accuracy, find_best_setting
from match_types import Outcome
from tests.helpers import make_match_row, router_from_match_rows


class TestComputeAccuracy:
    """Tests for router accuracy on labeled match rows."""

    def test_should_return_fraction_of_correct_predictions(self):
        """Should return 2/3 when router correctly predicts 2 of 3 test examples."""
        # Train on all A wins -> router will always predict A
        train_match_rows = [
            make_match_row(Outcome.WIN_A, row_id=str(i)) for i in range(10)
        ]
        train_embeddings = np.ones((10, 2))  # (n_train, dim)
        router = router_from_match_rows(
            train_match_rows, train_embeddings, n_neighbors=3
        )

        # Test set: 2 A wins, 1 B win -> router gets 2 right, 1 wrong
        test_match_rows = [
            make_match_row(Outcome.WIN_A, row_id="t1"),  # Router predicts A: right
            make_match_row(Outcome.WIN_A, row_id="t2"),  # Router predicts A: right
            make_match_row(Outcome.WIN_B, row_id="t3"),  # Router predicts A: wrong
        ]
        test_embeddings = np.ones((3, 2))  # (n_test, dim)

        accuracy = compute_accuracy(router, test_match_rows, test_embeddings)
        assert accuracy == 2 / 3

    def test_should_raise_for_empty_input(self):
        """Should raise ValueError when given no match rows."""
        router = router_from_match_rows(
            [make_match_row()], np.ones((1, 2)), n_neighbors=1
        )

        with pytest.raises(ValueError, match="at least one match row"):
            compute_accuracy(router, [], np.empty((0, 2)))


class TestComputeBaselineAccuracy:
    """Tests for the accuracy of always predicting one outcome."""

    def test_should_return_fraction_with_that_outcome(self):
        """Should return 2/3 when 2 of 3 match rows have the predicted outcome."""
        match_rows = [
            make_match_row(Outcome.TIE, row_id="t1"),
            make_match_row(Outcome.TIE, row_id="t2"),
            make_match_row(Outcome.WIN_A, row_id="t3"),
        ]

        assert compute_baseline_accuracy(match_rows, Outcome.TIE) == 2 / 3

    def test_should_raise_for_empty_input(self):
        """Should raise ValueError when given no match rows."""
        with pytest.raises(ValueError, match="at least one match row"):
            compute_baseline_accuracy([], Outcome.TIE)


class TestFindBestSetting:
    """Tests for picking the best setting from a grid of accuracies."""

    def test_should_return_row_and_column_of_highest_accuracy(self):
        """Should return the (row, column) index of the highest accuracy."""
        accuracies = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.9, 0.5],
            ]
        )  # (n_p_values, n_tie_thresholds)

        assert find_best_setting(accuracies) == (1, 1)

    def test_should_pick_first_in_search_order_when_settings_tie(self):
        """Should pick the earliest of several best settings, going row by row."""
        accuracies = np.array(
            [
                [0.1, 0.7],
                [0.7, 0.7],
            ]
        )  # (n_p_values, n_tie_thresholds)

        assert find_best_setting(accuracies) == (0, 1)
