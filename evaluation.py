"""Evaluation utilities for measuring router accuracy."""

import numpy as np

from match_types import MatchRow, Outcome
from router import EagleRouter


def compute_accuracy(
    router: EagleRouter,
    match_rows: list[MatchRow],
    embeddings: np.ndarray,
) -> float:
    """Compute the fraction of match rows whose outcome the router predicts.

    Args:
        router: Router to evaluate
        match_rows: Match rows with ground truth scores (score_a, score_b)
        embeddings: Prompt embedding for each match row, shape (n, dim)

    Returns:
        Accuracy between 0.0 and 1.0
    """
    if not match_rows:
        raise ValueError("Need at least one match row to compute accuracy")

    correct = sum(
        router.predict(embedding) == match_row.outcome
        for match_row, embedding in zip(match_rows, embeddings, strict=True)
    )
    return correct / len(match_rows)


def compute_baseline_accuracy(match_rows: list[MatchRow], outcome: Outcome) -> float:
    """Compute the accuracy of a baseline that always predicts one outcome.

    Args:
        match_rows: Match rows with ground truth scores (score_a, score_b)
        outcome: The outcome the baseline always predicts

    Returns:
        Accuracy between 0.0 and 1.0
    """
    if not match_rows:
        raise ValueError("Need at least one match row to compute accuracy")

    correct = sum(match_row.outcome == outcome for match_row in match_rows)
    return correct / len(match_rows)


def find_best_setting(accuracies: np.ndarray) -> tuple[int, int]:
    """Find the setting with the highest accuracy in a grid search.

    When several settings tie for best, the first one in search order (row by
    row) wins.

    Args:
        accuracies: Accuracy of each setting, shape (n_p_values, n_tie_thresholds)

    Returns:
        (row, column) index of the best setting
    """
    row, column = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    return int(row), int(column)
