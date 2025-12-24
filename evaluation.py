"""Evaluation utilities for measuring router accuracy."""

import numpy as np

from match_types import MatchRow, Outcome
from router import EagleRouter


def compute_accuracy(
    router: EagleRouter,
    rows: list[MatchRow],
    embeddings: np.ndarray,
) -> float:
    """Fraction of examples where router prediction matches actual outcome.

    Args:
        router: Trained router to evaluate
        rows: Examples with ground truth scores (score_a, score_b)
        embeddings: Prompt embeddings for each row, shape (n, dim)

    Returns:
        Accuracy between 0.0 and 1.0
    """
    if not rows:
        return 0.0

    correct = 0
    for row, embedding in zip(rows, embeddings, strict=True):
        predicted = router.predict(embedding)
        if predicted == row.outcome:
            correct += 1

    return correct / len(rows)


def compute_baseline_accuracy(rows: list[MatchRow], outcome: Outcome) -> float:
    """Accuracy of a baseline that always predicts the same outcome.

    Args:
        rows: Examples with ground truth scores (score_a, score_b)
        outcome: The outcome the baseline always predicts

    Returns:
        Accuracy between 0.0 and 1.0
    """
    if not rows:
        return 0.0

    return sum(row.outcome == outcome for row in rows) / len(rows)
