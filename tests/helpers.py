"""Shared helpers for tests."""

from typing import Any

import numpy as np

from match_types import MatchRow, Outcome
from parameters import DEFAULT_K
from router import EagleRouter, build_train_index, compute_global_ratings

# Judge scores (score_a, score_b) that give each outcome
_SCORES_FOR_OUTCOME = {
    Outcome.WIN_A: (1.0, 0.0),
    Outcome.WIN_B: (0.0, 1.0),
    Outcome.TIE: (0.5, 0.5),
}


def make_match_row(
    outcome: Outcome = Outcome.WIN_A,
    row_id: str = "test",
) -> MatchRow:
    """Create a match row with a placeholder prompt and scores giving outcome."""
    score_a, score_b = _SCORES_FOR_OUTCOME[outcome]
    return MatchRow(row_id=row_id, prompt="test", score_a=score_a, score_b=score_b)


def router_from_match_rows(
    match_rows: list[MatchRow],
    embeddings: np.ndarray,
    k: float = DEFAULT_K,
    **router_kwargs: Any,
) -> EagleRouter:
    """Build an EagleRouter from training match rows and their embeddings.

    Uses the same k for the global and local ratings, as main.py does.

    Args:
        match_rows: Training match rows
        embeddings: Prompt embedding for each match row, shape (n, dim)
        k: Elo K-factor for both the global and local ratings
        **router_kwargs: Other EagleRouter settings (n_neighbors, p, tie_threshold)

    Returns:
        Router over the given training match rows
    """
    global_ratings = compute_global_ratings(match_rows, k=k)
    train_index = build_train_index(match_rows, embeddings)
    return EagleRouter(global_ratings, train_index, k=k, **router_kwargs)
