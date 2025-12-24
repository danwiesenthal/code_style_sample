"""Shared helpers for tests."""

from typing import Any

import numpy as np

from match_types import MatchRow
from router import EagleRouter, build_train_index, compute_global_ratings


def make_match(
    row_id: str = "test",
    score_a: float = 1.0,
    score_b: float = 0.0,
) -> MatchRow:
    """Create a MatchRow with a placeholder prompt. Defaults to an A win."""
    return MatchRow(row_id=row_id, prompt="test", score_a=score_a, score_b=score_b)


def make_router(
    rows: list[MatchRow],
    embeddings: np.ndarray,
    **router_kwargs: Any,
) -> EagleRouter:
    """Build an EagleRouter from training rows and their embeddings."""
    global_ratings = compute_global_ratings(rows)
    train_index = build_train_index(rows, embeddings)
    return EagleRouter(global_ratings, train_index, **router_kwargs)
