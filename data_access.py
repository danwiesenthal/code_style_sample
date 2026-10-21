"""Data access module for loading and sampling the SPROUT dataset."""

import random
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from datasets import load_dataset

from match_types import MatchRow
from parameters import DEFAULT_SEED, DEFAULT_TEST_N, DEFAULT_TRAIN_N, DEFAULT_VAL_N

ModelId = Literal["openai-gpt-4o", "openai-gpt-4o-mini"]
MODEL_A: ModelId = "openai-gpt-4o"
MODEL_B: ModelId = "openai-gpt-4o-mini"

# HuggingFace dataset with train/validation/test splits. Each row has a key, a
# prompt, and a column per model holding its response and judge score.
DATASET_ID = "CARROT-LLM-Routing/SPROUT"


def _to_match_rows(split_data: Iterable[dict[str, Any]]) -> list[MatchRow]:
    """Convert one split's dataset rows to match rows.

    Model A's judge score becomes score_a and model B's becomes score_b, which
    is what decides which model counts as A throughout the repo.
    """
    return [
        MatchRow(
            row_id=str(item["key"]),
            prompt=item["prompt"],
            score_a=float(item[MODEL_A]["score"]),
            score_b=float(item[MODEL_B]["score"]),
        )
        for item in split_data
    ]


def load_splits() -> tuple[list[MatchRow], list[MatchRow], list[MatchRow]]:
    """Load the train, validation and test splits from HuggingFace.

    Returns:
        Train, validation and test match rows, in dataset order
    """
    dataset = load_dataset(DATASET_ID)

    train_match_rows = _to_match_rows(dataset["train"])
    val_match_rows = _to_match_rows(dataset["validation"])
    test_match_rows = _to_match_rows(dataset["test"])

    return train_match_rows, val_match_rows, test_match_rows


def sample_match_rows(
    match_rows: Sequence[MatchRow],
    n: int,
    seed: int = DEFAULT_SEED,
) -> list[MatchRow]:
    """Deterministically sample n match rows, keeping their original order.

    Args:
        match_rows: Match rows to sample from
        n: Number of match rows to sample; all of them if n >= len(match_rows)
        seed: Random seed, so the same inputs always give the same sample

    Returns:
        The sampled match rows, in the order they appear in match_rows
    """
    if n >= len(match_rows):
        return list(match_rows)

    rng = random.Random(seed)
    indices = rng.sample(range(len(match_rows)), n)
    return [match_rows[i] for i in sorted(indices)]


def get_sampled_splits(
    train_n: int = DEFAULT_TRAIN_N,
    val_n: int = DEFAULT_VAL_N,
    test_n: int = DEFAULT_TEST_N,
    seed: int = DEFAULT_SEED,
) -> tuple[list[MatchRow], list[MatchRow], list[MatchRow]]:
    """Load the splits and sample each one to the requested size.

    Args:
        train_n: Number of training match rows
        val_n: Number of validation match rows
        test_n: Number of test match rows
        seed: Random seed for sampling

    Returns:
        Sampled train, validation and test match rows
    """
    train_match_rows, val_match_rows, test_match_rows = load_splits()

    train_sample = sample_match_rows(train_match_rows, train_n, seed)
    val_sample = sample_match_rows(val_match_rows, val_n, seed)
    test_sample = sample_match_rows(test_match_rows, test_n, seed)

    return train_sample, val_sample, test_sample
