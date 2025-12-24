"""Data access module for loading and sampling the SPROUT dataset."""

from typing import Literal, Sequence
import random

from datasets import Dataset, load_dataset

from match_types import MatchRow

ModelId = Literal["openai-gpt-4o", "openai-gpt-4o-mini"]
MODEL_A: ModelId = "openai-gpt-4o"
MODEL_B: ModelId = "openai-gpt-4o-mini"

# HuggingFace dataset with train/validation/test splits. Each row has a key, a
# prompt, and a column per model holding its response and judge score.
DATASET_ID = "CARROT-LLM-Routing/SPROUT"


def load_splits() -> tuple[list[MatchRow], list[MatchRow], list[MatchRow]]:
    """Load train, val, test splits from HuggingFace."""
    dataset = load_dataset(DATASET_ID)

    def to_match_rows(split_data: Dataset) -> list[MatchRow]:
        rows = []
        for item in split_data:
            row = MatchRow(
                row_id=str(item["key"]),
                prompt=item["prompt"],
                score_a=float(item[MODEL_A]["score"]),
                score_b=float(item[MODEL_B]["score"]),
            )
            rows.append(row)
        return rows

    train_rows = to_match_rows(dataset["train"])
    val_rows = to_match_rows(dataset["validation"])
    test_rows = to_match_rows(dataset["test"])

    return train_rows, val_rows, test_rows


def sample_rows(rows: Sequence[MatchRow], n: int, seed: int = 42) -> list[MatchRow]:
    """Deterministically sample n rows."""
    if n >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    return [rows[i] for i in sorted(indices)]


def get_sampled_splits(
    train_n: int = 2000,
    val_n: int = 200,
    test_n: int = 200,
    seed: int = 42,
) -> tuple[list[MatchRow], list[MatchRow], list[MatchRow]]:
    """Load splits and sample to specified sizes."""
    train_rows, val_rows, test_rows = load_splits()

    train_sample = sample_rows(train_rows, train_n, seed)
    val_sample = sample_rows(val_rows, val_n, seed)
    test_sample = sample_rows(test_rows, test_n, seed)

    return train_sample, val_sample, test_sample
