"""Tests for data_access module.

Tests that load the dataset hit the HuggingFace API, so they're marked as
integration tests. HuggingFace caches locally after first download
(~/.cache/huggingface/datasets).
"""

import pytest

import data_access
from data_access import (
    MODEL_A,
    MODEL_B,
    _to_match_rows,
    get_sampled_splits,
    load_splits,
    sample_match_rows,
)
from match_types import MatchRow
from tests.helpers import make_match_row


@pytest.mark.integration
class TestLoadSplits:
    """Tests for loading the SPROUT dataset from HuggingFace."""

    def test_should_load_all_splits_with_correct_sizes(self):
        """Should load the exact train/val/test split sizes."""
        train, val, test = load_splits()
        assert len(train) == 30968
        assert len(val) == 6636
        assert len(test) == 6637

    def test_should_populate_all_match_row_fields(self):
        """Should fill in every field of a loaded match row."""
        train, _, _ = load_splits()
        match_row = train[0]
        assert isinstance(match_row.row_id, str)
        assert match_row.row_id
        assert isinstance(match_row.prompt, str)
        assert match_row.prompt
        assert isinstance(match_row.score_a, float)
        assert isinstance(match_row.score_b, float)


class TestToMatchRows:
    """Tests for converting dataset rows to match rows."""

    def test_should_take_each_score_from_its_model(self):
        """Should read score_a from model A's column and score_b from model B's."""
        split_data = [
            {
                "key": "k1",
                "prompt": "Hello",
                MODEL_A: {"score": 0.9},
                MODEL_B: {"score": 0.2},
            }
        ]

        assert _to_match_rows(split_data) == [
            MatchRow(row_id="k1", prompt="Hello", score_a=0.9, score_b=0.2)
        ]


class TestSampleMatchRows:
    """Tests for deterministic sampling."""

    def test_should_return_same_match_rows_for_same_seed(self):
        """Should pick the same match rows every time for a given seed."""
        match_rows = [make_match_row(row_id=str(i)) for i in range(100)]

        first = sample_match_rows(match_rows, 10, seed=1)
        second = sample_match_rows(match_rows, 10, seed=1)
        assert first == second

    def test_should_keep_input_order(self):
        """Should return sampled match rows in the order they appear in the input."""
        match_rows = [make_match_row(row_id=str(i)) for i in range(100)]

        sampled = sample_match_rows(match_rows, 10)
        sampled_ids = [int(match_row.row_id) for match_row in sampled]
        assert sampled_ids == sorted(sampled_ids)

    def test_should_return_all_match_rows_when_n_exceeds_length(self):
        """Should return every match row when asked for more than exist."""
        match_rows = [make_match_row(row_id=str(i)) for i in range(5)]

        assert sample_match_rows(match_rows, 10) == match_rows


class TestGetSampledSplits:
    """Tests for sampling each split, with the dataset replaced by fake splits."""

    @pytest.fixture(autouse=True)
    def fake_splits(self, monkeypatch):
        """Replace load_splits with 10 match rows per split, ids named by split."""
        splits = tuple(
            [make_match_row(row_id=f"{split}-{i}") for i in range(10)]
            for split in ("train", "val", "test")
        )
        monkeypatch.setattr(data_access, "load_splits", lambda: splits)

    def test_should_sample_each_split_to_its_size(self):
        """Should return train_n, val_n and test_n match rows."""
        train, val, test = get_sampled_splits(train_n=5, val_n=3, test_n=2)

        assert (len(train), len(val), len(test)) == (5, 3, 2)

    def test_should_sample_each_split_from_its_own_match_rows(self):
        """Should never mix match rows from one split into another."""
        train, val, test = get_sampled_splits(train_n=5, val_n=5, test_n=5)

        assert all(match_row.row_id.startswith("train-") for match_row in train)
        assert all(match_row.row_id.startswith("val-") for match_row in val)
        assert all(match_row.row_id.startswith("test-") for match_row in test)
