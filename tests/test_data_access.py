"""Tests for data_access module.

Tests that load the dataset hit the HuggingFace API, so they're marked as
integration tests. HuggingFace caches locally after first download
(~/.cache/huggingface/datasets).
"""

import pytest

from data_access import load_splits, sample_rows
from tests.helpers import make_match


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
        """Should have row_id, prompt, score_a, score_b on each row."""
        train, _, _ = load_splits()
        row = train[0]
        assert isinstance(row.row_id, str) and row.row_id
        assert isinstance(row.prompt, str) and row.prompt
        assert isinstance(row.score_a, float)
        assert isinstance(row.score_b, float)


class TestSampleRows:
    """Tests for deterministic sampling."""

    def test_should_return_same_rows_for_same_seed(self):
        """Should pick the same rows every time for a given seed."""
        rows = [make_match(str(i)) for i in range(100)]

        assert sample_rows(rows, 10, seed=1) == sample_rows(rows, 10, seed=1)

    def test_should_keep_input_order(self):
        """Should return sampled rows in the order they appear in the input."""
        rows = [make_match(str(i)) for i in range(100)]

        sampled_ids = [int(row.row_id) for row in sample_rows(rows, 10)]
        assert sampled_ids == sorted(sampled_ids)

    def test_should_return_all_rows_when_n_exceeds_length(self):
        """Should return every row when asked for more rows than exist."""
        rows = [make_match(str(i)) for i in range(5)]

        assert sample_rows(rows, 10) == rows
