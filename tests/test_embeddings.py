"""Tests for embeddings module.

Integration tests - these call the OpenAI API, so they need OPENAI_API_KEY.
Prompts containing a uuid are never cached, so those tests always hit the API.
"""

import uuid

import numpy as np
import pytest

from embeddings import _load_cache, _prompt_hash, get_embeddings

pytestmark = pytest.mark.integration

# text-embedding-3-large returns 3072-dimensional vectors
EXPECTED_EMBEDDING_DIM = 3072


class TestGetEmbeddings:
    """Tests for OpenAI embedding retrieval with caching."""

    def test_should_return_embeddings_with_correct_shape(self):
        """Should return array of shape (n, 3072) for n prompts."""
        prompts = ["Hello world", "Test prompt"]
        embeddings = get_embeddings(prompts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == EXPECTED_EMBEDDING_DIM

    def test_should_call_api_for_fresh_prompt(self):
        """Should successfully call OpenAI API for a never-seen-before prompt.

        Uses uuid to ensure the prompt has never been cached.
        """
        fresh_prompt = f"Random test {uuid.uuid4()}"
        embeddings = get_embeddings([fresh_prompt])

        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == EXPECTED_EMBEDDING_DIM

    def test_should_cache_embeddings_and_return_them_on_second_call(self):
        """Should store fetched embeddings in cache and return them on repeat.

        First call hits the API and writes the cache; second call reads it back.
        """
        fresh_prompt = f"Cache test {uuid.uuid4()}"
        prompt_hash = _prompt_hash(fresh_prompt)

        # First call - hits API and writes cache
        first = get_embeddings([fresh_prompt])

        # Cache should hold exactly what was returned
        cache = _load_cache()
        assert prompt_hash in cache
        cached = np.array(cache[prompt_hash], dtype=np.float32)
        np.testing.assert_array_equal(cached, first[0])

        # Second call - should return the cached embedding unchanged
        second = get_embeddings([fresh_prompt])
        np.testing.assert_array_equal(second, first)
