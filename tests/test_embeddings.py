"""Tests for embeddings module.

Unit tests swap in fake backends, so they run without models or API keys.
Integration tests run the real backends: Qwen downloads its model from
HuggingFace on first use, and OpenAI makes paid API calls (marked `openai`).
"""

import numpy as np
import pytest

import embeddings
from embeddings import get_embeddings


@pytest.fixture(autouse=True)
def temporary_cache(monkeypatch, tmp_path):
    """Give each test an empty cache, separate from the real one."""
    monkeypatch.setattr(embeddings, "CACHE_DIR", tmp_path)


@pytest.fixture
def embedded_prompts(monkeypatch):
    """Replace both backends with fakes that record the prompts they embed.

    Fake vectors are [backend id, prompt length]: Qwen's id is 1, OpenAI's is 2.
    """
    embedded = {"qwen": [], "openai": []}

    def fake(backend, backend_id):
        def embed(texts):
            embedded[backend].extend(texts)
            vectors = [[backend_id, len(t)] for t in texts]
            return np.array(vectors, dtype=np.float32)  # (n_texts, 2)

        return embed

    monkeypatch.setattr(embeddings, "_embed_qwen_texts", fake("qwen", 1.0))
    monkeypatch.setattr(embeddings, "_embed_openai_texts", fake("openai", 2.0))
    return embedded


class TestGetEmbeddings:
    """Tests for caching and backend selection, using fake backends."""

    def test_should_embed_only_prompts_missing_from_cache(self, embedded_prompts):
        """Should embed each prompt once, reusing cached vectors after that."""
        get_embeddings(["a", "bb"])
        get_embeddings(["bb", "ccc", "a"])

        assert embedded_prompts["qwen"] == ["a", "bb", "ccc"]

    def test_should_embed_repeated_prompt_once(self, embedded_prompts):
        """Should embed a prompt once even when it repeats within one call."""
        get_embeddings(["a", "a", "bb"])

        assert embedded_prompts["qwen"] == ["a", "bb"]

    def test_should_return_vectors_in_prompt_order(self, embedded_prompts):
        """Should line up cached and newly embedded vectors with the input order."""
        get_embeddings(["a", "bb"])

        result = get_embeddings(["bb", "ccc", "a", "bb"])  # (4, 2)

        np.testing.assert_array_equal(
            result, [[1.0, 2.0], [1.0, 3.0], [1.0, 1.0], [1.0, 2.0]]
        )

    def test_should_default_to_qwen_when_openai_key_is_set(
        self, embedded_prompts, monkeypatch
    ):
        """Should use the local model unless OpenAI is selected explicitly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        get_embeddings(["a"])

        assert embedded_prompts == {"qwen": ["a"], "openai": []}

    def test_should_keep_each_backends_cache_separate(self, embedded_prompts):
        """Should never return one backend's cached vectors for the other."""
        qwen = get_embeddings(["a"])
        openai = get_embeddings(["a"], backend="openai")

        assert embedded_prompts == {"qwen": ["a"], "openai": ["a"]}
        assert not np.array_equal(qwen, openai)


@pytest.mark.integration
class TestQwenBackend:
    """Runs the local model, downloading it from HuggingFace on first use."""

    def test_should_return_one_vector_per_prompt(self):
        """Should return array of shape (n, 1024) for n prompts."""
        result = get_embeddings(["Hello world", "Test prompt"])

        assert result.shape == (2, 1024)

    def test_should_reject_prompt_over_token_limit(self):
        """Should raise rather than silently truncate an over-long prompt."""
        with pytest.raises(ValueError, match="over the model's limit"):
            get_embeddings(["hello " * 100_000])


@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIBackend:
    """Calls the paid OpenAI API, so needs OPENAI_API_KEY."""

    def test_should_return_one_vector_per_prompt(self):
        """Should return array of shape (n, 3072) for n prompts."""
        result = get_embeddings(["Hello world", "Test prompt"], backend="openai")

        assert result.shape == (2, 3072)
