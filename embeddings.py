"""Embeddings module for OpenAI text embeddings with disk caching."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-large"
CACHE_DIR = Path(__file__).parent / "cache"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Get or create singleton OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _client = OpenAI(api_key=api_key)
    return _client


def _prompt_hash(prompt: str) -> str:
    """Hash prompt content for cache key."""
    return hashlib.md5(prompt.encode()).hexdigest()


def _cache_path() -> Path:
    """Get cache file path, creating directory if needed."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / "embeddings_cache.json"


def _load_cache() -> dict[str, list[float]]:
    """Load embedding cache from disk. Keyed by prompt hash."""
    path = _cache_path()
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    """Save embedding cache to disk."""
    with open(_cache_path(), "w") as f:
        json.dump(cache, f)


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Call OpenAI API to embed texts. No caching.

    Returns array of shape (n, embedding_dim).
    Uses float32 to reduce memory footprint (float64 unnecessary for similarity).
    """
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


def get_embeddings(prompts: list[str]) -> np.ndarray:
    """Get embeddings for prompts, cached by prompt content hash.

    Cache is checked first; only missing prompts are fetched from API.
    Returns array of shape (n, embedding_dim) in same order as input prompts.
    """
    cache = _load_cache()

    # Check cache, track what needs fetching
    prompt_hashes = [_prompt_hash(p) for p in prompts]
    cached_embeddings: dict[str, np.ndarray] = {}
    to_fetch: list[tuple[str, str]] = []  # (hash, prompt)

    for h, prompt in zip(prompt_hashes, prompts):
        if h in cache:
            cached_embeddings[h] = np.array(cache[h], dtype=np.float32)
        else:
            to_fetch.append((h, prompt))

    # Fetch missing from API
    if to_fetch:
        fetch_prompts = [prompt for _, prompt in to_fetch]
        new_embeddings = _embed_texts(fetch_prompts)

        for (h, _), emb in zip(to_fetch, new_embeddings, strict=True):
            cache[h] = emb.tolist()
            cached_embeddings[h] = emb

        _save_cache(cache)

    # Reconstruct in original order
    return np.array([cached_embeddings[h] for h in prompt_hashes], dtype=np.float32)
