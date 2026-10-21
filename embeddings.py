"""Embeddings module for local Qwen or OpenAI embeddings with disk caching."""

import functools
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from openai import OpenAI

# sentence_transformers imports torch, which is slow to load, so it's only
# imported when the Qwen backend is actually used
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EmbeddingBackend = Literal["qwen", "openai"]
EMBEDDING_MODELS: dict[EmbeddingBackend, str] = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "openai": "text-embedding-3-large",
}

# Runs locally with no API key; OpenAI has to be selected explicitly
DEFAULT_EMBEDDING_BACKEND: EmbeddingBackend = "qwen"

CACHE_DIR = Path(__file__).parent / "cache"


@functools.cache
def _get_openai_client() -> OpenAI:
    """Create the OpenAI client once; later calls return the same client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


@functools.cache
def _get_qwen_model() -> "SentenceTransformer":
    """Load the local Qwen model once (downloading it on first use)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODELS["qwen"])


def _prompt_hash(prompt: str) -> str:
    """Hash prompt content for cache key."""
    return hashlib.md5(prompt.encode()).hexdigest()


def _cache_path(backend: EmbeddingBackend) -> Path:
    """Get the cache file path for a backend's model.

    One file per model, since embeddings from different models aren't comparable.
    """
    model = EMBEDDING_MODELS[backend].replace("/", "--")
    return CACHE_DIR / f"{model}.json"


def _load_cache(backend: EmbeddingBackend) -> dict[str, list[float]]:
    """Load embedding cache from disk. Keyed by prompt hash."""
    path = _cache_path(backend)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, list[float]], backend: EmbeddingBackend) -> None:
    """Save embedding cache to disk, creating the cache directory if needed."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(_cache_path(backend), "w") as f:
        json.dump(cache, f)


def _embed_qwen_texts(texts: list[str], batch_size: int = 4) -> np.ndarray:
    """Embed texts with the local Qwen model. No caching.

    Raises ValueError for any text over the model's token limit, rather than
    letting the model silently truncate it. Small batches keep memory in check
    for long prompts.

    Returns array of shape (n_texts, dim).
    """
    model = _get_qwen_model()

    # No limit means the model won't truncate, so there's nothing to check
    max_tokens = model.max_seq_length
    for text in texts:
        n_tokens = len(model.tokenizer(text, truncation=False)["input_ids"])
        if max_tokens is not None and n_tokens > max_tokens:
            raise ValueError(
                f"Prompt has {n_tokens} tokens, over the model's limit of {max_tokens}"
            )

    embeddings = model.encode(texts, batch_size=batch_size)  # (n_texts, dim)
    return np.asarray(embeddings, dtype=np.float32)


def _embed_openai_texts(texts: list[str], batch_size: int = 20) -> np.ndarray:
    """Call OpenAI API to embed texts in batches. No caching.

    Returns array of shape (n_texts, dim).
    Uses float32 to reduce memory footprint (float64 unnecessary for similarity).
    """
    client = _get_openai_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODELS["openai"], input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def get_embeddings(
    prompts: list[str],
    backend: EmbeddingBackend = DEFAULT_EMBEDDING_BACKEND,
) -> np.ndarray:
    """Get embeddings for prompts, cached by prompt content hash.

    Uses the local Qwen model unless OpenAI is selected explicitly; having
    OPENAI_API_KEY set doesn't change the default. Only prompts missing from
    the cache are embedded, and each one only once, even if it repeats.

    Args:
        prompts: Prompts to embed
        backend: Which embedding model to use

    Returns:
        Array of shape (n_prompts, dim), in the same order as prompts
    """
    cache = _load_cache(backend)

    # Prompts not in the cache yet, each listed once, in first-seen order
    missing = list(dict.fromkeys(p for p in prompts if _prompt_hash(p) not in cache))

    if missing:
        if backend == "qwen":
            new_embeddings = _embed_qwen_texts(missing)  # (n_missing, dim)
        else:
            new_embeddings = _embed_openai_texts(missing)  # (n_missing, dim)

        for prompt, embedding in zip(missing, new_embeddings, strict=True):
            cache[_prompt_hash(prompt)] = embedding.tolist()
        _save_cache(cache, backend)

    # Read every prompt's embedding back from the cache, in input order
    return np.array([cache[_prompt_hash(p)] for p in prompts], dtype=np.float32)
