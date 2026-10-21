"""Embeddings module for local Qwen or OpenAI embeddings with disk caching."""

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
CACHE_DIR = Path(__file__).parent / "cache"

_client: OpenAI | None = None
_qwen_model: "SentenceTransformer | None" = None


def _get_client() -> OpenAI:
    """Get or create singleton OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _client = OpenAI(api_key=api_key)
    return _client


def _get_qwen_model() -> "SentenceTransformer":
    """Get or load singleton local Qwen model. Downloads it on first use."""
    global _qwen_model
    if _qwen_model is None:
        from sentence_transformers import SentenceTransformer

        _qwen_model = SentenceTransformer(EMBEDDING_MODELS["qwen"])
    return _qwen_model


def _prompt_hash(prompt: str) -> str:
    """Hash prompt content for cache key."""
    return hashlib.md5(prompt.encode()).hexdigest()


def _cache_path(backend: EmbeddingBackend) -> Path:
    """Get cache file path for a backend's model, creating directory if needed.

    One file per model, since embeddings from different models aren't comparable.
    """
    CACHE_DIR.mkdir(exist_ok=True)
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
    """Save embedding cache to disk."""
    with open(_cache_path(backend), "w") as f:
        json.dump(cache, f)


def _embed_qwen_texts(texts: list[str], batch_size: int = 4) -> np.ndarray:
    """Embed texts with the local Qwen model. No caching.

    Raises ValueError for any text over the model's token limit, rather than
    letting the model silently truncate it. Small batches keep memory in check
    for long prompts.

    Returns array of shape (n, embedding_dim).
    """
    model = _get_qwen_model()
    for text in texts:
        n_tokens = len(model.tokenizer(text, truncation=False)["input_ids"])
        if n_tokens > model.max_seq_length:
            raise ValueError(
                f"Prompt has {n_tokens} tokens, over the model's limit of "
                f"{model.max_seq_length}"
            )

    embeddings = model.encode(texts, batch_size=batch_size)
    return np.asarray(embeddings, dtype=np.float32)


def _embed_openai_texts(texts: list[str], batch_size: int = 20) -> np.ndarray:
    """Call OpenAI API to embed texts in batches. No caching.

    Returns array of shape (n, embedding_dim).
    Uses float32 to reduce memory footprint (float64 unnecessary for similarity).
    """
    client = _get_client()
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
    backend: EmbeddingBackend = "qwen",
) -> np.ndarray:
    """Get embeddings for prompts, cached by prompt content hash.

    Uses the local Qwen model unless OpenAI is selected explicitly; having
    OPENAI_API_KEY set doesn't change the default.
    Cache is checked first; only missing prompts are embedded.
    Returns array of shape (n, embedding_dim) in same order as input prompts.
    """
    cache = _load_cache(backend)

    # Check cache, track what needs embedding
    prompt_hashes = [_prompt_hash(p) for p in prompts]
    cached_embeddings: dict[str, np.ndarray] = {}
    to_embed: list[tuple[str, str]] = []  # (hash, prompt)

    for h, prompt in zip(prompt_hashes, prompts):
        if h in cache:
            cached_embeddings[h] = np.array(cache[h], dtype=np.float32)
        else:
            to_embed.append((h, prompt))

    # Embed missing prompts with the selected backend
    if to_embed:
        new_prompts = [prompt for _, prompt in to_embed]
        if backend == "qwen":
            new_embeddings = _embed_qwen_texts(new_prompts)
        else:
            new_embeddings = _embed_openai_texts(new_prompts)

        for (h, _), emb in zip(to_embed, new_embeddings, strict=True):
            cache[h] = emb.tolist()
            cached_embeddings[h] = emb

        _save_cache(cache, backend)

    # Reconstruct in original order
    return np.array([cached_embeddings[h] for h in prompt_hashes], dtype=np.float32)
