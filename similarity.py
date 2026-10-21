"""Similarity search for nearest neighbor retrieval."""

from dataclasses import dataclass, field

import numpy as np

from match_types import MatchRow


@dataclass(eq=False)  # No ==: comparing the embedding arrays would raise
class TrainIndex:
    """Training match rows indexed for similarity search.

    Bundles match rows and their embeddings together - they're parallel
    arrays where index i in both refers to the same training match row.
    In the shape comments, n is the number of training match rows and dim is
    the embedding dimension.

    Attributes:
        match_rows: Training match rows
        embeddings: Prompt embedding for each match row
    """

    match_rows: list[MatchRow]
    embeddings: np.ndarray  # (n, dim)
    _normed_embeddings: np.ndarray = field(init=False, repr=False)  # (n, dim)

    def __post_init__(self) -> None:
        """Validate inputs and precompute normalized embeddings.

        Called automatically by dataclass after __init__.
        """
        if len(self.match_rows) != len(self.embeddings):
            raise ValueError(
                f"match_rows ({len(self.match_rows)}) and embeddings "
                f"({len(self.embeddings)}) must have same length"
            )
        # Normalize embeddings for cosine similarity (dot product on unit vectors)
        # keepdims=True gives shape (n, 1) for broadcasting in division
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)  # (n, 1)
        self._normed_embeddings = self.embeddings / norms  # (n, dim)

    def get_nearest_neighbors(
        self,
        query_embedding: np.ndarray,
        n_neighbors: int,
    ) -> list[MatchRow]:
        """Return the n_neighbors most similar training match rows.

        Similarity is cosine similarity between prompt embeddings.

        Args:
            query_embedding: Embedding vector for the query, shape (dim,)
            n_neighbors: Number of neighbors to return

        Returns:
            List of n_neighbors MatchRows, ordered by descending similarity
        """
        # Normalize query
        query_normed = query_embedding / np.linalg.norm(query_embedding)  # (dim,)

        # Cosine similarity = dot product of normalized vectors
        similarities = self._normed_embeddings @ query_normed  # (n,)

        # Get indices of the top n_neighbors (descending similarity)
        top_indices = np.argsort(similarities)[::-1][:n_neighbors]  # (n_neighbors,)

        return [self.match_rows[i] for i in top_indices]
