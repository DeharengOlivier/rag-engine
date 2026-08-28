"""A minimal local vector store.

Stores embedding vectors in a single numpy matrix alongside the chunks they
came from, and answers nearest-neighbor queries by cosine similarity.

Why a numpy matrix instead of an external service (FAISS, a vector DB, ...)?

- Zero infrastructure: it runs anywhere numpy runs, with no server to manage.
- For the small-to-medium corpora this engine targets, a vectorized dot product
  over a normalized matrix is fast and exact.
- Persistence is just a ``.npy`` file (the matrix) plus a small JSON sidecar
  (the chunk text and metadata), which is easy to inspect and version-control-ignore.

The interface (``add`` / ``search`` / ``save`` / ``load``) is deliberately the
same shape a real backend would expose, so swapping in FAISS later is localized.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag_engine.ingestion import Chunk

logger = logging.getLogger(__name__)

# File names used inside the index directory.
_VECTORS_FILE = "vectors.npy"
_META_FILE = "meta.json"


@dataclass
class SearchResult:
    """A single search hit: the matched chunk and its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """An in-memory, numpy-backed vector store with disk persistence.

    Vectors are expected to be L2-normalized (the provided embedders do this),
    so cosine similarity is computed as a plain dot product. ``search`` does not
    assume normalization of the query and normalizes it defensively.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Append ``vectors`` (one row per chunk) and their ``chunks``.

        Raises:
            ValueError: On a row/chunk count mismatch or wrong dimensionality.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected vectors of shape (n, {self._dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != chunk count ({len(chunks)})"
            )
        if vectors.shape[0] == 0:
            return
        self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Return the ``top_k`` most similar chunks to ``query_vector``.

        Args:
            query_vector: A 1-D vector of length ``dim``.
            top_k: Maximum number of results to return.

        Returns:
            Results sorted by descending cosine similarity. Empty if the store
            is empty or ``top_k <= 0``.
        """
        if len(self._chunks) == 0 or top_k <= 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(
                f"Query vector has length {q.shape[0]}, expected {self._dim}"
            )
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Cosine similarity == dot product for normalized stored vectors.
        scores = self._vectors @ q
        k = min(top_k, scores.shape[0])
        # argpartition gives the top-k cheaply, then we sort just those k.
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_idx
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the store to ``directory`` (created if needed)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS_FILE, self._vectors)
        meta = {
            "dim": self._dim,
            "chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                }
                for c in self._chunks
            ],
        }
        (directory / _META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "index saved dir=%s vectors=%d dim=%d",
            directory,
            len(self._chunks),
            self._dim,
        )

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        """Load a store previously written by :meth:`save`.

        Raises:
            FileNotFoundError: If the index files are missing.
        """
        directory = Path(directory)
        vectors_path = directory / _VECTORS_FILE
        meta_path = directory / _META_FILE
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No vector index found in '{directory}'. Run ingestion first."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(dim=int(meta["dim"]))
        store._vectors = np.load(vectors_path).astype(np.float32)
        store._chunks = [
            Chunk(
                text=c["text"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                metadata=c.get("metadata", {}),
            )
            for c in meta["chunks"]
        ]
        logger.info(
            "index loaded dir=%s vectors=%d dim=%d",
            directory,
            len(store._chunks),
            store._dim,
        )
        return store
