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
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag_engine.ingestion import Chunk

logger = logging.getLogger(__name__)

# File names used inside the index directory.
_VECTORS_FILE = "vectors.npy"
_META_FILE = "meta.json"

# Sibling directories used to swap a new index into place atomically. They only
# exist for the duration of a save, and are removed even when one fails.
_STAGING_SUFFIX = ".staging"
_PREVIOUS_SUFFIX = ".previous"


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
            SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the store to ``directory``, replacing any index already there.

        The index spans two files that must agree with each other, so the write
        is staged in a sibling directory and swapped into place. An interrupted
        save therefore leaves the previous index intact, instead of a pair of
        files describing two different indexes.

        Args:
            directory: Where the index lives. Created if needed.

        Raises:
            OSError: The index could not be written or swapped into place. The
                index that was already there, if any, is left untouched.
        """
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = directory.parent / f"{directory.name}{_STAGING_SUFFIX}"
        previous = directory.parent / f"{directory.name}{_PREVIOUS_SUFFIX}"

        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            np.save(staging / _VECTORS_FILE, self._vectors)
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
            (staging / _META_FILE).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._swap_into_place(staging, directory, previous)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(previous, ignore_errors=True)

        logger.info(
            "index saved dir=%s vectors=%d dim=%d",
            directory,
            len(self._chunks),
            self._dim,
        )

    @staticmethod
    def _swap_into_place(staging: Path, directory: Path, previous: Path) -> None:
        """Move ``staging`` onto ``directory``, restoring the old index on failure."""
        if directory.exists():
            os.replace(directory, previous)
        try:
            os.replace(staging, directory)
        except OSError:
            # The old index has already been moved aside: put it back before
            # giving up, so a failed save is a no-op rather than a data loss.
            if previous.exists() and not directory.exists():
                os.replace(previous, directory)
            raise

    @staticmethod
    def _read_metadata(meta_path: Path, directory: Path) -> dict[str, Any]:
        """Parse the metadata sidecar, or say precisely why it cannot be used."""
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"'{_META_FILE}' in '{directory}' is not readable JSON ({exc}). "
                "Re-run ingestion to rebuild the index."
            ) from exc
        if not isinstance(meta, dict) or "dim" not in meta or "chunks" not in meta:
            raise ValueError(
                f"'{_META_FILE}' in '{directory}' is missing 'dim' or 'chunks'. "
                "Re-run ingestion to rebuild the index."
            )
        try:
            meta["dim"] = int(meta["dim"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{_META_FILE}' in '{directory}' declares a non-numeric dim "
                f"({meta['dim']!r}). Re-run ingestion to rebuild the index."
            ) from exc
        if meta["dim"] <= 0:
            raise ValueError(
                f"'{_META_FILE}' in '{directory}' declares dim {meta['dim']}, "
                "which cannot be a vector width. Re-run ingestion to rebuild it."
            )
        return meta

    @staticmethod
    def _check_consistency(
        vectors: np.ndarray,
        chunk_entries: list[Any],
        dim: int,
        directory: Path,
    ) -> None:
        """Raise unless the two persisted files describe the same index.

        Raises:
            ValueError: The vectors and the metadata disagree on the number of
                entries or on the vector width.
        """
        if vectors.ndim != 2 or vectors.shape[1] != dim:
            raise ValueError(
                f"Index in '{directory}' is inconsistent: metadata declares dim "
                f"{dim} but the stored vectors have shape {vectors.shape}. "
                "Re-run ingestion to rebuild the index."
            )
        if len(chunk_entries) != vectors.shape[0]:
            raise ValueError(
                f"Index in '{directory}' is inconsistent: {vectors.shape[0]} "
                f"vector(s) but {len(chunk_entries)} chunk(s). An interrupted "
                "save can leave the two files out of step. Re-run ingestion to "
                "rebuild the index."
            )

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        """Load a store previously written by :meth:`save`.

        The two files are checked against each other before the store is handed
        back: a mismatch means the index on disk cannot be trusted, and the
        caller is told to rebuild rather than given a store that fails later,
        mid-query, with an error pointing nowhere near the cause.

        Args:
            directory: Directory holding the index.

        Returns:
            The loaded store.

        Raises:
            FileNotFoundError: The index files are missing.
            ValueError: The index is unreadable or internally inconsistent.
        """
        directory = Path(directory)
        vectors_path = directory / _VECTORS_FILE
        meta_path = directory / _META_FILE
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No vector index found in '{directory}'. Run ingestion first."
            )

        meta = cls._read_metadata(meta_path, directory)
        dim = meta["dim"]
        vectors = np.load(vectors_path).astype(np.float32)
        cls._check_consistency(vectors, meta["chunks"], dim, directory)

        store = cls(dim=dim)
        store._vectors = vectors
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
