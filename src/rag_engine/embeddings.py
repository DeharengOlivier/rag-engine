"""Text embedders: turn strings into dense vectors.

Two implementations are provided behind a common :class:`Embedder` protocol:

- :class:`HashingEmbedder` (default): a pure-numpy, dependency-free embedder.
  It maps tokens into a fixed-dimensional space via a hashing trick, then
  L2-normalizes. It needs no model download and no network, so the whole engine
  runs offline out of the box. It is deterministic and good enough to retrieve
  obviously-relevant chunks, which makes it ideal for tests and demos.

- :class:`SentenceTransformerEmbedder` (optional): wraps the
  ``sentence-transformers`` library for real semantic embeddings. The import is
  lazy, so the dependency is only required if you actually select this embedder.

``build_embedder(config)`` selects the implementation from configuration.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from rag_engine.config import RagConfig

# Token pattern: words and numbers, lowercased. Simple and language-agnostic.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer used by the hashing embedder."""
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    """Common interface for all embedders."""

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a ``(len(texts), dim)`` float32 array."""
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row, leaving all-zero rows untouched (avoids div-by-zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class HashingEmbedder:
    """Deterministic, offline embedder using the hashing trick on tokens.

    Each token is hashed into a bucket in ``dim``-dimensional space. We use a
    second hash bit to assign a sign, which reduces collision bias (signed
    hashing / "feature hashing"). The resulting bag-of-tokens vector is
    L2-normalized so cosine similarity reduces to a dot product.

    This is not a semantic model, but it reliably surfaces chunks that share
    vocabulary with the query, which is exactly what a deterministic test and an
    offline demo need.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _tokenize(text):
            # Stable hash across processes (Python's built-in hash() is salted).
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        matrix = np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)
        return _l2_normalize(matrix)


class SentenceTransformerEmbedder:
    """Optional embedder backed by ``sentence-transformers`` (lazy import).

    The model is loaded on construction; importing this class does not import
    the heavy dependency. Selecting this embedder without the package installed
    raises a clear, actionable error.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only with extra installed
            raise ImportError(
                "The 'sentence-transformers' package is required for the "
                "sentence-transformers embedder. Install it with "
                "'pip install \"rag-engine[embeddings]\"', or set "
                "RAG_EMBEDDER=hashing to run offline."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(config: RagConfig) -> Embedder:
    """Instantiate the embedder selected in ``config``.

    Raises:
        ValueError: If ``config.embedder`` is not a recognized value.
    """
    name = config.embedder.lower()
    if name == "hashing":
        return HashingEmbedder(dim=config.embedding_dim)
    if name in ("sentence-transformers", "sentence_transformers", "st"):
        return SentenceTransformerEmbedder(model_name=config.embedding_model)
    raise ValueError(
        f"Unknown embedder '{config.embedder}'. "
        "Expected 'hashing' or 'sentence-transformers'."
    )
