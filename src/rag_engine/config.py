"""Runtime configuration for the RAG engine.

The configuration is a plain dataclass populated from environment variables.

Design choices worth noting:

- ``RagConfig.from_env`` reads only *non-secret* settings (model names, paths,
  thresholds). Secrets such as ``ANTHROPIC_API_KEY`` are deliberately NOT read
  here: providers read their key from the environment lazily, at call time, so
  importing this package never touches a secret and never fails because one is
  missing.
- Every setting has a sensible offline-first default, so a fresh checkout runs
  with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default directory (relative to the working directory) where the vector index
# is persisted. Kept out of version control via .gitignore.
DEFAULT_INDEX_DIR = ".rag_index"


def _get_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on missing/invalid input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on missing/invalid input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class RagConfig:
    """All tunable settings for a :class:`~rag_engine.pipeline.RagPipeline`.

    Attributes:
        embedder: Which embedder to use (``"hashing"`` or ``"sentence-transformers"``).
        embedding_model: Model name used only by the sentence-transformers embedder.
        embedding_dim: Dimensionality of the hashing embedder's vectors.
        llm_provider: Which answer generator to use
            (``"extractive"``, ``"anthropic"`` or ``"openai"``).
        llm_model: Model name used by the anthropic/openai providers.
        top_k: Number of chunks to retrieve per query.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        similarity_threshold: Minimum cosine similarity for a chunk to count as
            "supporting" evidence. If no chunk clears it, the engine refuses.
        index_dir: Directory where the vector index is saved/loaded.
    """

    # Embeddings
    embedder: str = "hashing"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 256

    # LLM generation
    llm_provider: str = "extractive"
    llm_model: str = "claude-3-5-haiku-latest"

    # Retrieval / chunking
    top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Guardrails
    similarity_threshold: float = 0.15

    # Storage
    index_dir: Path = field(default_factory=lambda: Path(DEFAULT_INDEX_DIR))

    @classmethod
    def from_env(cls) -> "RagConfig":
        """Build a config from environment variables (no secrets read here)."""
        index_dir = os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR)
        return cls(
            embedder=os.environ.get("RAG_EMBEDDER", "hashing"),
            embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_dim=_get_int("RAG_EMBEDDING_DIM", 256),
            llm_provider=os.environ.get("RAG_LLM_PROVIDER", "extractive"),
            llm_model=os.environ.get("RAG_LLM_MODEL", "claude-3-5-haiku-latest"),
            top_k=_get_int("RAG_TOP_K", 4),
            chunk_size=_get_int("RAG_CHUNK_SIZE", 600),
            chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 100),
            similarity_threshold=_get_float("RAG_SIMILARITY_THRESHOLD", 0.15),
            index_dir=Path(index_dir),
        )
