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
- The environment is an input boundary, so it is validated there and trusted
  afterwards. A value that cannot be parsed, or that falls outside the range the
  engine can act on, raises immediately and names the offending variable. It is
  never silently replaced by a default: a typo in a deployment config must fail
  at startup, not turn into a pipeline that quietly answers nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Default directory (relative to the working directory) where the vector index
# is persisted. Kept out of version control via .gitignore.
DEFAULT_INDEX_DIR = ".rag_index"

# The accepted values for the three enum-like settings. Kept as ordered tuples so
# error messages list them in a stable, documented order.
EMBEDDERS = ("hashing", "sentence-transformers")
LLM_PROVIDERS = ("extractive", "anthropic", "openai")
ANONYMIZERS = ("none", "regex", "presidio")

# Spellings accepted on input and rewritten to the canonical value above, so the
# rest of the package only ever compares against one form.
_EMBEDDER_ALIASES = {
    "sentence_transformers": "sentence-transformers",
    "st": "sentence-transformers",
}
_ANONYMIZER_ALIASES = {"off": "none"}

# Bounds for every outbound LLM call. The SDK defaults are far too permissive
# (600 seconds for anthropic), which turns one slow provider into a hung process.
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_MAX_RETRIES = 2
# Retries are bounded so the worst case stays predictable: a wedged provider must
# not be able to hold a request for an unbounded amount of time.
MAX_LLM_RETRIES = 5


def _get_str(name: str, default: str) -> str:
    """Read a string environment variable, treating blank as unset.

    Args:
        name: Environment variable to read.
        default: Value used when the variable is unset or blank.

    Returns:
        The stripped value, or ``default`` when the variable is unset or blank.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _get_int(name: str, default: int) -> int:
    """Read an int environment variable.

    Args:
        name: Environment variable to read.
        default: Value used when the variable is unset or blank.

    Returns:
        The parsed integer, or ``default`` when the variable is unset or blank.

    Raises:
        ValueError: The variable is set to something that is not an integer.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{name} must be an integer, got {raw!r}. "
            f"Unset it to use the default ({default})."
        ) from None


def _get_float(name: str, default: float) -> float:
    """Read a float environment variable.

    Args:
        name: Environment variable to read.
        default: Value used when the variable is unset or blank.

    Returns:
        The parsed float, or ``default`` when the variable is unset or blank.

    Raises:
        ValueError: The variable is set to something that is not a number.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ValueError(
            f"{name} must be a number, got {raw!r}. "
            f"Unset it to use the default ({default})."
        ) from None


def _canonical(value: str, aliases: dict[str, str]) -> str:
    """Normalize an enum-like value: trimmed, lowercased, aliases resolved."""
    normalized = value.strip().lower()
    return aliases.get(normalized, normalized)


def _require_one_of(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Raise unless ``value`` is one of ``allowed``."""
    if value not in allowed:
        options = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of: {options}. Got {value!r}.")


def _require_positive(field_name: str, value: int) -> None:
    """Raise unless ``value`` is strictly positive."""
    if value <= 0:
        raise ValueError(f"{field_name} must be strictly positive, got {value}.")


def _require_within(
    field_name: str, value: float, low: float, high: float
) -> None:
    """Raise unless ``low <= value <= high``."""
    if not low <= value <= high:
        raise ValueError(
            f"{field_name} must be within [{low}, {high}], got {value}."
        )


@dataclass
class RagConfig:
    """All tunable settings for a :class:`~rag_engine.pipeline.RagPipeline`.

    Every field is validated on construction, so a ``RagConfig`` instance is by
    definition a usable configuration: the rest of the package never re-checks it.
    The three enum-like fields are also normalized to their canonical spelling
    (trimmed, lowercased, aliases resolved), so comparisons downstream are exact.

    Attributes:
        embedder: Which embedder to use, one of :data:`EMBEDDERS`.
        embedding_model: Model name used only by the sentence-transformers embedder.
        embedding_dim: Dimensionality of the hashing embedder's vectors, > 0.
        llm_provider: Which answer generator to use, one of :data:`LLM_PROVIDERS`.
        llm_model: Model name used by the anthropic/openai providers.
        llm_timeout_seconds: Timeout applied to every API-backed call, > 0.
        llm_max_retries: Retries the provider SDK may attempt on a failed call,
            within ``[0, MAX_LLM_RETRIES]``. The SDKs back off exponentially
            with jitter between attempts.
        top_k: Number of chunks to retrieve per query, > 0.
        chunk_size: Target chunk size in characters, > 0.
        chunk_overlap: Overlap between consecutive chunks in characters, within
            ``[0, chunk_size)``. An overlap equal to the chunk size would stop the
            sliding window from advancing.
        similarity_threshold: Minimum cosine similarity for a chunk to count as
            "supporting" evidence, within ``[-1.0, 1.0]``. If no chunk clears it,
            the engine refuses to answer.
        anonymizer: PII anonymizer applied at ingestion, one of :data:`ANONYMIZERS`.
        anonymize_model: spaCy model name used by the presidio anonymizer.
        anonymize_threshold: Minimum detector confidence for the presidio
            anonymizer to redact an entity, within ``[0.0, 1.0]``.
        index_dir: Directory where the vector index is saved/loaded.

    Raises:
        ValueError: Any field is outside the range or set of values above.
    """

    # Embeddings
    embedder: str = "hashing"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 256

    # LLM generation
    llm_provider: str = "extractive"
    llm_model: str = "claude-3-5-haiku-latest"
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES

    # Retrieval / chunking
    top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Guardrails
    similarity_threshold: float = 0.15

    # PII anonymization (applied at ingestion, before embedding/indexing)
    anonymizer: str = "none"
    anonymize_model: str = "en_core_web_sm"
    anonymize_threshold: float = 0.5

    # Storage
    index_dir: Path = field(default_factory=lambda: Path(DEFAULT_INDEX_DIR))

    def __post_init__(self) -> None:
        """Validate every field, whether it came from the environment or from code.

        Raises:
            ValueError: A field is outside its accepted set or range.
        """
        self.embedder = _canonical(self.embedder, _EMBEDDER_ALIASES)
        self.llm_provider = _canonical(self.llm_provider, {})
        self.anonymizer = _canonical(self.anonymizer, _ANONYMIZER_ALIASES)

        _require_one_of("embedder", self.embedder, EMBEDDERS)
        _require_one_of("llm_provider", self.llm_provider, LLM_PROVIDERS)
        _require_one_of("anonymizer", self.anonymizer, ANONYMIZERS)

        _require_positive("embedding_dim", self.embedding_dim)
        _require_positive("top_k", self.top_k)
        _require_positive("chunk_size", self.chunk_size)

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be within [0, chunk_size), "
                f"got {self.chunk_overlap} for chunk_size {self.chunk_size}."
            )

        # Cosine similarity lives in [-1, 1]: a threshold outside that range
        # either refuses every question or filters nothing.
        _require_within(
            "similarity_threshold", self.similarity_threshold, -1.0, 1.0
        )
        _require_within("anonymize_threshold", self.anonymize_threshold, 0.0, 1.0)

        if self.llm_timeout_seconds <= 0:
            raise ValueError(
                "llm_timeout_seconds must be strictly positive, got "
                f"{self.llm_timeout_seconds}."
            )
        if not 0 <= self.llm_max_retries <= MAX_LLM_RETRIES:
            raise ValueError(
                f"llm_max_retries must be within [0, {MAX_LLM_RETRIES}], got "
                f"{self.llm_max_retries}."
            )

        # Accept a plain string for convenience, but store the validated type.
        if not isinstance(self.index_dir, Path):
            self.index_dir = Path(self.index_dir)

    @classmethod
    def from_env(cls) -> "RagConfig":
        """Build a config from environment variables (no secrets read here).

        Returns:
            A validated configuration.

        Raises:
            ValueError: An environment variable is unparseable or out of range.
                The message names the variable and its accepted values.
        """
        index_dir = _get_str("RAG_INDEX_DIR", DEFAULT_INDEX_DIR)
        return cls(
            embedder=_get_str("RAG_EMBEDDER", "hashing"),
            embedding_model=_get_str("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_dim=_get_int("RAG_EMBEDDING_DIM", 256),
            llm_provider=_get_str("RAG_LLM_PROVIDER", "extractive"),
            llm_model=_get_str("RAG_LLM_MODEL", "claude-3-5-haiku-latest"),
            llm_timeout_seconds=_get_float(
                "RAG_LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS
            ),
            llm_max_retries=_get_int("RAG_LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES),
            top_k=_get_int("RAG_TOP_K", 4),
            chunk_size=_get_int("RAG_CHUNK_SIZE", 600),
            chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 100),
            similarity_threshold=_get_float("RAG_SIMILARITY_THRESHOLD", 0.15),
            anonymizer=_get_str("RAG_ANONYMIZER", "none"),
            anonymize_model=_get_str("RAG_ANONYMIZE_MODEL", "en_core_web_sm"),
            anonymize_threshold=_get_float("RAG_ANONYMIZE_THRESHOLD", 0.5),
            index_dir=Path(index_dir),
        )
