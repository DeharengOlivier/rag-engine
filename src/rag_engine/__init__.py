"""rag-engine: a small, offline-first Retrieval-Augmented Generation engine.

The package is organized into single-responsibility modules:

- ``config``      : runtime configuration loaded from environment variables.
- ``errors``      : the exception types the package raises.
- ``observability``: the package logger, and its configuration for applications.
- ``ingestion``   : load and chunk text documents from a folder.
- ``anonymizer``  : PII redaction (offline regex + optional Presidio) at ingestion.
- ``embeddings``  : pluggable text-to-vector embedders (offline + optional models).
- ``vectorstore`` : local numpy-backed vector store with cosine similarity.
- ``retriever``   : turn a question into the top-k most relevant chunks.
- ``llm``         : pluggable answer generators (offline extractive + optional APIs).
- ``guardrails``  : grounding checks, citations, and safe refusals.
- ``pipeline``    : the ``RagPipeline`` that wires everything together.
- ``evaluation``  : a tiny harness to measure retrieval and answer quality.

Everything runs fully offline by default (hashing embedder + extractive LLM),
so no API key or network access is required.
"""

from rag_engine.anonymizer import AnonymizationResult, build_anonymizer
from rag_engine.config import RagConfig
from rag_engine.errors import ConfigError
from rag_engine.observability import configure_logging
from rag_engine.pipeline import RagPipeline

__all__ = [
    "AnonymizationResult",
    "ConfigError",
    "RagConfig",
    "RagPipeline",
    "build_anonymizer",
    "configure_logging",
]
__version__ = "0.1.0"
