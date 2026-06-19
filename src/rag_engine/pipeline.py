"""The RagPipeline: wires ingestion, retrieval, guardrails, and generation.

This is the single entry point most callers need. It owns the embedder, vector
store, retriever, and LLM, and exposes two operations:

- :meth:`RagPipeline.ingest` — load a folder of documents, embed and index them,
  and persist the index to disk.
- :meth:`RagPipeline.answer` — retrieve context for a question, apply the
  grounding guardrail, generate an answer, and return it with citations.

The pipeline is deliberately thin: each step lives in its own module and is
unit-testable in isolation. The pipeline just composes them in the right order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_engine.config import RagConfig
from rag_engine.embeddings import Embedder, build_embedder
from rag_engine.guardrails import (
    REFUSAL_MESSAGE,
    Citation,
    build_citations,
    passes_grounding,
)
from rag_engine.ingestion import load_and_chunk
from rag_engine.llm import LLM, build_llm
from rag_engine.retriever import Retriever
from rag_engine.vectorstore import SearchResult, VectorStore


@dataclass
class RagAnswer:
    """The result of :meth:`RagPipeline.answer`.

    Attributes:
        answer: The generated answer, or a refusal message.
        citations: Sources supporting the answer (empty when refused).
        used_chunks: The retrieved chunks considered for this question.
        refused: True if the grounding guardrail blocked an answer.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    used_chunks: list[SearchResult] = field(default_factory=list)
    refused: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict (useful for CLI/JSON output)."""
        return {
            "answer": self.answer,
            "refused": self.refused,
            "citations": [asdict(c) for c in self.citations],
            "used_chunks": [
                {
                    "source": r.chunk.source,
                    "chunk_index": r.chunk.chunk_index,
                    "score": round(r.score, 4),
                }
                for r in self.used_chunks
            ],
        }


class RagPipeline:
    """End-to-end Retrieval-Augmented Generation over a local corpus."""

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ) -> None:
        """Create a pipeline.

        Args:
            config: Configuration. Defaults to :meth:`RagConfig.from_env`.
            embedder: Optional embedder override (mainly for testing). If not
                given, one is built from ``config``.
            llm: Optional LLM override (mainly for testing). If not given, one is
                built from ``config``.
        """
        self.config = config or RagConfig.from_env()
        self._embedder = embedder or build_embedder(self.config)
        self._llm = llm or build_llm(self.config)
        # The store is created on ingest() or loaded on demand by answer().
        self._store: VectorStore | None = None

    # -- Ingestion -------------------------------------------------------- #

    def ingest(self, folder: str | Path) -> int:
        """Load, chunk, embed, and index every document in ``folder``.

        The resulting index is persisted to ``config.index_dir`` so subsequent
        ``answer`` calls (even in a new process) can load it.

        Args:
            folder: Directory of ``.txt`` / ``.md`` documents.

        Returns:
            The number of chunks indexed.
        """
        chunks = load_and_chunk(
            folder,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        store = VectorStore(dim=self._embedder.dim)
        if chunks:
            vectors = self._embedder.embed([c.text for c in chunks])
            store.add(vectors, chunks)
        store.save(self.config.index_dir)
        self._store = store
        return len(chunks)

    # -- Querying --------------------------------------------------------- #

    def _ensure_store(self) -> VectorStore:
        """Return the in-memory store, loading it from disk if needed."""
        if self._store is None:
            self._store = VectorStore.load(self.config.index_dir)
        return self._store

    def answer(self, question: str) -> RagAnswer:
        """Answer ``question`` from the indexed corpus, with grounding + citations.

        Steps: retrieve top-k chunks -> apply the grounding guardrail -> if the
        gate fails, refuse; otherwise generate an answer and attach citations.

        Args:
            question: The user's question.

        Returns:
            A :class:`RagAnswer`. ``refused`` is True when no retrieved chunk
            cleared the configured similarity threshold.
        """
        store = self._ensure_store()
        retriever = Retriever(self._embedder, store)
        results = retriever.retrieve(question, top_k=self.config.top_k)

        # Grounding gate: refuse rather than answer from an unsupported context.
        if not passes_grounding(results, self.config.similarity_threshold):
            return RagAnswer(
                answer=REFUSAL_MESSAGE,
                citations=[],
                used_chunks=results,
                refused=True,
            )

        # Only feed the grounded chunks to the generator.
        grounded = [
            r for r in results if r.score >= self.config.similarity_threshold
        ]
        answer_text = self._llm.generate(question, grounded)
        citations = build_citations(results, self.config.similarity_threshold)
        return RagAnswer(
            answer=answer_text,
            citations=citations,
            used_chunks=results,
            refused=False,
        )
