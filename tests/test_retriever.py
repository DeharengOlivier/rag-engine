"""Tests for the retriever using the offline hashing embedder."""

from __future__ import annotations

from rag_engine.config import RagConfig
from rag_engine.embeddings import build_embedder
from rag_engine.ingestion import Chunk
from rag_engine.retriever import Retriever
from rag_engine.vectorstore import VectorStore


def _build_retriever() -> Retriever:
    config = RagConfig(embedder="hashing", embedding_dim=256)
    embedder = build_embedder(config)

    chunks = [
        Chunk(
            text="The library is open on Saturday from 10:00 to 16:00.",
            source="library.md",
            chunk_index=0,
        ),
        Chunk(
            text="Recycling is collected every Thursday morning.",
            source="recycling.md",
            chunk_index=0,
        ),
        Chunk(
            text="Riverside Park has a playground and a duck pond.",
            source="parks.md",
            chunk_index=0,
        ),
    ]
    store = VectorStore(dim=embedder.dim)
    store.add(embedder.embed([c.text for c in chunks]), chunks)
    return Retriever(embedder, store)


def test_retrieves_relevant_chunk_for_obvious_query():
    retriever = _build_retriever()
    results = retriever.retrieve("When is recycling collected?", top_k=3)

    assert results, "should retrieve at least one chunk"
    # The recycling chunk should be the top hit for a recycling question.
    assert results[0].chunk.source == "recycling.md"


def test_blank_query_returns_no_results():
    retriever = _build_retriever()
    assert retriever.retrieve("   ", top_k=3) == []
