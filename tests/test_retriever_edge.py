"""Edge-case tests for the retriever (top-k limits, ties, empty store)."""

from __future__ import annotations

import numpy as np

from rag_engine.config import RagConfig
from rag_engine.embeddings import build_embedder
from rag_engine.ingestion import Chunk
from rag_engine.retriever import Retriever
from rag_engine.vectorstore import VectorStore


def _retriever_from(chunks: list[Chunk]) -> Retriever:
    embedder = build_embedder(RagConfig(embedder="hashing", embedding_dim=256))
    store = VectorStore(dim=embedder.dim)
    if chunks:
        store.add(embedder.embed([c.text for c in chunks]), chunks)
    return Retriever(embedder, store)


def test_top_k_caps_number_of_results():
    chunks = [
        Chunk(text=f"document number {i}", source=f"d{i}.md", chunk_index=0)
        for i in range(10)
    ]
    retriever = _retriever_from(chunks)
    results = retriever.retrieve("document", top_k=3)
    assert len(results) == 3


def test_empty_query_on_populated_store_returns_empty():
    chunks = [Chunk(text="hello", source="a.md", chunk_index=0)]
    retriever = _retriever_from(chunks)
    assert retriever.retrieve("", top_k=5) == []


def test_retrieve_on_empty_store_returns_empty():
    retriever = _retriever_from([])
    assert retriever.retrieve("anything", top_k=5) == []


def test_results_sorted_descending_by_score():
    chunks = [
        Chunk(text="recycling collection thursday morning", source="r.md", chunk_index=0),
        Chunk(text="library opening hours saturday", source="l.md", chunk_index=0),
        Chunk(text="park playground duck pond", source="p.md", chunk_index=0),
    ]
    retriever = _retriever_from(chunks)
    results = retriever.retrieve("when is recycling collected", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_identical_chunks_produce_tied_scores():
    # Two chunks with identical text embed identically, so both tie at the top.
    chunks = [
        Chunk(text="exact same content here", source="a.md", chunk_index=0),
        Chunk(text="exact same content here", source="b.md", chunk_index=0),
    ]
    retriever = _retriever_from(chunks)
    results = retriever.retrieve("exact same content here", top_k=2)
    assert len(results) == 2
    assert np.isclose(results[0].score, results[1].score)
