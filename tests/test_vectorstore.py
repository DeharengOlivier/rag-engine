"""Tests for the numpy-backed vector store."""

from __future__ import annotations

import numpy as np

from rag_engine.ingestion import Chunk
from rag_engine.vectorstore import VectorStore


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(text=text, source="doc.md", chunk_index=idx)


def test_add_and_topk_ordering():
    store = VectorStore(dim=3)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # most similar to the query below
            [0.0, 1.0, 0.0],  # orthogonal
            [0.9, 0.1, 0.0],  # close to the first
        ],
        dtype=np.float32,
    )
    chunks = [_chunk("a", 0), _chunk("b", 1), _chunk("c", 2)]
    store.add(vectors, chunks)

    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)

    assert len(results) == 3
    # Scores must be sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The exact match (chunk "a") must rank first.
    assert results[0].chunk.text == "a"
    # The orthogonal vector ("b") must rank last.
    assert results[-1].chunk.text == "b"


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore(dim=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.add(vectors, [_chunk("x", 0), _chunk("y", 1)])
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)

    assert len(loaded) == 2
    results = loaded.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0].chunk.text == "x"
    assert results[0].chunk.chunk_index == 0


def test_search_on_empty_store_returns_empty():
    store = VectorStore(dim=4)
    assert store.search(np.zeros(4, dtype=np.float32), top_k=3) == []
