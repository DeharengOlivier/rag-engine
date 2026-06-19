"""Edge-case tests for the numpy-backed vector store."""

from __future__ import annotations

import numpy as np
import pytest

from rag_engine.ingestion import Chunk
from rag_engine.vectorstore import VectorStore


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, source="doc.md", chunk_index=idx)


def test_zero_dim_is_rejected():
    with pytest.raises(ValueError):
        VectorStore(dim=0)


def test_add_wrong_dimension_raises():
    store = VectorStore(dim=3)
    bad = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        store.add(bad, [_chunk("a")])


def test_add_row_count_mismatch_raises():
    store = VectorStore(dim=2)
    vectors = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        store.add(vectors, [_chunk("only-one")])


def test_add_empty_is_noop():
    store = VectorStore(dim=2)
    store.add(np.zeros((0, 2), dtype=np.float32), [])
    assert len(store) == 0


def test_add_1d_vector_rejected():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add(np.zeros(3, dtype=np.float32), [_chunk("a")])


def test_search_query_dimension_mismatch_raises():
    store = VectorStore(dim=3)
    store.add(np.eye(3, dtype=np.float32), [_chunk("a"), _chunk("b"), _chunk("c")])
    with pytest.raises(ValueError):
        store.search(np.zeros(2, dtype=np.float32), top_k=1)


def test_search_topk_zero_or_negative_returns_empty():
    store = VectorStore(dim=2)
    store.add(np.eye(2, dtype=np.float32), [_chunk("a"), _chunk("b")])
    assert store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=0) == []
    assert store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=-5) == []


def test_search_topk_larger_than_store_is_clamped():
    store = VectorStore(dim=2)
    store.add(np.eye(2, dtype=np.float32), [_chunk("a"), _chunk("b")])
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=99)
    assert len(results) == 2


def test_search_normalizes_unnormalized_query():
    store = VectorStore(dim=2)
    store.add(np.eye(2, dtype=np.float32), [_chunk("a"), _chunk("b")])
    # A query scaled by 1000 must give the same score as the unit query.
    big = store.search(np.array([1000.0, 0.0], dtype=np.float32), top_k=1)
    unit = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert big[0].score == pytest.approx(unit[0].score)
    assert big[0].score == pytest.approx(1.0)


def test_search_with_tied_scores_returns_all_tied():
    # Two stored vectors equally similar to the query: both must appear, and the
    # top score must be the shared (tied) value.
    store = VectorStore(dim=2)
    v = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    store.add(v, [_chunk("a"), _chunk("b")])

    results = store.search(np.array([1.0, 1.0], dtype=np.float32), top_k=2)
    assert len(results) == 2
    assert results[0].score == pytest.approx(results[1].score)
    assert {r.chunk.text for r in results} == {"a", "b"}


def test_save_load_determinism_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5, 8)).astype(np.float32)
    store = VectorStore(dim=8)
    store.add(raw, [_chunk(f"c{i}", i) for i in range(5)])
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    q = rng.standard_normal(8).astype(np.float32)

    before = store.search(q, top_k=5)
    after = loaded.search(q, top_k=5)

    # Persisted index must reproduce the same ranking and scores bit-for-bit.
    assert [r.chunk.text for r in before] == [r.chunk.text for r in after]
    assert [r.score for r in before] == [r.score for r in after]
    assert loaded.dim == store.dim


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "no-index-here")


def test_save_preserves_metadata(tmp_path):
    store = VectorStore(dim=2)
    chunk = Chunk(text="hi", source="doc.md", chunk_index=3, metadata={"k": "v"})
    store.add(np.array([[1.0, 0.0]], dtype=np.float32), [chunk])
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    res = loaded.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert res[0].chunk.metadata == {"k": "v"}
    assert res[0].chunk.chunk_index == 3
