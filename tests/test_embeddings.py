"""Tests for the offline hashing embedder and embedder selection."""

from __future__ import annotations

import numpy as np
import pytest

from rag_engine.config import RagConfig
from rag_engine.embeddings import Embedder, HashingEmbedder, build_embedder


def test_hashing_embedder_shape_and_normalization():
    emb = HashingEmbedder(dim=64)
    vectors = emb.embed(["hello world", "another piece of text here"])

    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    # Each non-empty row is L2-normalized to unit length.
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_is_deterministic_across_instances():
    a = HashingEmbedder(dim=128).embed(["the quick brown fox"])
    b = HashingEmbedder(dim=128).embed(["the quick brown fox"])
    # Stable hashing (md5) must give identical vectors across instances/processes.
    assert np.array_equal(a, b)


def test_hashing_embedder_satisfies_protocol():
    emb = HashingEmbedder(dim=32)
    assert isinstance(emb, Embedder)
    assert emb.dim == 32


def test_hashing_embedder_empty_list_returns_empty_matrix():
    emb = HashingEmbedder(dim=16)
    out = emb.embed([])
    assert out.shape == (0, 16)


def test_hashing_embedder_empty_string_row_is_all_zero():
    emb = HashingEmbedder(dim=16)
    out = emb.embed(["", "real words"])
    # No tokens -> zero vector left untouched by the normalizer.
    assert np.allclose(out[0], 0.0)
    assert np.linalg.norm(out[1]) == pytest.approx(1.0, abs=1e-5)


def test_hashing_embedder_rejects_nonpositive_dim():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_similar_texts_have_higher_cosine_than_unrelated():
    emb = HashingEmbedder(dim=512)
    v = emb.embed(
        [
            "recycling is collected on thursday",
            "recycling collection happens thursday morning",
            "the library opens on saturday at ten",
        ]
    )
    related = float(v[0] @ v[1])
    unrelated = float(v[0] @ v[2])
    assert related > unrelated


def test_build_embedder_returns_hashing_by_default():
    emb = build_embedder(RagConfig(embedder="hashing", embedding_dim=128))
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == 128


def test_build_embedder_unknown_name_raises():
    with pytest.raises(ValueError):
        build_embedder(RagConfig(embedder="not-a-real-embedder"))
