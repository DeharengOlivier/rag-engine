"""Tests for configuration validation at the boundary.

Environment variables are an input boundary. A typo in one of them must fail
immediately, with a message naming the variable, rather than silently producing
a config that misbehaves much later inside the pipeline.
"""

from __future__ import annotations

import pytest

from rag_engine.config import RagConfig


# --- enum-like fields -------------------------------------------------------


def test_unknown_embedder_is_rejected():
    with pytest.raises(ValueError, match="embedder"):
        RagConfig(embedder="not-a-real-embedder")


def test_unknown_llm_provider_is_rejected():
    with pytest.raises(ValueError, match="llm_provider"):
        RagConfig(llm_provider="nope")


def test_unknown_anonymizer_is_rejected():
    with pytest.raises(ValueError, match="anonymizer"):
        RagConfig(anonymizer="whatever")


def test_rejection_message_lists_the_accepted_values():
    with pytest.raises(ValueError) as exc:
        RagConfig(llm_provider="nope")
    message = str(exc.value)
    assert "extractive" in message
    assert "anthropic" in message
    assert "openai" in message


def test_enum_values_are_normalized_to_their_canonical_spelling():
    config = RagConfig(embedder="  Sentence_Transformers ", llm_provider="ANTHROPIC")
    assert config.embedder == "sentence-transformers"
    assert config.llm_provider == "anthropic"


def test_known_aliases_are_resolved():
    assert RagConfig(embedder="st").embedder == "sentence-transformers"
    assert RagConfig(anonymizer="off").anonymizer == "none"


# --- numeric fields ---------------------------------------------------------


@pytest.mark.parametrize("top_k", [0, -1, -5])
def test_non_positive_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        RagConfig(top_k=top_k)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        RagConfig(chunk_size=chunk_size)


def test_negative_chunk_overlap_is_rejected():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RagConfig(chunk_overlap=-1)


@pytest.mark.parametrize("overlap", [600, 700])
def test_chunk_overlap_at_or_above_chunk_size_is_rejected(overlap):
    # Equal or larger would make the sliding window fail to advance.
    with pytest.raises(ValueError, match="chunk_overlap"):
        RagConfig(chunk_size=600, chunk_overlap=overlap)


@pytest.mark.parametrize("dim", [0, -256])
def test_non_positive_embedding_dim_is_rejected(dim):
    with pytest.raises(ValueError, match="embedding_dim"):
        RagConfig(embedding_dim=dim)


@pytest.mark.parametrize("threshold", [1.01, 5.0, -1.01])
def test_similarity_threshold_outside_cosine_range_is_rejected(threshold):
    # Cosine similarity lives in [-1, 1]; a threshold outside it either never
    # matches (the engine refuses every question) or never filters anything.
    with pytest.raises(ValueError, match="similarity_threshold"):
        RagConfig(similarity_threshold=threshold)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_anonymize_threshold_outside_unit_range_is_rejected(threshold):
    with pytest.raises(ValueError, match="anonymize_threshold"):
        RagConfig(anonymize_threshold=threshold)


# --- environment parsing ----------------------------------------------------


def test_unparseable_int_names_the_variable(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "not-an-int")
    with pytest.raises(ValueError, match="RAG_TOP_K"):
        RagConfig.from_env()


def test_unparseable_float_names_the_variable(monkeypatch):
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "high")
    with pytest.raises(ValueError, match="RAG_SIMILARITY_THRESHOLD"):
        RagConfig.from_env()


def test_out_of_range_env_value_names_the_field(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "-5")
    with pytest.raises(ValueError, match="top_k"):
        RagConfig.from_env()


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_env_value_falls_back_to_the_default(monkeypatch, blank):
    # Unset and blank are the same intent: "I did not configure this".
    monkeypatch.setenv("RAG_TOP_K", blank)
    assert RagConfig.from_env().top_k == 4


# --- the valid path still works --------------------------------------------


def test_valid_configuration_is_accepted():
    config = RagConfig(
        embedder="sentence-transformers",
        llm_provider="anthropic",
        anonymizer="presidio",
        top_k=10,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_dim=384,
        similarity_threshold=0.4,
        anonymize_threshold=0.8,
    )
    assert config.top_k == 10
    assert config.chunk_overlap == 200


def test_a_string_index_dir_is_stored_as_a_path():
    from pathlib import Path

    assert RagConfig(index_dir="/tmp/somewhere").index_dir == Path("/tmp/somewhere")


def test_defaults_are_valid():
    # The defaults must themselves satisfy every rule above.
    assert RagConfig().top_k == 4
