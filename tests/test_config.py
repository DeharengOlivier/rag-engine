"""Tests for RagConfig defaults and environment-variable parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_engine.config import DEFAULT_INDEX_DIR, RagConfig


def test_defaults_are_offline_first():
    config = RagConfig()
    assert config.embedder == "hashing"
    assert config.llm_provider == "extractive"
    assert config.embedding_dim == 256
    assert config.top_k == 4
    assert config.index_dir == Path(DEFAULT_INDEX_DIR)


def test_from_env_uses_defaults_when_unset(monkeypatch):
    for var in [
        "RAG_EMBEDDER",
        "RAG_EMBEDDING_MODEL",
        "RAG_EMBEDDING_DIM",
        "RAG_LLM_PROVIDER",
        "RAG_LLM_MODEL",
        "RAG_TOP_K",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_SIMILARITY_THRESHOLD",
        "RAG_INDEX_DIR",
    ]:
        monkeypatch.delenv(var, raising=False)

    config = RagConfig.from_env()
    assert config.embedder == "hashing"
    assert config.llm_provider == "extractive"
    assert config.embedding_dim == 256
    assert config.similarity_threshold == 0.15


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDER", "sentence-transformers")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "512")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "800")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "120")
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.42")
    monkeypatch.setenv("RAG_INDEX_DIR", "/tmp/custom_index")

    config = RagConfig.from_env()
    assert config.embedder == "sentence-transformers"
    assert config.llm_provider == "openai"
    assert config.embedding_dim == 512
    assert config.top_k == 7
    assert config.chunk_size == 800
    assert config.chunk_overlap == 120
    assert config.similarity_threshold == 0.42
    assert config.index_dir == Path("/tmp/custom_index")


# Specification change: an unparseable value used to fall back to the default.
# It now raises. A typo in a deployment config must fail at startup rather than
# turn into a pipeline that quietly behaves differently from what was asked.
def test_from_env_rejects_an_unparseable_int(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_DIM", "not-an-int")
    with pytest.raises(ValueError, match="RAG_EMBEDDING_DIM"):
        RagConfig.from_env()


def test_from_env_rejects_an_unparseable_float(monkeypatch):
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "high")
    with pytest.raises(ValueError, match="RAG_SIMILARITY_THRESHOLD"):
        RagConfig.from_env()


def test_from_env_treats_a_blank_value_as_unset(monkeypatch):
    # Blank and unset carry the same intent, for strings as well as numbers.
    monkeypatch.setenv("RAG_TOP_K", "")
    monkeypatch.setenv("RAG_EMBEDDER", "   ")
    config = RagConfig.from_env()
    assert config.top_k == 4
    assert config.embedder == "hashing"


def test_from_env_does_not_read_secrets(monkeypatch):
    # API keys must never be read into the config object.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    config = RagConfig.from_env()
    for value in vars(config).values():
        assert "sk-should-not-be-used" != value
