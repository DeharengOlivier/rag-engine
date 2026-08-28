"""Tests for the offline extractive LLM and provider selection.

These stay fully offline: only the extractive provider is exercised end to end.
The API-backed providers are checked only for their lazy, key-free error paths.
"""

from __future__ import annotations

import pytest

from rag_engine.config import RagConfig
from rag_engine.ingestion import Chunk
from rag_engine.llm import (
    AnthropicLLM,
    ExtractiveLLM,
    LLM,
    OpenAILLM,
    build_llm,
)
from rag_engine.vectorstore import SearchResult


def _ctx(text: str, source: str, score: float = 0.9) -> SearchResult:
    return SearchResult(chunk=Chunk(text=text, source=source, chunk_index=0), score=score)


def test_extractive_llm_satisfies_protocol():
    assert isinstance(ExtractiveLLM(), LLM)


def test_extractive_empty_context_returns_fallback():
    out = ExtractiveLLM().generate("q?", [])
    assert "enough context" in out.lower()


def test_extractive_includes_context_text_and_source_basename():
    out = ExtractiveLLM().generate(
        "When is recycling?",
        [_ctx("Recycling is collected Thursday.", "/deep/path/recycling.md")],
    )
    assert "Recycling is collected Thursday." in out
    assert "recycling.md" in out
    assert "/deep/path/" not in out  # only the basename is shown


def test_extractive_limits_to_max_passages():
    contexts = [_ctx(f"passage {i}", f"d{i}.md") for i in range(10)]
    out = ExtractiveLLM().generate("q?", contexts)
    # Only the first max_passages chunks are surfaced.
    assert "passage 0" in out
    assert f"passage {ExtractiveLLM.max_passages - 1}" in out
    assert f"passage {ExtractiveLLM.max_passages}" not in out


def test_extractive_numbers_passages_from_one():
    contexts = [_ctx("first", "a.md"), _ctx("second", "b.md")]
    out = ExtractiveLLM().generate("q?", contexts)
    assert "[1]" in out and "[2]" in out


def test_build_llm_returns_extractive_by_default():
    assert isinstance(build_llm(RagConfig(llm_provider="extractive")), ExtractiveLLM)


def test_build_llm_rejects_a_provider_with_no_branch():
    # See test_build_anonymizer_rejects_a_name_with_no_branch: the config layer
    # rejects unknown providers, so this exercises the exhaustiveness guard.
    config = RagConfig()
    config.llm_provider = "nonsense"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_llm(config)


def test_anthropic_provider_raises_without_key_when_package_present(monkeypatch):
    # If the anthropic package is not installed, an ImportError is the expected
    # offline behavior; if it is installed, a missing key must raise RuntimeError.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pytest.importorskip("anthropic")
    with pytest.raises(RuntimeError):
        AnthropicLLM().generate("q?", [_ctx("ctx", "a.md")])


def test_openai_provider_raises_without_key_when_package_present(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pytest.importorskip("openai")
    with pytest.raises(RuntimeError):
        OpenAILLM().generate("q?", [_ctx("ctx", "a.md")])
