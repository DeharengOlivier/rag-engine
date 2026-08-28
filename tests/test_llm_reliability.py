"""Tests for the failure-path settings of the API-backed providers.

Every outbound call must carry a timeout and a bounded number of retries. These
tests inject fake SDK modules, so they assert that invariant offline and without
either optional package installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from rag_engine.config import RagConfig
from rag_engine.ingestion import Chunk
from rag_engine.llm import AnthropicLLM, OpenAILLM, build_llm
from rag_engine.vectorstore import SearchResult


def _ctx(text: str = "ctx", source: str = "a.md") -> SearchResult:
    return SearchResult(chunk=Chunk(text=text, source=source, chunk_index=0), score=0.9)


class _Recorder:
    """Captures how the SDK client was constructed and how often."""

    def __init__(self) -> None:
        self.client_kwargs: list[dict] = []
        self.call_kwargs: list[dict] = []


def _fake_anthropic_module(recorder: _Recorder) -> types.ModuleType:
    class _Messages:
        def create(self, **kwargs):
            recorder.call_kwargs.append(kwargs)
            block = types.SimpleNamespace(type="text", text="an answer")
            return types.SimpleNamespace(content=[block])

    class _Anthropic:
        def __init__(self, **kwargs):
            recorder.client_kwargs.append(kwargs)
            self.messages = _Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = _Anthropic
    return module


def _fake_openai_module(recorder: _Recorder) -> types.ModuleType:
    class _Completions:
        def create(self, **kwargs):
            recorder.call_kwargs.append(kwargs)
            message = types.SimpleNamespace(content="an answer")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class _OpenAI:
        def __init__(self, **kwargs):
            recorder.client_kwargs.append(kwargs)
            self.chat = types.SimpleNamespace(completions=_Completions())

    module = types.ModuleType("openai")
    module.OpenAI = _OpenAI
    return module


@pytest.fixture
def anthropic_recorder(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic_module(recorder))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    return recorder


@pytest.fixture
def openai_recorder(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "openai", _fake_openai_module(recorder))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    return recorder


# --- every outbound call is bounded ----------------------------------------


def test_anthropic_client_is_built_with_a_timeout_and_bounded_retries(
    anthropic_recorder,
):
    AnthropicLLM(timeout_seconds=12.5, max_retries=3).generate("q?", [_ctx()])
    kwargs = anthropic_recorder.client_kwargs[0]
    assert kwargs["timeout"] == 12.5
    assert kwargs["max_retries"] == 3


def test_openai_client_is_built_with_a_timeout_and_bounded_retries(openai_recorder):
    OpenAILLM(timeout_seconds=12.5, max_retries=3).generate("q?", [_ctx()])
    kwargs = openai_recorder.client_kwargs[0]
    assert kwargs["timeout"] == 12.5
    assert kwargs["max_retries"] == 3


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_defaults_are_bounded_too(provider, anthropic_recorder, openai_recorder):
    # A caller that configures nothing must still get a finite timeout: the SDK
    # defaults (600s for anthropic) are far too long to be a useful bound.
    recorder = anthropic_recorder if provider == "anthropic" else openai_recorder
    llm = AnthropicLLM() if provider == "anthropic" else OpenAILLM()
    llm.generate("q?", [_ctx()])
    kwargs = recorder.client_kwargs[0]
    assert 0 < kwargs["timeout"] <= 120
    assert 0 <= kwargs["max_retries"] <= 5


# --- the settings come from the configuration -------------------------------


def test_build_llm_passes_the_configured_bounds(anthropic_recorder):
    config = RagConfig(
        llm_provider="anthropic", llm_timeout_seconds=7.0, llm_max_retries=1
    )
    build_llm(config).generate("q?", [_ctx()])
    kwargs = anthropic_recorder.client_kwargs[0]
    assert kwargs["timeout"] == 7.0
    assert kwargs["max_retries"] == 1


def test_from_env_reads_the_bounds(monkeypatch):
    monkeypatch.setenv("RAG_LLM_TIMEOUT_SECONDS", "9.5")
    monkeypatch.setenv("RAG_LLM_MAX_RETRIES", "4")
    config = RagConfig.from_env()
    assert config.llm_timeout_seconds == 9.5
    assert config.llm_max_retries == 4


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_non_positive_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="llm_timeout_seconds"):
        RagConfig(llm_timeout_seconds=timeout)


def test_negative_retry_count_is_rejected():
    with pytest.raises(ValueError, match="llm_max_retries"):
        RagConfig(llm_max_retries=-1)


def test_unbounded_retry_count_is_rejected():
    # An unbounded retry budget turns one slow provider into an unbounded wait.
    with pytest.raises(ValueError, match="llm_max_retries"):
        RagConfig(llm_max_retries=100)


# --- the client is built once, not per call ---------------------------------


def test_anthropic_client_is_reused_across_calls(anthropic_recorder):
    llm = AnthropicLLM()
    llm.generate("first?", [_ctx()])
    llm.generate("second?", [_ctx()])
    assert len(anthropic_recorder.client_kwargs) == 1
    assert len(anthropic_recorder.call_kwargs) == 2


def test_openai_client_is_reused_across_calls(openai_recorder):
    llm = OpenAILLM()
    llm.generate("first?", [_ctx()])
    llm.generate("second?", [_ctx()])
    assert len(openai_recorder.client_kwargs) == 1
    assert len(openai_recorder.call_kwargs) == 2


# --- the pre-flight checks still come first ---------------------------------


def test_missing_key_is_reported_before_any_client_is_built(
    monkeypatch, anthropic_recorder
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM().generate("q?", [_ctx()])
    assert anthropic_recorder.client_kwargs == []
