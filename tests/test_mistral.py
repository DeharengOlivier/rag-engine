"""Provider contract and failure paths, with no network or credentials."""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import Mock

import pytest

from rag_engine.config import RagConfig
from rag_engine.ingestion import Chunk
from rag_engine.llm import MistralLLM, build_llm
from rag_engine.pipeline import RagPipeline
from rag_engine.vectorstore import SearchResult


def response(content="Thursday [1]", choices=True):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        if choices
        else []
    )


@pytest.fixture
def sdk(monkeypatch):
    class SDKError(Exception):
        def __init__(self, status, headers=None):
            self.status_code = status
            self.raw_response = types.SimpleNamespace(headers=headers or {})
            super().__init__("private echoed question and key")

    client = Mock()
    client.chat.complete.return_value = response()
    factory = Mock(return_value=client)
    module = types.ModuleType("mistralai.client")
    module.Mistral = factory
    errors = types.ModuleType("mistralai.client.errors")
    errors.MistralError = SDKError
    monkeypatch.setitem(sys.modules, "mistralai.client", module)
    monkeypatch.setitem(sys.modules, "mistralai.client.errors", errors)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-only-key")
    sleep = Mock()
    monkeypatch.setattr("rag_engine.llm.time.sleep", sleep)
    return types.SimpleNamespace(
        client=client, factory=factory, error=SDKError, sleep=sleep
    )


def test_native_request_and_safe_logs(sdk, caplog):
    llm = build_llm(RagConfig(llm_provider=" MISTRAL ", llm_timeout_seconds=1.25))
    contexts = [
        SearchResult(Chunk("Private recycling details", "/private/recycling.md", 0), 0.9)
    ]
    with caplog.at_level(logging.INFO, logger="rag_engine.llm"):
        assert llm.generate("private question", contexts) == "Thursday [1]"
        llm.generate("second", contexts)
    sdk.factory.assert_called_once_with(
        api_key="test-only-key", timeout_ms=1250, retry_config=None
    )
    request = sdk.client.chat.complete.call_args.kwargs
    assert request["model"] == "mistral-small-latest"
    assert request["max_tokens"] == 1024
    assert "ONLY the provided" in request["messages"][0]["content"]
    assert "[1] (source: recycling.md)" in request["messages"][1]["content"]
    assert "/private/" not in request["messages"][1]["content"]
    for private in ["test-only-key", "Private recycling", "private question", "Thursday"]:
        assert private not in caplog.text
    assert "provider=mistral" in caplog.text


def test_lazy_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "mistralai.client", None)
    llm = MistralLLM()
    with pytest.raises(ImportError, match="pip install"):
        llm.generate("q", [])


def test_missing_key(sdk, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY")
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        MistralLLM().generate("q", [])
    sdk.factory.assert_not_called()


@pytest.mark.parametrize(
    "content", [None, "  ", [], [types.SimpleNamespace(type="thinking", text="secret")]]
)
def test_empty_or_nontext_answers_fail(sdk, content):
    sdk.client.chat.complete.return_value = response(content)
    with pytest.raises(RuntimeError, match="no text"):
        MistralLLM().generate("q", [])


def test_no_choices_fail(sdk):
    sdk.client.chat.complete.return_value = response(choices=False)
    with pytest.raises(RuntimeError, match="no answer choices"):
        MistralLLM().generate("q", [])


def test_text_chunks_exclude_thinking(sdk):
    sdk.client.chat.complete.return_value = response(
        [
            types.SimpleNamespace(type="thinking", text="secret"),
            types.SimpleNamespace(type="text", text=" answer [1] "),
        ]
    )
    assert MistralLLM().generate("q", []) == "answer [1]"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_retries_are_bounded(sdk, status):
    sdk.client.chat.complete.side_effect = sdk.error(status)
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        MistralLLM(max_retries=2).generate("q", [])
    assert "private" not in str(error.value)
    assert error.value.__suppress_context__
    assert sdk.client.chat.complete.call_count == 3
    assert sdk.sleep.call_count == 2
    assert 1 <= sdk.sleep.call_args_list[0].args[0] <= 1.25
    assert 2 <= sdk.sleep.call_args_list[1].args[0] <= 2.25


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_errors_are_not_retried(sdk, status):
    sdk.client.chat.complete.side_effect = sdk.error(status)
    with pytest.raises(RuntimeError):
        MistralLLM().generate("q", [])
    assert sdk.client.chat.complete.call_count == 1
    sdk.sleep.assert_not_called()


@pytest.mark.parametrize("header", ["bad", "nan", "inf", "-1", "31"])
def test_unbounded_or_invalid_retry_after_is_not_retried(sdk, header):
    sdk.client.chat.complete.side_effect = sdk.error(429, {"retry-after": header})
    with pytest.raises(RuntimeError):
        MistralLLM().generate("q", [])
    sdk.sleep.assert_not_called()


def test_retry_after_and_recovery(sdk):
    sdk.client.chat.complete.side_effect = [
        sdk.error(429, {"retry-after": "3"}),
        response(),
    ]
    assert MistralLLM().generate("q", []) == "Thursday [1]"
    sdk.sleep.assert_called_once_with(3.0)


def test_zero_retries(sdk):
    sdk.client.chat.complete.side_effect = sdk.error(503)
    with pytest.raises(RuntimeError):
        MistralLLM(max_retries=0).generate("q", [])
    assert sdk.client.chat.complete.call_count == 1


def test_transport_failures_propagate_without_retry(sdk):
    sdk.client.chat.complete.side_effect = TimeoutError("timeout")
    with pytest.raises(TimeoutError):
        MistralLLM().generate("q", [])
    assert sdk.client.chat.complete.call_count == 1


def test_environment_model_selection(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "mistral")
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    assert RagConfig.from_env().llm_model == "mistral-small-latest"
    monkeypatch.setenv("RAG_LLM_MODEL", "mistral-large-latest")
    assert RagConfig.from_env().llm_model == "mistral-large-latest"
    assert RagConfig(llm_provider="openai").llm_model == "gpt-4o-mini"


def test_pipeline_preserves_citations_and_refuses_before_api(sdk, tmp_path):
    from pathlib import Path

    pipeline = RagPipeline(
        RagConfig(llm_provider="mistral", index_dir=tmp_path / "index")
    )
    pipeline.ingest(Path(__file__).resolve().parent.parent / "data/sample")
    result = pipeline.answer("When is recycling collected in Maplewood?")
    assert not result.refused
    assert any(c.source == "recycling.md" for c in result.citations)
    sdk.client.chat.complete.reset_mock()
    pipeline.config.similarity_threshold = 0.99
    assert pipeline.answer("helium boiling point?").refused
    sdk.client.chat.complete.assert_not_called()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_invalid_timeout_rejected_before_client(timeout):
    with pytest.raises(ValueError, match="llm_timeout_seconds"):
        MistralLLM(timeout_seconds=timeout)
