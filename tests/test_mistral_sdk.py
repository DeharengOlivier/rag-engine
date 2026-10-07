"""Real optional SDK serialization and error contract, over an in-memory transport."""

import json

import pytest

from rag_engine.llm import MistralLLM


def test_real_sdk_request_response_and_no_hidden_retries(monkeypatch):
    sdk = pytest.importorskip("mistralai.client")
    httpx = pytest.importorskip("httpx")
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "synthetic",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "mistral-small-latest",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Thursday [1]"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )
        return httpx.Response(503, json={"message": "private echo"})

    real_factory = sdk.Mistral
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        monkeypatch.setattr(
            sdk, "Mistral", lambda **kwargs: real_factory(client=transport, **kwargs)
        )
        monkeypatch.setenv("MISTRAL_API_KEY", "synthetic-test-key")
        llm = MistralLLM(timeout_seconds=1.25, max_retries=0)
        assert llm.generate("synthetic question", []) == "Thursday [1]"
        with pytest.raises(RuntimeError, match="HTTP 503"):
            llm.generate("synthetic question", [])
    assert len(requests) == 2
    assert str(requests[0].url) == "https://api.mistral.ai/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "mistral-small-latest"
    assert payload["max_tokens"] == 1024
    assert requests[0].extensions["timeout"]["read"] == 1.25
