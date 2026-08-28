"""Answer generators (LLM providers) behind a common interface.

Three providers are offered:

- :class:`ExtractiveLLM` (default): a no-dependency, offline generator. It does
  not call any model; it stitches the retrieved context into a readable answer.
  This guarantees the whole engine runs with no API key and no network, which is
  ideal for tests, demos, and air-gapped use.

- :class:`AnthropicLLM` (optional): calls the Anthropic Messages API. The
  ``anthropic`` package and ``ANTHROPIC_API_KEY`` are read lazily, at call time.

- :class:`OpenAILLM` (optional): calls the OpenAI Chat Completions API. The
  ``openai`` package and ``OPENAI_API_KEY`` are read lazily, at call time.

Both API-backed providers bound every outbound call: a timeout and a finite
number of retries, taken from the configuration. The SDKs handle the exponential
backoff and jitter between attempts, and the client is built once per provider
instance so the connection pool is reused.

All providers receive the question plus the retrieved context blocks and return
a single answer string. Grounding/citation enforcement lives in
:mod:`rag_engine.guardrails`, not here, so providers stay focused on generation.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from rag_engine.config import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    RagConfig,
)
from rag_engine.vectorstore import SearchResult

# Shared instruction given to real LLM providers: answer only from context and
# refuse otherwise. This keeps the API-backed paths grounded too.
_SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the provided "
    "context. If the context does not contain enough information to answer, say "
    "you do not have enough context. Cite sources by their bracketed numbers."
)


@runtime_checkable
class LLM(Protocol):
    """Common interface for answer generators."""

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        """Produce an answer to ``question`` grounded in ``contexts``."""
        ...


def _format_context_block(contexts: list[SearchResult]) -> str:
    """Render retrieved chunks as a numbered context block for a prompt."""
    lines = []
    for i, result in enumerate(contexts, start=1):
        source = os.path.basename(result.chunk.source)
        lines.append(f"[{i}] (source: {source})\n{result.chunk.text}")
    return "\n\n".join(lines)


class ExtractiveLLM:
    """Offline generator that stitches retrieved context into an answer.

    It performs no model inference. The answer is the highest-scoring chunks
    presented as supporting passages, prefixed by a short lead-in. This is
    intentionally simple and transparent: every sentence in the answer comes
    verbatim from the corpus, which makes grounding trivially verifiable.
    """

    # How many top chunks to surface in the stitched answer.
    max_passages: int = 3

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        if not contexts:
            return "I don't have enough context to answer that."
        passages = contexts[: self.max_passages]
        parts = [
            "Based on the retrieved documents, here is what I found:",
            "",
        ]
        for i, result in enumerate(passages, start=1):
            source = os.path.basename(result.chunk.source)
            parts.append(f"[{i}] (from {source}) {result.chunk.text}")
        return "\n".join(parts)


class AnthropicLLM:
    """Answer generator backed by the Anthropic Messages API (lazy import)."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Return the configured client, building it on first use.

        Raises:
            ImportError: The optional ``anthropic`` package is not installed.
            RuntimeError: ``ANTHROPIC_API_KEY`` is not set.
        """
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise ImportError(
                "The 'anthropic' package is required for the anthropic provider. "
                "Install it with 'pip install \"rag-engine[llm]\"', or set "
                "RAG_LLM_PROVIDER=extractive to run offline."
            ) from exc

        # Read the key at call time only; never at import time.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it, or use "
                "RAG_LLM_PROVIDER=extractive to run offline."
            )

        self._client = anthropic.Anthropic(
            timeout=self._timeout_seconds, max_retries=self._max_retries
        )
        return self._client

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        client = self._get_client()
        context_block = _format_context_block(contexts)
        user_content = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above and cite sources by their "
            "bracketed numbers."
        )
        # Adaptive thinking is the recommended setting on this model family.
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": user_content}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


class OpenAILLM:
    """Answer generator backed by the OpenAI Chat Completions API (lazy import)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Return the configured client, building it on first use.

        Raises:
            ImportError: The optional ``openai`` package is not installed.
            RuntimeError: ``OPENAI_API_KEY`` is not set.
        """
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - needs the optional extra
            raise ImportError(
                "The 'openai' package is required for the openai provider. "
                "Install it with 'pip install \"rag-engine[llm]\"', or set "
                "RAG_LLM_PROVIDER=extractive to run offline."
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it, or use "
                "RAG_LLM_PROVIDER=extractive to run offline."
            )

        self._client = openai.OpenAI(
            timeout=self._timeout_seconds, max_retries=self._max_retries
        )
        return self._client

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        client = self._get_client()
        context_block = _format_context_block(contexts)
        user_content = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above and cite sources by their "
            "bracketed numbers."
        )
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def build_llm(config: RagConfig) -> LLM:
    """Instantiate the LLM provider selected in ``config``.

    Raises:
        ValueError: If ``config.llm_provider`` is not a recognized value.
    """
    # config.llm_provider is already validated and canonical (see RagConfig), so
    # the comparisons are exact and the final raise only fires if a value is added
    # to LLM_PROVIDERS without a branch here.
    if config.llm_provider == "extractive":
        return ExtractiveLLM()
    if config.llm_provider == "anthropic":
        return AnthropicLLM(
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
            max_retries=config.llm_max_retries,
        )
    if config.llm_provider == "openai":
        return OpenAILLM(
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
            max_retries=config.llm_max_retries,
        )
    raise ValueError(
        f"Unknown LLM provider '{config.llm_provider}'. "
        "Expected 'extractive', 'anthropic' or 'openai'."
    )
