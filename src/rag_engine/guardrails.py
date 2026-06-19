"""Grounding guardrails and citations.

A RAG system is only trustworthy if it answers from retrieved evidence and
refuses when that evidence is missing. This module enforces two rules:

1. **Grounding gate**: if no retrieved chunk clears a similarity threshold, the
   engine must refuse rather than let the LLM answer from parametric memory.
2. **Citations**: when it does answer, every supporting chunk is attached as a
   citation so the answer is auditable.

Keeping this logic separate from generation means the same guardrails apply
regardless of which LLM provider produced the text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rag_engine.vectorstore import SearchResult

# Message returned whenever the grounding gate is not cleared.
REFUSAL_MESSAGE = (
    "I don't have enough context in the provided documents to answer that "
    "question confidently."
)


@dataclass
class Citation:
    """A single source reference attached to an answer.

    Attributes:
        index: 1-based citation number, matching ``[n]`` markers in answers.
        source: Path of the document the cited chunk came from.
        score: Similarity score of the cited chunk against the query.
        snippet: A short preview of the cited chunk's text.
    """

    index: int
    source: str
    score: float
    snippet: str


def passes_grounding(contexts: list[SearchResult], threshold: float) -> bool:
    """Return True if at least one chunk's score clears ``threshold``.

    Results are assumed sorted by descending score, so we only need to check the
    top one, but we scan defensively in case an unsorted list is passed.
    """
    return any(result.score >= threshold for result in contexts)


def build_citations(
    contexts: list[SearchResult],
    threshold: float,
    *,
    snippet_chars: int = 160,
) -> list[Citation]:
    """Build citations for every context chunk that clears ``threshold``.

    Args:
        contexts: Retrieved chunks with scores (sorted by descending score).
        threshold: Minimum similarity for a chunk to be cited.
        snippet_chars: Maximum characters of chunk text to include as a preview.

    Returns:
        One :class:`Citation` per qualifying chunk, numbered from 1.
    """
    citations: list[Citation] = []
    for i, result in enumerate(contexts, start=1):
        if result.score < threshold:
            continue
        text = result.chunk.text.strip().replace("\n", " ")
        snippet = text[:snippet_chars] + ("..." if len(text) > snippet_chars else "")
        citations.append(
            Citation(
                index=i,
                source=os.path.basename(result.chunk.source),
                score=round(result.score, 4),
                snippet=snippet,
            )
        )
    return citations
