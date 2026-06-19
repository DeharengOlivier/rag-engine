"""Tests for the grounding guardrail and citation building."""

from __future__ import annotations

from rag_engine.guardrails import build_citations, passes_grounding
from rag_engine.ingestion import Chunk
from rag_engine.vectorstore import SearchResult


def _result(score: float, source: str = "doc.md") -> SearchResult:
    return SearchResult(
        chunk=Chunk(text="some supporting text", source=source, chunk_index=0),
        score=score,
    )


def test_refuses_when_no_chunk_passes_threshold():
    results = [_result(0.05), _result(0.02)]
    assert passes_grounding(results, threshold=0.15) is False


def test_passes_when_a_chunk_clears_threshold():
    results = [_result(0.40), _result(0.02)]
    assert passes_grounding(results, threshold=0.15) is True


def test_build_citations_attaches_only_qualifying_chunks():
    results = [_result(0.40, "a.md"), _result(0.05, "b.md")]
    citations = build_citations(results, threshold=0.15)

    # Only the chunk above the threshold is cited.
    assert len(citations) == 1
    assert citations[0].source == "a.md"
    assert citations[0].index == 1
    assert citations[0].snippet  # snippet is attached


def test_build_citations_empty_when_nothing_qualifies():
    results = [_result(0.01), _result(0.02)]
    assert build_citations(results, threshold=0.15) == []
