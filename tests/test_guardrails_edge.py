"""Edge-case tests for the grounding gate and citation construction."""

from __future__ import annotations

from rag_engine.guardrails import (
    REFUSAL_MESSAGE,
    build_citations,
    passes_grounding,
)
from rag_engine.ingestion import Chunk
from rag_engine.vectorstore import SearchResult


def _result(score: float, source: str = "doc.md", text: str = "supporting text") -> SearchResult:
    return SearchResult(chunk=Chunk(text=text, source=source, chunk_index=0), score=score)


def test_score_exactly_at_threshold_passes():
    # The gate uses >=, so a score equal to the threshold must clear it.
    results = [_result(0.15)]
    assert passes_grounding(results, threshold=0.15) is True


def test_score_just_below_threshold_refuses():
    results = [_result(0.1499)]
    assert passes_grounding(results, threshold=0.15) is False


def test_empty_contexts_never_pass():
    assert passes_grounding([], threshold=0.0) is False


def test_passes_scans_all_results_not_just_first():
    # An out-of-order list where only a later element clears the threshold.
    results = [_result(0.01), _result(0.05), _result(0.9)]
    assert passes_grounding(results, threshold=0.15) is True


def test_citation_at_exact_threshold_is_included():
    citations = build_citations([_result(0.15, "a.md")], threshold=0.15)
    assert len(citations) == 1


def test_citation_index_reflects_position_in_full_list():
    # A skipped (below-threshold) middle chunk leaves a gap in citation indices,
    # because indices enumerate the full results list, not just qualifiers.
    results = [_result(0.9, "a.md"), _result(0.05, "b.md"), _result(0.5, "c.md")]
    citations = build_citations(results, threshold=0.15)
    assert [(c.index, c.source) for c in citations] == [(1, "a.md"), (3, "c.md")]


def test_citation_snippet_is_truncated_with_ellipsis():
    long_text = "word " * 100
    citations = build_citations([_result(0.9, "a.md", text=long_text)], threshold=0.15, )
    snippet = citations[0].snippet
    assert snippet.endswith("...")
    assert len(snippet) <= 160 + 3


def test_citation_snippet_short_text_has_no_ellipsis():
    citations = build_citations([_result(0.9, "a.md", text="short")], threshold=0.15)
    assert citations[0].snippet == "short"


def test_citation_source_is_basename_only():
    results = [_result(0.9, "/some/deep/path/to/library.md")]
    citations = build_citations(results, threshold=0.15)
    assert citations[0].source == "library.md"


def test_citation_score_is_rounded():
    citations = build_citations([_result(0.123456789, "a.md")], threshold=0.0)
    assert citations[0].score == round(0.123456789, 4)


def test_refusal_message_mentions_lack_of_context():
    assert "context" in REFUSAL_MESSAGE.lower()
