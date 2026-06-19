"""Tests for the evaluation harness (recall@k and keyword-score math).

Uses a tiny known corpus and the offline pipeline so the metrics are exact and
checkable by hand, with no network or API key.
"""

from __future__ import annotations

import json

from rag_engine.config import RagConfig
from rag_engine.evaluation import EvalCase, evaluate, load_eval_cases
from rag_engine.pipeline import RagPipeline


def _tiny_corpus(tmp_path):
    """A 3-document corpus with clearly separable vocabulary."""
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "recycling.md").write_text(
        "Recycling is collected every Thursday morning in Maplewood.",
        encoding="utf-8",
    )
    (folder / "library.md").write_text(
        "The library opens on Saturday from 10:00 to 16:00.",
        encoding="utf-8",
    )
    (folder / "parks.md").write_text(
        "Hilltop Park has a community garden and a fitness area.",
        encoding="utf-8",
    )
    return folder


def _ingested_pipeline(tmp_path) -> RagPipeline:
    config = RagConfig(
        embedder="hashing",
        llm_provider="extractive",
        index_dir=tmp_path / "index",
        top_k=1,  # top_k=1 makes recall@k math exact and easy to reason about.
        similarity_threshold=0.0,
    )
    pipeline = RagPipeline(config)
    pipeline.ingest(_tiny_corpus(tmp_path))
    return pipeline


def test_empty_cases_returns_zeroed_report(tmp_path):
    report = evaluate(_ingested_pipeline(tmp_path), [])
    assert report.num_cases == 0
    assert report.recall_at_k == 0.0
    assert report.keyword_score == 0.0


def test_perfect_recall_on_separable_corpus(tmp_path):
    pipeline = _ingested_pipeline(tmp_path)
    cases = [
        EvalCase("When is recycling collected?", "recycling.md", ["Thursday"]),
        EvalCase("What are the library opening hours?", "library.md", ["10:00"]),
        EvalCase("What does Hilltop Park have?", "parks.md", ["garden"]),
    ]
    report = evaluate(pipeline, cases)
    assert report.num_cases == 3
    # All three expected sources are the top-1 hit -> recall@1 == 1.0.
    assert report.recall_at_k == 1.0
    # Extractive answers quote the chunk verbatim -> all keywords present.
    assert report.keyword_score == 1.0


def test_recall_is_fraction_of_hits(tmp_path):
    pipeline = _ingested_pipeline(tmp_path)
    cases = [
        EvalCase("When is recycling collected?", "recycling.md"),
        # This expected source can never be the top-1 hit for this query.
        EvalCase("When is recycling collected?", "parks.md"),
    ]
    report = evaluate(pipeline, cases)
    # Exactly one of two cases hits -> recall@k == 0.5.
    assert report.recall_at_k == 0.5


def test_keyword_score_is_fraction_found(tmp_path):
    pipeline = _ingested_pipeline(tmp_path)
    # "Thursday" appears in the answer; "Wednesday" does not -> 1 of 2 found.
    cases = [
        EvalCase(
            "When is recycling collected?",
            "recycling.md",
            ["Thursday", "Wednesday"],
        )
    ]
    report = evaluate(pipeline, cases)
    assert report.keyword_score == 0.5


def test_no_expected_source_counts_any_retrieval_as_hit(tmp_path):
    pipeline = _ingested_pipeline(tmp_path)
    cases = [EvalCase("recycling", expected_source=None)]
    report = evaluate(pipeline, cases)
    assert report.recall_at_k == 1.0


def test_refused_case_has_no_retrieval_hit(tmp_path):
    pipeline = _ingested_pipeline(tmp_path)
    pipeline.config.similarity_threshold = 0.99  # force a refusal
    cases = [EvalCase("totally unrelated quantum chromodynamics", "recycling.md")]
    report = evaluate(pipeline, cases)
    assert report.per_case[0]["refused"] is True


def test_load_eval_cases_from_json(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "q1",
                    "expected_source": "a.md",
                    "answer_keywords": ["x", "y"],
                },
                {"question": "q2"},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(path)
    assert len(cases) == 2
    assert cases[0].question == "q1"
    assert cases[0].expected_source == "a.md"
    assert cases[0].answer_keywords == ["x", "y"]
    # Missing optional fields default cleanly.
    assert cases[1].expected_source is None
    assert cases[1].answer_keywords == []


def test_report_format_is_renderable(tmp_path):
    pipeline = _ingested_pipeline(tmp_path)
    report = evaluate(pipeline, [EvalCase("recycling", "recycling.md", ["Thursday"])])
    text = report.format()
    assert "Recall@k" in text
    assert "Keyword score" in text
