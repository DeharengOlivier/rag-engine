"""End-to-end pipeline tests, fully offline (hashing embedder + extractive LLM)."""

from __future__ import annotations

from pathlib import Path

from rag_engine.config import RagConfig
from rag_engine.pipeline import RagPipeline

# Path to the bundled sample corpus (repo_root/data/sample).
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def _pipeline(tmp_path) -> RagPipeline:
    # Isolate the index directory per test so runs don't interfere.
    config = RagConfig(
        embedder="hashing",
        llm_provider="extractive",
        index_dir=tmp_path / "index",
        top_k=4,
    )
    return RagPipeline(config)


def test_ingest_then_answer_in_corpus_question(tmp_path):
    pipeline = _pipeline(tmp_path)
    n = pipeline.ingest(SAMPLE_DIR)
    assert n > 0, "sample corpus should produce chunks"

    result = pipeline.answer("When is recycling collected in Maplewood?")

    assert result.refused is False
    assert result.answer.strip()
    assert result.citations, "a grounded answer must carry citations"
    # The recycling document should be among the retrieved sources.
    sources = {c.source for c in result.citations}
    assert "recycling.md" in sources


def test_answer_refuses_out_of_corpus_question(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(SAMPLE_DIR)

    # Use a high threshold so an unrelated question cannot clear the grounding
    # gate, making the refusal behavior deterministic offline.
    pipeline.config.similarity_threshold = 0.95
    result = pipeline.answer("What is the boiling point of liquid helium?")

    assert result.refused is True
    assert not result.citations
    assert "enough context" in result.answer.lower()


def test_index_persists_across_pipeline_instances(tmp_path):
    # First pipeline ingests and saves the index.
    first = _pipeline(tmp_path)
    first.ingest(SAMPLE_DIR)

    # A fresh pipeline with the same index_dir should load it and answer.
    second = _pipeline(tmp_path)
    result = second.answer("What are the library opening hours?")
    assert result.used_chunks, "second pipeline should load the persisted index"
