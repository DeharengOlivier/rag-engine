"""Edge-case pipeline tests: multi-source citations, empty corpus, fake LLM.

Fully offline. A controlled tiny corpus and an injected fake LLM make the
citation and grounding behavior deterministic and checkable.
"""

from __future__ import annotations

from rag_engine.config import RagConfig
from rag_engine.ingestion import Chunk
from rag_engine.pipeline import RagPipeline
from rag_engine.vectorstore import SearchResult


class _RecordingLLM:
    """A fake LLM that records exactly which contexts the pipeline passed it."""

    def __init__(self) -> None:
        self.seen: list[SearchResult] = []

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        self.seen = contexts
        return "FAKE ANSWER"


def _corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "recycling.md").write_text(
        "Recycling collection happens every Thursday in Maplewood.", encoding="utf-8"
    )
    (folder / "compost.md").write_text(
        "Compost and recycling bins are emptied on Thursday too.", encoding="utf-8"
    )
    (folder / "parks.md").write_text(
        "The riverside park has a duck pond and a playground.", encoding="utf-8"
    )
    return folder


def _config(tmp_path, **over) -> RagConfig:
    base = dict(
        embedder="hashing",
        llm_provider="extractive",
        index_dir=tmp_path / "index",
        top_k=3,
        similarity_threshold=0.0,
    )
    base.update(over)
    return RagConfig(**base)


def test_empty_corpus_ingest_returns_zero_and_answer_refuses(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    pipeline = RagPipeline(_config(tmp_path))
    assert pipeline.ingest(folder) == 0

    result = pipeline.answer("anything at all")
    # Nothing indexed -> no chunk can clear the gate -> refusal, no citations.
    assert result.refused is True
    assert result.citations == []
    assert result.used_chunks == []


def test_multiple_sources_yield_multiple_citations(tmp_path):
    pipeline = RagPipeline(_config(tmp_path, similarity_threshold=0.01))
    pipeline.ingest(_corpus(tmp_path))

    result = pipeline.answer("When is recycling and compost collected on Thursday?")
    assert result.refused is False
    sources = [c.source for c in result.citations]
    # Both Thursday-collection docs should surface and be cited.
    assert "recycling.md" in sources
    assert "compost.md" in sources


def test_citation_indices_are_sequential_when_all_qualify(tmp_path):
    pipeline = RagPipeline(_config(tmp_path, top_k=3, similarity_threshold=0.0))
    pipeline.ingest(_corpus(tmp_path))
    result = pipeline.answer("recycling compost thursday park")
    indices = [c.index for c in result.citations]
    # With threshold 0.0 every retrieved chunk qualifies -> contiguous 1..n.
    assert indices == list(range(1, len(indices) + 1))


def test_pipeline_only_feeds_grounded_chunks_to_llm(tmp_path):
    fake = _RecordingLLM()
    # Threshold above the weakest hit but below the strongest, so the pipeline
    # must drop sub-threshold chunks before calling the LLM.
    config = _config(tmp_path, top_k=3, similarity_threshold=0.05)
    pipeline = RagPipeline(config, llm=fake)
    pipeline.ingest(_corpus(tmp_path))

    result = pipeline.answer("recycling collection thursday")
    assert result.answer == "FAKE ANSWER"
    assert fake.seen, "the LLM should receive at least one grounded chunk"
    assert all(r.score >= config.similarity_threshold for r in fake.seen)


def test_refusal_does_not_call_llm(tmp_path):
    class _ExplodingLLM:
        def generate(self, question, contexts):  # pragma: no cover - must not run
            raise AssertionError("LLM must not be called on a refusal")

    config = _config(tmp_path, similarity_threshold=0.999)
    pipeline = RagPipeline(config, llm=_ExplodingLLM())
    pipeline.ingest(_corpus(tmp_path))
    result = pipeline.answer("recycling")
    assert result.refused is True


def test_answer_to_dict_is_json_serializable(tmp_path):
    import json

    pipeline = RagPipeline(_config(tmp_path))
    pipeline.ingest(_corpus(tmp_path))
    result = pipeline.answer("recycling thursday")
    d = result.to_dict()
    # Round-trips through JSON without error.
    json.loads(json.dumps(d))
    assert "answer" in d and "citations" in d and "used_chunks" in d
