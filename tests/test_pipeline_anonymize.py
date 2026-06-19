"""Integration test: PII must be redacted before it reaches the index on disk.

This is the property that matters for privacy. The vector store persists each
chunk's text to a JSON metadata sidecar, so if anonymization works, no raw PII
can be found anywhere in the saved index.
"""

from __future__ import annotations

import json

from rag_engine.config import RagConfig
from rag_engine.pipeline import RagPipeline


def _write_doc_with_pii(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "client.md").write_text(
        "Client record.\n\n"
        "Contact email: alice.martin@example.com\n"
        "Phone: +33 6 12 34 56 78\n"
        "Card on file: 4111 1111 1111 1111\n"
        "The account was opened last spring and is in good standing.\n",
        encoding="utf-8",
    )


def test_ingest_with_regex_anonymizer_keeps_pii_out_of_the_index(tmp_path):
    docs = tmp_path / "docs"
    _write_doc_with_pii(docs)

    config = RagConfig(
        anonymizer="regex",
        index_dir=tmp_path / "index",
    )
    pipeline = RagPipeline(config)
    count = pipeline.ingest(docs)

    assert count > 0

    # The pipeline reports what it removed.
    assert pipeline.last_pii_report.get("EMAIL_ADDRESS", 0) >= 1
    assert pipeline.last_pii_report.get("PHONE_NUMBER", 0) >= 1
    assert pipeline.last_pii_report.get("CREDIT_CARD", 0) >= 1

    # The persisted index (and its metadata sidecar) must contain no raw PII.
    meta_text = (tmp_path / "index" / "meta.json").read_text(encoding="utf-8")
    assert "alice.martin@example.com" not in meta_text
    assert "4111 1111 1111 1111" not in meta_text
    assert "12 34 56 78" not in meta_text
    # The typed placeholders should be present instead.
    assert "<EMAIL_ADDRESS>" in meta_text
    assert "<CREDIT_CARD>" in meta_text

    # Non-PII context is preserved, so the documents stay retrievable.
    assert "good standing" in meta_text


def test_ingest_without_anonymizer_leaves_text_unchanged(tmp_path):
    docs = tmp_path / "docs"
    _write_doc_with_pii(docs)

    config = RagConfig(anonymizer="none", index_dir=tmp_path / "index")
    pipeline = RagPipeline(config)
    pipeline.ingest(docs)

    assert pipeline.last_pii_report == {}
    meta = json.loads((tmp_path / "index" / "meta.json").read_text(encoding="utf-8"))
    all_text = " ".join(c["text"] for c in meta["chunks"])
    # With anonymization off, the original PII is still there (the default
    # behaviour, unchanged for callers who do not opt in).
    assert "alice.martin@example.com" in all_text


def test_answer_still_works_after_anonymized_ingest(tmp_path):
    docs = tmp_path / "docs"
    _write_doc_with_pii(docs)

    config = RagConfig(anonymizer="regex", index_dir=tmp_path / "index")
    pipeline = RagPipeline(config)
    pipeline.ingest(docs)

    # A question about the non-PII content should still retrieve and answer.
    result = pipeline.answer("What is the standing of the account?")
    assert not result.refused
    assert result.citations
