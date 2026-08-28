"""Tests for document loading and chunking."""

from __future__ import annotations

from rag_engine.ingestion import chunk_text, clean_text, load_and_chunk


def test_clean_text_collapses_whitespace():
    raw = "Hello    world\t\t!\n\n\n\nNext   paragraph  "
    cleaned = clean_text(raw)
    assert "    " not in cleaned
    assert "\t" not in cleaned
    # 4 blank lines collapse to at most one blank line (two newlines).
    assert "\n\n\n" not in cleaned


def test_chunk_text_produces_overlapping_chunks():
    # Build text long enough to force multiple chunks.
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)

    assert len(chunks) >= 2, "expected the text to split into multiple chunks"
    assert all(c.strip() for c in chunks), "no chunk should be empty"

    # Consecutive chunks should overlap: the end of one chunk shares some text
    # with the start of the next.
    first_tail = chunks[0][-50:]
    assert any(token in chunks[1] for token in first_tail.split()), (
        "consecutive chunks should overlap"
    )


def test_chunk_text_short_input_is_single_chunk():
    assert chunk_text("short text", chunk_size=600, chunk_overlap=100) == ["short text"]


def test_chunk_text_empty_input():
    assert chunk_text("", chunk_size=600, chunk_overlap=100) == []


def test_load_and_chunk_reads_sample_dir(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\n" + "alpha " * 300, encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta " * 300, encoding="utf-8")
    (tmp_path / "ignored.pdf").write_text("not read", encoding="utf-8")

    chunks = load_and_chunk(tmp_path, chunk_size=200, chunk_overlap=40)

    assert chunks, "should produce chunks from .md and .txt files"
    sources = {chunk.source for chunk in chunks}
    # The .pdf must be ignored; only .md and .txt are supported.
    assert not any(s.endswith(".pdf") for s in sources)
    assert any(s.endswith("a.md") for s in sources)
    assert any(s.endswith("b.txt") for s in sources)
