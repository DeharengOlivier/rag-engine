"""Edge-case tests for chunking and ingestion boundaries."""

from __future__ import annotations

import pytest

from rag_engine.ingestion import chunk_text, iter_document_paths, load_and_chunk


def test_chunk_overlap_actually_overlaps_by_construction():
    # With no whitespace, the splitter cannot nudge the cut point, so the step
    # is exactly chunk_size - chunk_overlap and overlap is exact.
    text = "x" * 1000
    size, overlap = 300, 100
    chunks = chunk_text(text, chunk_size=size, chunk_overlap=overlap)

    assert len(chunks) >= 2
    # Each interior chunk should be exactly chunk_size long for space-poor text.
    assert len(chunks[0]) == size
    # The last `overlap` chars of chunk i equal the first `overlap` of chunk i+1.
    assert chunks[0][-overlap:] == chunks[1][:overlap]


def test_chunk_size_equal_to_text_length_is_single_chunk():
    text = "abcde fghij"  # length 11
    assert chunk_text(text, chunk_size=len(text), chunk_overlap=2) == [text]


def test_chunk_zero_overlap_has_no_shared_text():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=250, chunk_overlap=0)
    # No overlap: concatenation reconstructs the original exactly.
    assert "".join(chunks) == text


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0, chunk_overlap=0)


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, chunk_overlap=100)


def test_chunk_negative_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, chunk_overlap=-1)


def test_chunk_whitespace_only_input_is_empty():
    assert chunk_text("   \n\t  ", chunk_size=100, chunk_overlap=10) == []


def test_iter_document_paths_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_document_paths(tmp_path / "does-not-exist"))


def test_iter_document_paths_on_file_raises(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        list(iter_document_paths(f))


def test_iter_document_paths_is_sorted_and_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "c.md").write_text("c", encoding="utf-8")
    (tmp_path / "skip.json").write_text("{}", encoding="utf-8")

    paths = list(iter_document_paths(tmp_path))
    names = [p.name for p in paths]

    # Sorted for determinism, recursive into subfolders, only supported types.
    assert names == sorted(names)
    assert set(names) == {"a.txt", "b.md", "c.md"}


def test_load_and_chunk_empty_folder_returns_empty(tmp_path):
    assert load_and_chunk(tmp_path) == []


def test_load_and_chunk_assigns_sequential_chunk_indices(tmp_path):
    # A single long document should yield chunks indexed 0, 1, 2, ...
    (tmp_path / "doc.md").write_text("alpha " * 500, encoding="utf-8")
    chunks = load_and_chunk(tmp_path, chunk_size=200, chunk_overlap=40)

    assert len(chunks) >= 2
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    assert all(c.source.endswith("doc.md") for c in chunks)
