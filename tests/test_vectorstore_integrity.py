"""Regression battery: a half-written index must never be answered from.

The defect: save() wrote vectors.npy and meta.json one after the other with no
staging step, and load() trusted whatever it found. An interrupted save left the
two files describing different indexes, load() accepted the pair, and the first
query died with "IndexError: list index out of range" far away from the cause.

These tests cover the reported case, the invariant behind it, the boundaries
around it, and the neighbouring call site (the pipeline).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rag_engine.config import RagConfig
from rag_engine.ingestion import Chunk
from rag_engine.pipeline import RagPipeline
from rag_engine.vectorstore import _META_FILE, _VECTORS_FILE, VectorStore


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, source="doc.md", chunk_index=idx)


def _store(rows: int = 4, dim: int = 8) -> VectorStore:
    store = VectorStore(dim=dim)
    vectors = np.eye(rows, dim, dtype=np.float32)
    store.add(vectors, [_chunk(f"c{i}", i) for i in range(rows)])
    return store


def _rewrite_meta(directory, mutate) -> None:
    """Apply ``mutate`` to the persisted metadata, simulating a torn write."""
    path = directory / _META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    mutate(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")


# --- the exact reported case ------------------------------------------------


def test_load_rejects_fewer_chunks_than_vectors(tmp_path):
    _store(rows=4).save(tmp_path)
    _rewrite_meta(tmp_path, lambda m: m.__setitem__("chunks", m["chunks"][:1]))

    with pytest.raises(ValueError, match="1 chunk"):
        VectorStore.load(tmp_path)


# --- the invariant ----------------------------------------------------------


def test_load_rejects_more_chunks_than_vectors(tmp_path):
    _store(rows=2).save(tmp_path)
    _rewrite_meta(tmp_path, lambda m: m["chunks"].append(dict(m["chunks"][0])))

    with pytest.raises(ValueError, match="chunk"):
        VectorStore.load(tmp_path)


def test_load_rejects_vectors_of_the_wrong_width(tmp_path):
    _store(rows=3, dim=8).save(tmp_path)
    _rewrite_meta(tmp_path, lambda m: m.__setitem__("dim", 16))

    with pytest.raises(ValueError, match="dim"):
        VectorStore.load(tmp_path)


def test_the_error_names_the_directory_so_it_can_be_rebuilt(tmp_path):
    _store(rows=4).save(tmp_path)
    _rewrite_meta(tmp_path, lambda m: m.__setitem__("chunks", []))

    with pytest.raises(ValueError, match=str(tmp_path.name)):
        VectorStore.load(tmp_path)


# --- the boundaries ---------------------------------------------------------


def test_an_empty_index_is_valid(tmp_path):
    VectorStore(dim=8).save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 0
    assert loaded.dim == 8


def test_load_rejects_a_non_positive_dim(tmp_path):
    _store().save(tmp_path)
    _rewrite_meta(tmp_path, lambda m: m.__setitem__("dim", 0))

    with pytest.raises(ValueError, match="dim"):
        VectorStore.load(tmp_path)


def test_load_reports_unreadable_metadata_clearly(tmp_path):
    _store().save(tmp_path)
    (tmp_path / _META_FILE).write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=_META_FILE):
        VectorStore.load(tmp_path)


def test_load_still_reports_a_missing_index_as_such(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nothing-here")


# --- an interrupted save leaves the previous index intact -------------------


def test_a_failed_save_does_not_destroy_the_previous_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    _store(rows=4).save(index_dir)

    # Fail at the moment the new index would replace the old one.
    real_replace = os.replace

    def exploding_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", exploding_replace)
    with pytest.raises(OSError):
        _store(rows=2).save(index_dir)
    monkeypatch.setattr(os, "replace", real_replace)

    # The index that was there before is still loadable and still complete.
    recovered = VectorStore.load(index_dir)
    assert len(recovered) == 4


def test_a_failed_swap_puts_the_previous_index_back(tmp_path, monkeypatch):
    # The narrow window: the old index has already been moved aside when the
    # swap fails. Without the rollback, the directory would simply be gone.
    index_dir = tmp_path / "index"
    _store(rows=4).save(index_dir)

    real_replace = os.replace
    calls = {"n": 0}

    def failing_second_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_second_replace)
    with pytest.raises(OSError):
        _store(rows=2).save(index_dir)
    monkeypatch.undo()

    recovered = VectorStore.load(index_dir)
    assert len(recovered) == 4


def test_a_failed_save_leaves_no_staging_directory_behind(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    _store(rows=4).save(index_dir)

    monkeypatch.setattr(
        np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        _store(rows=2).save(index_dir)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "index"]
    assert leftovers == []


def test_saving_over_an_existing_index_replaces_it(tmp_path):
    index_dir = tmp_path / "index"
    _store(rows=4).save(index_dir)
    _store(rows=2).save(index_dir)

    loaded = VectorStore.load(index_dir)
    assert len(loaded) == 2
    assert sorted(p.name for p in index_dir.iterdir()) == [_META_FILE, _VECTORS_FILE]


# --- the neighbouring call site ---------------------------------------------


def test_the_pipeline_refuses_a_torn_index_instead_of_crashing(tmp_path):
    config = RagConfig(
        embedder="hashing",
        llm_provider="extractive",
        index_dir=tmp_path / "index",
    )
    pipeline = RagPipeline(config)
    pipeline.ingest(_corpus(tmp_path))

    _rewrite_meta(config.index_dir, lambda m: m.__setitem__("chunks", m["chunks"][:1]))

    # A fresh pipeline must load the index and reject it, with an error naming
    # the problem, rather than answering from a store it cannot trust.
    with pytest.raises(ValueError, match="chunk"):
        RagPipeline(config).answer("anything at all?")


def _corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    for i in range(3):
        (folder / f"doc{i}.md").write_text(
            f"Document {i} about municipal recycling collection schedules. " * 20,
            encoding="utf-8",
        )
    return folder
