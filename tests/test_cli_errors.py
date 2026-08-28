"""Tests for how the CLI fails.

A command-line tool turns an exception into a message and an exit code. A
traceback is a bug report about the tool, and it tells a user who mistyped an
environment variable nothing they can act on.

Exit codes: 0 success, 1 a runtime failure, 2 a bad configuration (the code
argparse already uses for a bad invocation).
"""

from __future__ import annotations

import json
import logging

import pytest

from rag_engine.cli import main
from rag_engine.observability import LOGGER_NAME
from rag_engine.vectorstore import _META_FILE


@pytest.fixture(autouse=True)
def restore_package_logger():
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers = handlers
    logger.setLevel(level)


def _offline_env(monkeypatch, index_dir):
    monkeypatch.setenv("RAG_EMBEDDER", "hashing")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))


def _write_corpus(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "recycling.md").write_text(
        "Recycling is collected every Thursday.", encoding="utf-8"
    )
    return folder


# --- a bad configuration --------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RAG_TOP_K", "oops"),
        ("RAG_TOP_K", "-5"),
        ("RAG_EMBEDDER", "not-a-real-embedder"),
        ("RAG_SIMILARITY_THRESHOLD", "5.0"),
        ("RAG_LOG_LEVEL", "chatty"),
    ],
)
def test_a_bad_setting_is_reported_not_raised(
    variable, value, monkeypatch, tmp_path, capsys
):
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.setenv(variable, value)

    assert main(["query", "anything"]) == 2

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Configuration error" in captured.err
    # The message must name what to fix.
    assert variable in captured.err or variable.removeprefix("RAG_").lower() in (
        captured.err.lower()
    )


def test_a_bad_setting_stops_ingest_too(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "9999")
    folder = _write_corpus(tmp_path)

    assert main(["ingest", str(folder)]) == 2
    assert "Configuration error" in capsys.readouterr().err


# --- a runtime failure ------------------------------------------------------


def test_ingesting_a_missing_folder_is_reported(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")

    assert main(["ingest", str(tmp_path / "nowhere")]) == 1

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "not found" in captured.err


def test_ingesting_a_file_instead_of_a_folder_is_reported(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    target = tmp_path / "a-file.md"
    target.write_text("hello", encoding="utf-8")

    assert main(["ingest", str(target)]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_querying_a_torn_index_is_reported(monkeypatch, tmp_path, capsys):
    index_dir = tmp_path / "index"
    _offline_env(monkeypatch, index_dir)
    main(["ingest", str(_write_corpus(tmp_path))])
    capsys.readouterr()

    meta_path = index_dir / _META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["chunks"] = []
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert main(["query", "anything"]) == 1

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "ingestion" in captured.err  # tells the user how to recover


def test_evaluating_a_missing_file_is_reported(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    main(["ingest", str(_write_corpus(tmp_path))])
    capsys.readouterr()

    assert main(["eval", str(tmp_path / "no-such-eval.json")]) == 1
    assert "Traceback" not in capsys.readouterr().err


# --- success is still success ----------------------------------------------


def test_a_valid_run_still_returns_zero(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    assert main(["ingest", str(_write_corpus(tmp_path))]) == 0
    assert main(["query", "When is recycling collected?"]) == 0
    assert capsys.readouterr().err == ""
