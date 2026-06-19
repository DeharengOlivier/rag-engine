"""Tests for the CLI, fully offline via RAG_* environment configuration."""

from __future__ import annotations

import json

import pytest

from rag_engine.cli import main


def _write_corpus(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "recycling.md").write_text(
        "Recycling is collected every Thursday morning.", encoding="utf-8"
    )
    return folder


def _offline_env(monkeypatch, index_dir):
    monkeypatch.setenv("RAG_EMBEDDER", "hashing")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.0")


def test_ingest_then_query(monkeypatch, tmp_path, capsys):
    folder = _write_corpus(tmp_path)
    _offline_env(monkeypatch, tmp_path / "index")

    assert main(["ingest", str(folder)]) == 0
    assert "Indexed" in capsys.readouterr().out

    assert main(["query", "When is recycling collected?"]) == 0
    out = capsys.readouterr().out
    assert "Thursday" in out
    assert "Sources:" in out


def test_query_json_output(monkeypatch, tmp_path, capsys):
    folder = _write_corpus(tmp_path)
    _offline_env(monkeypatch, tmp_path / "index")
    main(["ingest", str(folder)])
    capsys.readouterr()

    assert main(["query", "recycling collection", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "answer" in payload
    assert "citations" in payload


def test_query_without_index_reports_error(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "missing_index")
    rc = main(["query", "anything"])
    assert rc == 1
    assert "No index found" in capsys.readouterr().err


def test_eval_subcommand(monkeypatch, tmp_path, capsys):
    folder = _write_corpus(tmp_path)
    _offline_env(monkeypatch, tmp_path / "index")
    main(["ingest", str(folder)])
    capsys.readouterr()

    evalfile = tmp_path / "eval.json"
    evalfile.write_text(
        json.dumps(
            [
                {
                    "question": "When is recycling collected?",
                    "expected_source": "recycling.md",
                    "answer_keywords": ["Thursday"],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert main(["eval", str(evalfile)]) == 0
    assert "Evaluation report" in capsys.readouterr().out


def test_no_subcommand_errors(capsys):
    with pytest.raises(SystemExit):
        main([])
