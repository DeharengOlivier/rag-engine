"""Tests for the CLI paths the happy-path suite never walked.

The anonymize command, the refusal and citation output of query, the PII summary
after an ingest, and the verbosity flag. All offline, driven through main() the
way a user drives the binary.
"""

from __future__ import annotations

import json
import logging

import pytest

from rag_engine.cli import _log_level, main
from rag_engine.observability import LOGGER_NAME


@pytest.fixture(autouse=True)
def restore_package_logger():
    """main() configures logging; keep that out of the other tests."""
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers = handlers
    logger.setLevel(level)


def _offline_env(monkeypatch, index_dir):
    monkeypatch.setenv("RAG_EMBEDDER", "hashing")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "extractive")
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))


def _write_corpus(tmp_path, text: str = "Recycling is collected every Thursday."):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "recycling.md").write_text(text, encoding="utf-8")
    return folder


# --- rag anonymize ----------------------------------------------------------


def test_anonymize_redacts_and_lists_what_it_found(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    assert main(["anonymize", "Write to jane.doe@example.com about it."]) == 0

    out = capsys.readouterr().out
    assert "jane.doe@example.com" not in out.split("Detected:")[0]
    assert "<EMAIL_ADDRESS>" in out
    assert "EMAIL_ADDRESS" in out


def test_anonymize_defaults_to_the_offline_backend(monkeypatch, tmp_path, capsys):
    # The command is a demo: it must work without configuring an anonymizer,
    # and without requiring the optional presidio dependency.
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.delenv("RAG_ANONYMIZER", raising=False)
    assert main(["anonymize", "Call 555-0142 tomorrow."]) == 0
    assert "regex" in capsys.readouterr().out


def test_anonymize_reports_a_clean_text_as_such(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    assert main(["anonymize", "The bins go out on Thursday."]) == 0
    assert "No PII detected" in capsys.readouterr().out


def test_anonymize_json_output_is_machine_readable(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    assert main(["anonymize", "--json", "Mail jane.doe@example.com now."]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["anonymizer"] == "regex"
    assert "<EMAIL_ADDRESS>" in payload["anonymized_text"]
    entity = payload["entities"][0]
    assert entity["type"] == "EMAIL_ADDRESS"
    assert entity["start"] < entity["end"]


def test_anonymize_treats_none_as_no_choice(monkeypatch, tmp_path, capsys):
    # The command exists to show what a backend would redact, so "none" (the
    # engine-wide default) is not a meaningful selection for it and falls back
    # to the offline regex backend.
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.setenv("RAG_ANONYMIZER", "none")
    assert main(["anonymize", "Mail jane.doe@example.com now."]) == 0
    assert "Backend: regex" in capsys.readouterr().out


# --- rag ingest -------------------------------------------------------------


def test_ingest_summarises_the_pii_it_removed(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.setenv("RAG_ANONYMIZER", "regex")
    folder = _write_corpus(
        tmp_path, "Reach the office at jane.doe@example.com or 555-0142."
    )

    assert main(["ingest", str(folder)]) == 0

    out = capsys.readouterr().out
    assert "Anonymized" in out
    assert "EMAIL_ADDRESS=1" in out
    # The counts are reported; the values themselves never are.
    assert "jane.doe@example.com" not in out


def test_ingest_of_a_clean_corpus_says_nothing_about_pii(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.setenv("RAG_ANONYMIZER", "regex")
    folder = _write_corpus(tmp_path)

    assert main(["ingest", str(folder)]) == 0
    assert "Anonymized" not in capsys.readouterr().out


# --- rag query --------------------------------------------------------------


def test_query_prints_sources_under_the_answer(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.0")
    folder = _write_corpus(tmp_path)
    main(["ingest", str(folder)])
    capsys.readouterr()

    assert main(["query", "When is recycling collected?"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out
    assert "recycling.md" in out


def test_query_that_is_refused_prints_no_sources(monkeypatch, tmp_path, capsys):
    _offline_env(monkeypatch, tmp_path / "index")
    folder = _write_corpus(tmp_path)
    main(["ingest", str(folder)])
    capsys.readouterr()

    # A threshold no chunk can clear makes the refusal deterministic offline.
    monkeypatch.setenv("RAG_SIMILARITY_THRESHOLD", "0.99")
    assert main(["query", "What is the boiling point of liquid helium?"]) == 0

    out = capsys.readouterr().out
    assert "enough context" in out.lower()
    assert "Sources:" not in out


# --- verbosity --------------------------------------------------------------


def test_verbosity_flags_map_to_levels(monkeypatch):
    monkeypatch.delenv("RAG_LOG_LEVEL", raising=False)
    assert _log_level(2) == "DEBUG"
    assert _log_level(3) == "DEBUG"
    assert _log_level(1) == "INFO"
    assert _log_level(0) == "WARNING"


def test_the_environment_sets_the_level_when_no_flag_is_given(monkeypatch):
    monkeypatch.setenv("RAG_LOG_LEVEL", "ERROR")
    assert _log_level(0) == "ERROR"
