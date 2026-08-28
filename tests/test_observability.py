"""Tests for logging: what is emitted, and what must never be.

Two things are asserted here. That the operations a reader needs to diagnose a
problem (ingestion, retrieval, refusals, provider calls) leave a trace. And that
the trace never contains user content: a question, an answer, a document, or a
secret. Logs are read by more people than the data itself.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from rag_engine.config import RagConfig
from rag_engine.observability import LOGGER_NAME, configure_logging
from rag_engine.pipeline import RagPipeline

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(autouse=True)
def restore_package_logger():
    """Keep logging configuration from leaking between tests in this module."""
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers = handlers
    logger.setLevel(level)


def _pipeline(tmp_path) -> RagPipeline:
    config = RagConfig(
        embedder="hashing",
        llm_provider="extractive",
        index_dir=tmp_path / "index",
        top_k=4,
    )
    return RagPipeline(config)


# --- the operations leave a trace ------------------------------------------


def test_ingest_logs_a_summary(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        pipeline = _pipeline(tmp_path)
        count = pipeline.ingest(SAMPLE_DIR)

    text = caplog.text
    assert "ingest" in text
    assert f"chunks={count}" in text
    assert "duration_ms=" in text


def test_index_save_and_load_are_logged(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(SAMPLE_DIR)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        # A fresh pipeline must load the index from disk.
        _pipeline(tmp_path).answer("When is recycling collected in Maplewood?")

    assert "index loaded" in caplog.text


def test_answer_logs_the_retrieval_outcome(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(SAMPLE_DIR)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        pipeline.answer("When is recycling collected in Maplewood?")

    text = caplog.text
    assert "refused=False" in text
    assert "retrieved=" in text
    assert "best_score=" in text


def test_a_refusal_is_logged_as_such(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(SAMPLE_DIR)
    pipeline.config.similarity_threshold = 0.95

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = pipeline.answer("What is the boiling point of liquid helium?")

    assert result.refused is True
    assert "refused=True" in caplog.text


def test_anonymized_entities_are_reported_as_counts(tmp_path, caplog):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text(
        "Contact jane.doe@example.com or call 555-0142 about the invoice.",
        encoding="utf-8",
    )
    config = RagConfig(
        embedder="hashing",
        llm_provider="extractive",
        anonymizer="regex",
        index_dir=tmp_path / "index",
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        RagPipeline(config).ingest(corpus)

    assert "pii_redacted=" in caplog.text
    assert "jane.doe@example.com" not in caplog.text


# --- user content never reaches the logs ------------------------------------


def test_the_question_is_not_logged(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(SAMPLE_DIR)
    question = "When is recycling collected in Maplewood?"

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        pipeline.answer(question)

    assert question not in caplog.text
    assert "recycling collected" not in caplog.text


def test_the_answer_is_not_logged(tmp_path, caplog):
    pipeline = _pipeline(tmp_path)
    pipeline.ingest(SAMPLE_DIR)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = pipeline.answer("When is recycling collected in Maplewood?")

    # The answer is stitched from the corpus, so a distinctive fragment of it
    # must not appear in the trace either.
    fragment = result.answer.strip()[:40]
    assert fragment
    assert fragment not in caplog.text


def test_document_text_is_not_logged(tmp_path, caplog):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    secret_sentence = "The launch code is hunter2 and the vault opens at dawn."
    (corpus / "note.md").write_text(secret_sentence, encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        config = RagConfig(
            embedder="hashing",
            llm_provider="extractive",
            index_dir=tmp_path / "index",
        )
        RagPipeline(config).ingest(corpus)

    assert "hunter2" not in caplog.text


# --- the library never configures logging for its host ----------------------


def test_importing_the_package_attaches_no_real_handler():
    # A library that attaches handlers hijacks the host application's logging.
    # Checked in a fresh interpreter: the claim is about importing the package,
    # not about whatever earlier tests left on the global logger.
    probe = (
        "import logging, rag_engine, rag_engine.pipeline;"
        "handlers = logging.getLogger('rag_engine').handlers;"
        "assert all(isinstance(h, logging.NullHandler) for h in handlers), handlers;"
        "logging.getLogger('rag_engine').warning('nothing should print this')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_configure_logging_attaches_exactly_one_handler():
    configure_logging("INFO")
    configure_logging("DEBUG")  # calling twice must not stack handlers
    logger = logging.getLogger(LOGGER_NAME)
    stream_handlers = [
        h for h in logger.handlers if not isinstance(h, logging.NullHandler)
    ]
    assert len(stream_handlers) == 1
    assert logger.level == logging.DEBUG


def test_configure_logging_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="log_level"):
        configure_logging("chatty")


def test_log_level_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAG_LOG_LEVEL", "warning")
    assert RagConfig.from_env().log_level == "WARNING"


def test_an_unknown_log_level_is_rejected_at_the_boundary():
    with pytest.raises(ValueError, match="log_level"):
        RagConfig(log_level="chatty")


# --- the CLI is the application that configures logging ---------------------


def test_cli_is_quiet_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.delenv("RAG_LOG_LEVEL", raising=False)
    from rag_engine.cli import main

    assert main(["ingest", str(SAMPLE_DIR)]) == 0
    # The command's own output goes to stdout; nothing is logged to stderr.
    captured = capsys.readouterr()
    assert "Indexed" in captured.out
    assert "ingest completed" not in captured.err


def test_cli_verbose_flag_emits_the_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    from rag_engine.cli import main

    assert main(["--verbose", "ingest", str(SAMPLE_DIR)]) == 0
    assert "ingest completed" in capsys.readouterr().err
