"""Logging for the engine.

Two rules shape this module.

**A library never configures logging for its host.** Every module logs through
``logging.getLogger(__name__)`` and nothing else. The package logger carries a
:class:`logging.NullHandler`, so importing ``rag_engine`` prints nothing. Only an
application decides where logs go: the CLI calls :func:`configure_logging`, and a
library user is free to ignore it and use their own configuration.

**Logs never carry user content.** Questions, answers, document text and
detected PII values stay out of every record, at every level. What is logged is
the shape of the operation: how many chunks, how many results, the best score,
whether the engine refused, how long it took. That is enough to diagnose a
problem without turning the log file into a second copy of the corpus.

Records use ``key=value`` pairs so they can be grepped or parsed without pulling
in a structured-logging dependency.
"""

from __future__ import annotations

import logging

#: Root logger for the package. Every module logger is a child of this one.
LOGGER_NAME = "rag_engine"

#: Accepted values for ``RagConfig.log_level`` and :func:`configure_logging`.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_HANDLER_NAME = "rag_engine.stream"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"

# Importing the package must never print anything on its own.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Send the package's logs to stderr at ``level``.

    Intended for applications (the CLI, a script). Calling it twice replaces the
    handler rather than stacking a second one, so repeated calls stay harmless.

    Args:
        level: One of :data:`LOG_LEVELS`, case-insensitive.

    Returns:
        The configured package logger.

    Raises:
        ValueError: ``level`` is not a known logging level.
    """
    normalized = level.strip().upper()
    if normalized not in LOG_LEVELS:
        options = ", ".join(LOG_LEVELS)
        raise ValueError(f"log_level must be one of: {options}. Got {level!r}.")

    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if h.name == _HANDLER_NAME]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, normalized))
    return logger


def duration_ms(started_at: float, ended_at: float) -> float:
    """Return an elapsed :func:`time.perf_counter` span in milliseconds, rounded."""
    return round((ended_at - started_at) * 1000, 1)
