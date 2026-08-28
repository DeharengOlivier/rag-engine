"""PII anonymization for documents before they enter the index.

A retrieval system over private documents has a privacy problem the moment it
persists anything: the vector index, its metadata sidecar, and any prompt sent
to an external LLM can all leak personally identifiable information (PII). This
module removes that risk at the source by anonymizing text *before* it is
embedded, indexed, or sent anywhere.

Two implementations sit behind a common :class:`Anonymizer` protocol:

- :class:`RegexAnonymizer` (default when anonymization is on): a dependency-free,
  deterministic detector for the most common structured PII (emails, phone
  numbers, credit cards, IBANs, IP addresses, social-security numbers). It needs
  no model and no network, so it keeps the engine offline-first and makes the
  test suite reproducible everywhere.

- :class:`PresidioAnonymizer` (optional): wraps Microsoft Presidio
  (``presidio-analyzer`` + ``presidio-anonymizer``) backed by a spaCy model. It
  adds named-entity detection (people, locations, organizations, dates) on top
  of the structured patterns. The import is lazy, so the dependency is only
  required if you actually select it.

Both return the same :class:`AnonymizationResult`, so the rest of the engine and
the tests do not care which backend produced it. Detected spans are replaced
with a typed placeholder such as ``<EMAIL_ADDRESS>`` or ``<PERSON>``, which keeps
the surrounding text readable and still retrievable while carrying no raw PII.

``build_anonymizer(config)`` selects the implementation from configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rag_engine.config import RagConfig


@dataclass
class DetectedEntity:
    """A single piece of PII found in a text.

    Attributes:
        entity_type: The category, e.g. ``"EMAIL_ADDRESS"`` or ``"PERSON"``.
        start: Start offset (inclusive) of the span in the original text.
        end: End offset (exclusive) of the span in the original text.
        text: The original substring that was detected (kept for auditing only;
            it is never written to the index).
        score: Detector confidence in ``[0, 1]``. Deterministic regex matches
            report ``1.0``.
    """

    entity_type: str
    start: int
    end: int
    text: str
    score: float = 1.0


@dataclass
class AnonymizationResult:
    """The outcome of anonymizing one text.

    Attributes:
        text: The anonymized text, with every detected span replaced by a typed
            placeholder like ``<PERSON>``.
        entities: Every entity that was detected and replaced.
    """

    text: str
    entities: list[DetectedEntity] = field(default_factory=list)

    @property
    def found_pii(self) -> bool:
        """True if at least one PII entity was detected."""
        return bool(self.entities)

    def entity_counts(self) -> dict[str, int]:
        """Return a count of detected entities per type, e.g. ``{"PERSON": 2}``."""
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts


@runtime_checkable
class Anonymizer(Protocol):
    """Common interface for all anonymizers."""

    @property
    def name(self) -> str:
        """Short identifier of the backend (used in logs and reports)."""
        ...

    def anonymize(self, text: str) -> AnonymizationResult:
        """Detect PII in ``text`` and return it with every span replaced."""
        ...


def _placeholder(entity_type: str) -> str:
    """Render the replacement token for an entity type, e.g. ``<EMAIL_ADDRESS>``."""
    return f"<{entity_type}>"


def _resolve_overlaps(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Drop overlapping spans, keeping the earliest then the longest match.

    Detectors can match the same characters in more than one way (a credit-card
    number also looks phone-ish). We sort by start offset, then by descending
    length, and greedily keep a span only if it does not overlap one already
    kept. This guarantees each character is anonymized at most once.
    """
    ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    kept: list[DetectedEntity] = []
    last_end = -1
    for entity in ordered:
        if entity.start >= last_end:
            kept.append(entity)
            last_end = entity.end
    return kept


def _apply(text: str, entities: list[DetectedEntity]) -> str:
    """Replace each detected span with its typed placeholder.

    Replacements are applied right-to-left so earlier offsets stay valid while
    later ones are rewritten.
    """
    result = text
    for entity in sorted(entities, key=lambda e: e.start, reverse=True):
        result = (
            result[: entity.start]
            + _placeholder(entity.entity_type)
            + result[entity.end :]
        )
    return result


class NoOpAnonymizer:
    """An anonymizer that does nothing (the default when anonymization is off).

    Returning the text unchanged lets the pipeline treat "no anonymization" as
    just another backend, with no special-casing in the calling code.
    """

    name = "none"

    def anonymize(self, text: str) -> AnonymizationResult:
        return AnonymizationResult(text=text, entities=[])


class RegexAnonymizer:
    """Deterministic, dependency-free detector for common structured PII.

    Each pattern targets a category of PII whose shape is regular enough to match
    reliably without a model. This is intentionally conservative: it favours
    precision (few false positives) over recall, because it is the offline
    fallback, not the full detector. For named entities (people, places,
    organizations) use :class:`PresidioAnonymizer`.

    Patterns are checked highest-priority first; overlapping matches are then
    resolved so every character is replaced at most once.
    """

    name = "regex"

    # Ordered by priority: structured, high-confidence patterns come first so a
    # credit-card number is never mistaken for a phone number, and so on.
    _PATTERNS: tuple[tuple[str, str], ...] = (
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ("CREDIT_CARD", r"\b\d{4}(?:[ -]?\d{4}){3}\b"),
        ("IBAN_CODE", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        ("US_SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        (
            "PHONE_NUMBER",
            # Optional +CC or 00 prefix, then groups of digits with common
            # separators, at least seven digits in total. Anchored so it does
            # not start or end in the middle of a longer alphanumeric token.
            r"(?<![\w+])(?:\+|00)?\d(?:[\d\s().-]{5,}\d)(?![\w])",
        ),
    )

    def __init__(self) -> None:
        self._compiled = [
            (entity_type, re.compile(pattern)) for entity_type, pattern in self._PATTERNS
        ]

    def anonymize(self, text: str) -> AnonymizationResult:
        if not text:
            return AnonymizationResult(text=text, entities=[])

        found: list[DetectedEntity] = []
        for entity_type, pattern in self._compiled:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                # A phone match needs enough digits to be a real number; this
                # rejects short numeric runs and most dates.
                if entity_type == "PHONE_NUMBER":
                    digits = sum(c.isdigit() for c in match.group())
                    if digits < 7:
                        continue
                found.append(
                    DetectedEntity(
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        text=match.group(),
                        score=1.0,
                    )
                )

        entities = _resolve_overlaps(found)
        return AnonymizationResult(text=_apply(text, entities), entities=entities)


class PresidioAnonymizer:
    """Optional anonymizer backed by Microsoft Presidio (lazy import).

    Presidio combines pattern recognizers with a spaCy NLP model, so it detects
    named entities (people, locations, organizations, dates) in addition to the
    structured PII the regex backend handles. Constructing this class loads the
    analyzer and the spaCy model; importing this module does not.

    Selecting this backend without the packages installed raises a clear,
    actionable error.
    """

    name = "presidio"

    def __init__(
        self,
        *,
        model_name: str = "en_core_web_sm",
        language: str = "en",
        score_threshold: float = 0.5,
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "The 'presidio-analyzer' and 'presidio-anonymizer' packages are "
                "required for the presidio anonymizer. Install them with "
                "'pip install \"rag-engine[pii]\"' and download a spaCy model "
                "('python -m spacy download en_core_web_sm'), or set "
                "RAG_ANONYMIZER=regex to run offline."
            ) from exc

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": model_name}],
            }
        )
        self._language = language
        self._score_threshold = score_threshold
        self._analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())
        self._anonymizer = AnonymizerEngine()

    def anonymize(self, text: str) -> AnonymizationResult:
        if not text:
            return AnonymizationResult(text=text, entities=[])

        results = self._analyzer.analyze(text=text, language=self._language)
        kept = [r for r in results if r.score >= self._score_threshold]
        entities = [
            DetectedEntity(
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start : r.end],
                score=float(r.score),
            )
            for r in kept
        ]
        # Reuse our own deterministic replacement so the output format matches
        # the regex backend exactly (<TYPE> placeholders, no character touched
        # twice), rather than depending on Presidio's operator defaults.
        entities = _resolve_overlaps(entities)
        return AnonymizationResult(text=_apply(text, entities), entities=entities)


def build_anonymizer(config: RagConfig) -> Anonymizer:
    """Instantiate the anonymizer selected in ``config``.

    Raises:
        ValueError: If ``config.anonymizer`` is not a recognized value.
    """
    # config.anonymizer is already validated and canonical (see RagConfig), so the
    # comparisons are exact and the final raise only fires if a value is added to
    # ANONYMIZERS without a branch here.
    if config.anonymizer == "none":
        return NoOpAnonymizer()
    if config.anonymizer == "regex":
        return RegexAnonymizer()
    if config.anonymizer == "presidio":
        return PresidioAnonymizer(
            model_name=config.anonymize_model,
            score_threshold=config.anonymize_threshold,
        )
    raise ValueError(
        f"Unknown anonymizer '{config.anonymizer}'. "
        "Expected 'none', 'regex', or 'presidio'."
    )
