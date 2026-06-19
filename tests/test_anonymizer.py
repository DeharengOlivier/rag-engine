"""Tests for PII anonymization (regex backend, factory, and result types).

The Presidio backend is covered by a separate test guarded with
``importorskip``, so this file runs fully offline with no extra dependencies.
"""

from __future__ import annotations

import pytest

from rag_engine.anonymizer import (
    AnonymizationResult,
    DetectedEntity,
    NoOpAnonymizer,
    RegexAnonymizer,
    build_anonymizer,
)
from rag_engine.config import RagConfig


# -- RegexAnonymizer: detection ------------------------------------------- #


def test_regex_redacts_email():
    result = RegexAnonymizer().anonymize("Reach me at john.doe@acme.com please.")
    assert result.text == "Reach me at <EMAIL_ADDRESS> please."
    assert result.entity_counts() == {"EMAIL_ADDRESS": 1}
    assert result.entities[0].text == "john.doe@acme.com"


def test_regex_redacts_credit_card():
    result = RegexAnonymizer().anonymize("Card 4111 1111 1111 1111 on file.")
    assert "<CREDIT_CARD>" in result.text
    assert "4111" not in result.text
    assert result.entity_counts() == {"CREDIT_CARD": 1}


def test_regex_redacts_iban():
    result = RegexAnonymizer().anonymize("IBAN DE89370400440532013000 confirmed.")
    assert "<IBAN_CODE>" in result.text
    assert "DE89" not in result.text


def test_regex_redacts_phone_number():
    result = RegexAnonymizer().anonymize("Call +33 6 12 34 56 78 tomorrow.")
    assert "<PHONE_NUMBER>" in result.text
    assert "12 34" not in result.text


def test_regex_redacts_ssn_and_ip():
    result = RegexAnonymizer().anonymize("SSN 123-45-6789 from host 10.0.0.42.")
    types = set(result.entity_counts())
    assert types == {"US_SSN", "IP_ADDRESS"}
    assert "123-45-6789" not in result.text
    assert "10.0.0.42" not in result.text


def test_regex_handles_multiple_entities_in_one_text():
    text = "Email a@b.com or call 0044 20 7946 0958."
    result = RegexAnonymizer().anonymize(text)
    counts = result.entity_counts()
    assert counts["EMAIL_ADDRESS"] == 1
    assert counts["PHONE_NUMBER"] == 1


# -- RegexAnonymizer: precision (no false positives) ----------------------- #


def test_regex_leaves_clean_text_untouched():
    text = "The library opens at nine and closes at five on weekdays."
    result = RegexAnonymizer().anonymize(text)
    assert result.text == text
    assert not result.found_pii


def test_regex_short_number_is_not_a_phone():
    # A four-digit year must not be redacted as a phone number.
    result = RegexAnonymizer().anonymize("Founded in 2024 by the team.")
    assert result.text == "Founded in 2024 by the team."
    assert not result.found_pii


def test_regex_credit_card_not_split_into_phone():
    # Overlap resolution must keep the single CREDIT_CARD span, not also emit a
    # PHONE_NUMBER over the same digits.
    result = RegexAnonymizer().anonymize("4111 1111 1111 1111")
    assert result.entity_counts() == {"CREDIT_CARD": 1}


def test_regex_empty_text():
    result = RegexAnonymizer().anonymize("")
    assert result.text == ""
    assert result.entities == []


# -- Result type ----------------------------------------------------------- #


def test_anonymization_result_counts_and_flag():
    result = AnonymizationResult(
        text="<PERSON> and <PERSON> wrote to <EMAIL_ADDRESS>.",
        entities=[
            DetectedEntity("PERSON", 0, 8, "Alice"),
            DetectedEntity("PERSON", 13, 21, "Bob"),
            DetectedEntity("EMAIL_ADDRESS", 25, 40, "a@b.com"),
        ],
    )
    assert result.found_pii
    assert result.entity_counts() == {"PERSON": 2, "EMAIL_ADDRESS": 1}


# -- NoOpAnonymizer -------------------------------------------------------- #


def test_noop_passthrough():
    result = NoOpAnonymizer().anonymize("john.doe@acme.com stays as-is")
    assert result.text == "john.doe@acme.com stays as-is"
    assert not result.found_pii


# -- Factory --------------------------------------------------------------- #


def test_build_anonymizer_none_is_noop():
    anon = build_anonymizer(RagConfig(anonymizer="none"))
    assert isinstance(anon, NoOpAnonymizer)
    assert anon.name == "none"


def test_build_anonymizer_regex():
    anon = build_anonymizer(RagConfig(anonymizer="regex"))
    assert isinstance(anon, RegexAnonymizer)
    assert anon.name == "regex"


def test_build_anonymizer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown anonymizer"):
        build_anonymizer(RagConfig(anonymizer="magic"))


# -- Presidio backend (skipped when the optional dependency is absent) ----- #


def test_presidio_detects_person_and_email():
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("presidio_anonymizer")
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("spaCy model 'en_core_web_sm' is not installed")

    from rag_engine.anonymizer import PresidioAnonymizer

    anon = PresidioAnonymizer(model_name="en_core_web_sm")
    result = anon.anonymize("John Doe lives in Paris, email john@acme.com.")
    types = set(result.entity_counts())
    # Presidio adds named-entity recognition the regex backend cannot do.
    assert "PERSON" in types
    assert "EMAIL_ADDRESS" in types
    assert "John Doe" not in result.text
    assert "john@acme.com" not in result.text
