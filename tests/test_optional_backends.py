"""Tests for the two optional backends, without installing their dependencies.

The presidio anonymizer and the sentence-transformers embedder were the only
untested code in the package, because their packages are heavy and are not
installed by default. Fake modules are injected in their place, which exercises
the code this repository owns: how a third-party result is mapped onto our own
types, and how the offline promise is kept when the package is missing.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from rag_engine.anonymizer import PresidioAnonymizer, build_anonymizer
from rag_engine.config import RagConfig
from rag_engine.embeddings import SentenceTransformerEmbedder, build_embedder

# --- presidio ---------------------------------------------------------------


class _FakeRecognizerResult:
    def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


def _install_fake_presidio(monkeypatch, results):
    """Put a minimal presidio stand-in in sys.modules and record its config."""
    recorded: dict = {}

    class _AnalyzerEngine:
        def __init__(self, nlp_engine=None):
            recorded["nlp_engine"] = nlp_engine

        def analyze(self, text, language):
            recorded["language"] = language
            return results

    class _NlpEngineProvider:
        def __init__(self, nlp_configuration=None):
            recorded["nlp_configuration"] = nlp_configuration

        def create_engine(self):
            return "engine"

    analyzer = types.ModuleType("presidio_analyzer")
    analyzer.AnalyzerEngine = _AnalyzerEngine
    nlp_engine = types.ModuleType("presidio_analyzer.nlp_engine")
    nlp_engine.NlpEngineProvider = _NlpEngineProvider
    anonymizer = types.ModuleType("presidio_anonymizer")
    anonymizer.AnonymizerEngine = type("AnonymizerEngine", (), {})

    monkeypatch.setitem(sys.modules, "presidio_analyzer", analyzer)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp_engine)
    monkeypatch.setitem(sys.modules, "presidio_anonymizer", anonymizer)
    return recorded


TEXT = "Jane Doe lives in Paris."
JANE = _FakeRecognizerResult("PERSON", 0, 8, 0.85)
PARIS = _FakeRecognizerResult("LOCATION", 18, 23, 0.4)


def test_presidio_replaces_each_span_with_a_typed_placeholder(monkeypatch):
    _install_fake_presidio(monkeypatch, [JANE, PARIS])
    result = PresidioAnonymizer(score_threshold=0.0).anonymize(TEXT)

    assert result.text == "<PERSON> lives in <LOCATION>."
    assert [e.entity_type for e in result.entities] == ["PERSON", "LOCATION"]
    assert [e.text for e in result.entities] == ["Jane Doe", "Paris"]


def test_presidio_drops_detections_below_the_threshold(monkeypatch):
    _install_fake_presidio(monkeypatch, [JANE, PARIS])
    result = PresidioAnonymizer(score_threshold=0.5).anonymize(TEXT)

    # Paris scored 0.4 and must survive untouched.
    assert result.text == "<PERSON> lives in Paris."
    assert [e.entity_type for e in result.entities] == ["PERSON"]


def test_presidio_never_replaces_a_character_twice(monkeypatch):
    # Two detectors claiming overlapping spans is normal; the output must stay
    # coherent rather than nesting one placeholder inside another.
    overlapping = [
        _FakeRecognizerResult("PERSON", 0, 8, 0.9),
        _FakeRecognizerResult("PERSON", 5, 8, 0.7),
    ]
    _install_fake_presidio(monkeypatch, overlapping)
    result = PresidioAnonymizer(score_threshold=0.0).anonymize(TEXT)

    assert result.text == "<PERSON> lives in Paris."
    assert len(result.entities) == 1


def test_presidio_on_empty_text_does_nothing(monkeypatch):
    _install_fake_presidio(monkeypatch, [JANE])
    result = PresidioAnonymizer().anonymize("")
    assert result.text == ""
    assert result.entities == []


def test_presidio_is_configured_with_the_requested_model(monkeypatch):
    recorded = _install_fake_presidio(monkeypatch, [])
    PresidioAnonymizer(model_name="fr_core_news_sm", language="fr").anonymize("x")

    models = recorded["nlp_configuration"]["models"]
    assert models == [{"lang_code": "fr", "model_name": "fr_core_news_sm"}]
    assert recorded["language"] == "fr"


def test_presidio_entity_counts_are_reported_per_type(monkeypatch):
    _install_fake_presidio(monkeypatch, [JANE, PARIS])
    result = PresidioAnonymizer(score_threshold=0.0).anonymize(TEXT)
    assert result.entity_counts() == {"PERSON": 1, "LOCATION": 1}


# --- sentence-transformers --------------------------------------------------


def _install_fake_sentence_transformers(monkeypatch, dim=4):
    recorded: dict = {}

    class _SentenceTransformer:
        def __init__(self, model_name):
            recorded["model_name"] = model_name

        def get_sentence_embedding_dimension(self):
            return dim

        def encode(self, texts, normalize_embeddings=False, convert_to_numpy=False):
            recorded["normalize_embeddings"] = normalize_embeddings
            recorded["convert_to_numpy"] = convert_to_numpy
            recorded["texts"] = texts
            return np.ones((len(texts), dim), dtype=np.float64) / np.sqrt(dim)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return recorded


def test_sentence_transformer_embedder_reports_the_model_dimension(monkeypatch):
    _install_fake_sentence_transformers(monkeypatch, dim=6)
    assert SentenceTransformerEmbedder().dim == 6


def test_sentence_transformer_embedder_asks_for_normalized_float32(monkeypatch):
    # The vector store treats cosine similarity as a plain dot product, which is
    # only correct if the vectors arrive normalized.
    recorded = _install_fake_sentence_transformers(monkeypatch)
    vectors = SentenceTransformerEmbedder().embed(["a", "b"])

    assert recorded["normalize_embeddings"] is True
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_sentence_transformer_embedder_on_no_text_returns_an_empty_matrix(
    monkeypatch,
):
    _install_fake_sentence_transformers(monkeypatch, dim=5)
    vectors = SentenceTransformerEmbedder().embed([])
    assert vectors.shape == (0, 5)
    assert vectors.dtype == np.float32


def test_build_embedder_passes_the_configured_model_name(monkeypatch):
    recorded = _install_fake_sentence_transformers(monkeypatch)
    build_embedder(
        RagConfig(embedder="sentence-transformers", embedding_model="all-mpnet-base-v2")
    )
    assert recorded["model_name"] == "all-mpnet-base-v2"


# --- the offline promise ----------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "factory"),
    [
        ("sentence_transformers", SentenceTransformerEmbedder),
        ("presidio_analyzer", PresidioAnonymizer),
    ],
)
def test_a_missing_optional_package_is_reported_actionably(
    module_name, factory, monkeypatch
):
    # Selecting a backend whose package is absent must say what to install and
    # how to keep running offline, not raise a bare ImportError from deep inside.
    monkeypatch.setitem(sys.modules, module_name, None)
    with pytest.raises(ImportError, match="pip install"):
        factory()


def test_build_anonymizer_configures_the_presidio_backend(monkeypatch):
    recorded = _install_fake_presidio(monkeypatch, [])
    config = RagConfig(
        anonymizer="presidio",
        anonymize_model="de_core_news_sm",
        anonymize_threshold=0.75,
    )
    anonymizer = build_anonymizer(config)
    anonymizer.anonymize("x")

    assert anonymizer.name == "presidio"
    assert recorded["nlp_configuration"]["models"][0]["model_name"] == "de_core_news_sm"
