# rag-engine

[![CI](https://github.com/DeharengOlivier/rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/DeharengOlivier/rag-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, well-architected, offline-first Retrieval-Augmented Generation (RAG)
engine for asking questions over your own documents.

## Overview

`rag-engine` ingests a folder of plain-text or Markdown documents, indexes them
locally, and answers questions about them with grounded, cited answers. It is
generic: point it at any collection of `.txt` / `.md` files and ask away.

The whole system runs fully offline by default. The default embedder is a
dependency-free hashing embedder (pure numpy), and the default answer generator
is an extractive provider that stitches together the retrieved context without
calling any model. There is no required API key and no network call. You can
optionally swap in `sentence-transformers` embeddings or an Anthropic / OpenAI
LLM through configuration.

## Architecture

```
                +-------------------+
   documents -->|     ingestion     |  load .txt/.md, clean, chunk (with overlap)
                +---------+---------+
                          |
                          v
                +-------------------+
                |   anonymization   |  redact PII before indexing
                |  (optional)       |  regex (offline) | presidio + spaCy
                +---------+---------+
                          |
                          v
                +-------------------+
                |    embeddings     |  hashing (offline) | sentence-transformers
                +---------+---------+
                          |
                          v
                +-------------------+
                |   vector store    |  numpy matrix + cosine similarity (save/load)
                +---------+---------+
                          |
   question --> embed --> v
                +-------------------+
                |     retriever     |  top-k most similar chunks (+ scores, sources)
                +---------+---------+
                          |
                          v
                +-------------------+
                |    guardrails     |  grounding gate + citations + safe refusal
                +---------+---------+
                          |
                          v
                +-------------------+
                |  LLM generation   |  extractive (offline) | anthropic | openai
                +---------+---------+
                          |
                          v
                  cited, grounded answer  (or "I don't have enough context")

   evaluation harness: runs labeled questions -> recall@k + keyword score report
```

## Why this design

- **Provider and embedder abstraction.** Embedders and LLMs sit behind small
  interfaces (`Embedder`, `LLM`) and are selected from configuration. Swapping a
  real embedding model or an API-backed LLM is a one-line config change, and the
  rest of the pipeline is unaffected.
- **Offline-first fallbacks.** The default hashing embedder and extractive LLM
  need only numpy and the standard library, so the engine (and its tests) run
  with no network and no API key. Heavy or paid dependencies are optional and
  imported lazily, only when selected.
- **PII anonymization at the source.** Anonymization runs at ingestion, before
  any text is embedded or written to disk, so the vector index and its metadata
  sidecar never hold raw personal data. An offline regex backend (no
  dependencies) redacts structured PII (emails, phones, credit cards, IBANs, IPs,
  SSNs); an optional Microsoft Presidio backend adds named-entity detection
  (people, locations, organizations) on top. Both replace each span with a typed
  placeholder like `<EMAIL_ADDRESS>` or `<PERSON>`, which keeps the surrounding
  text readable and retrievable while carrying no raw PII.
- **Grounding guardrails.** Retrieval scores gate generation: if no chunk clears
  a similarity threshold, the engine refuses instead of guessing. When it does
  answer, every supporting chunk is attached as a citation, so answers are
  auditable.
- **An index you can trust or rebuild.** The index spans two files that have to
  agree with each other, so a save is staged in a sibling directory and swapped
  into place: an interrupted write leaves the previous index intact instead of a
  mismatched pair. Loading checks the two files against each other and refuses a
  torn index, since the corpus is the source of truth and re-running ingestion
  is always the recovery path.
- **Evaluation built in.** A tiny harness measures retrieval recall@k and a
  keyword-coverage proxy for answer quality, catching the most common
  regressions without needing a second model to grade.

## Quickstart

```bash
# Install (numpy is the only hard dependency).
pip install -e .

# Index the bundled sample documents.
rag ingest data/sample

# Ask an in-corpus question (grounded, cited answer).
rag query "What are the library opening hours on Saturday?"

# Ask an out-of-corpus question (safe refusal).
rag query "What is the population of Mars?"

# Run the evaluation harness.
rag eval evals/sample_eval.json

# See what PII an anonymizer would redact in a piece of text (offline).
rag anonymize "Email me at jane.doe@acme.com or call +1 212 555 0199."
```

You can also use it as a library:

```python
from rag_engine import RagPipeline

pipeline = RagPipeline()
pipeline.ingest("data/sample")
result = pipeline.answer("Which day is recycling collected?")
print(result.answer)
print(result.citations)
print(result.refused)
```

## Privacy: PII anonymization

Indexing private documents is a privacy risk the moment anything is persisted:
the index, its metadata sidecar, and any prompt sent to an external LLM can leak
personal data. `rag-engine` addresses this by anonymizing text **at ingestion,
before it is embedded or written to disk**, so raw PII never enters the index.

Two backends sit behind one interface, selected with `RAG_ANONYMIZER`:

- `regex` (offline, no dependencies): deterministic detection of structured PII
  (emails, phone numbers, credit cards, IBANs, IP addresses, SSNs). Keeps the
  engine offline-first and the tests reproducible.
- `presidio` (optional): Microsoft Presidio plus a spaCy model, adding
  named-entity recognition (people, locations, organizations, dates) on top of
  the structured patterns.

```bash
# Offline regex backend.
rag anonymize "I am Jane Doe, jane@acme.com, +1 212 555 0199, card 4111 1111 1111 1111."
#  -> I am Jane Doe, <EMAIL_ADDRESS>, <PHONE_NUMBER>, card <CREDIT_CARD>.

# Optional Presidio backend also redacts names and places.
pip install -e ".[pii]"
python -m spacy download en_core_web_sm
RAG_ANONYMIZER=presidio rag anonymize "I am Jane Doe from Paris, jane@acme.com."
#  -> I am <PERSON> from <LOCATION>, <EMAIL_ADDRESS>.

# Index with anonymization on: PII is stripped before it reaches the index.
RAG_ANONYMIZER=regex rag ingest data/sample
#  -> Anonymized N PII entities before indexing (EMAIL_ADDRESS=..., ...).
```

The guarantee is verified by the test suite: after an anonymized ingest, the
saved `meta.json` contains the typed placeholders and none of the original PII,
while the surrounding non-PII text stays intact (so documents remain
retrievable).

## Configuration

All settings are read from environment variables (see `.env.example`). Secrets
are never read at import time: API keys are read lazily, only when an
API-backed provider is actually called.

Every variable is validated when the configuration is built. A value that cannot
be parsed, or that falls outside the range below, raises `ConfigError`
immediately with a message naming the variable: a typo in a deployment config
fails at startup instead of quietly changing how the engine behaves. A blank
value counts as unset. The CLI turns that into a message and exit code `2`
rather than a traceback (`0` success, `1` a run that failed, `2` a bad setting).

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_EMBEDDER` | `hashing` | `hashing` (offline) or `sentence-transformers` (optional). |
| `RAG_LLM_PROVIDER` | `extractive` | `extractive` (offline), `anthropic`, or `openai`. |
| `ANTHROPIC_API_KEY` | (unset) | Optional. Only used by the `anthropic` provider. |
| `OPENAI_API_KEY` | (unset) | Optional. Only used by the `openai` provider. |
| `RAG_TOP_K` | `4` | Number of chunks retrieved per query. Must be > 0. |
| `RAG_CHUNK_SIZE` | `600` | Target chunk size in characters. Must be > 0. |
| `RAG_CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks, in characters. Must be within `[0, RAG_CHUNK_SIZE)`. |
| `RAG_SIMILARITY_THRESHOLD` | `0.15` | Minimum cosine similarity for a chunk to count as evidence. Must be within `[-1.0, 1.0]`. |
| `RAG_ANONYMIZER` | `none` | PII redaction at ingestion: `none`, `regex` (offline), or `presidio`. |
| `RAG_ANONYMIZE_MODEL` | `en_core_web_sm` | spaCy model used by the `presidio` backend. |
| `RAG_ANONYMIZE_THRESHOLD` | `0.5` | Minimum detector confidence for `presidio` to redact an entity. Must be within `[0.0, 1.0]`. |
| `RAG_EMBEDDING_DIM` | `256` | Vector dimension for the hashing embedder. Must be > 0. |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name for the sentence-transformers embedder. |
| `RAG_LLM_MODEL` | `claude-3-5-haiku-latest` | Model name for the anthropic/openai providers. |
| `RAG_LLM_TIMEOUT_SECONDS` | `30` | Timeout on every API-backed call. Must be > 0. |
| `RAG_LLM_MAX_RETRIES` | `2` | Retries the provider SDK may attempt, with exponential backoff and jitter. Must be within `[0, 5]`. |
| `RAG_INDEX_DIR` | `.rag_index` | Directory where the vector index is saved. |
| `RAG_LOG_LEVEL` | `WARNING` | Level for the engine's logs: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |

Optional extras (only needed if you switch providers):

```bash
pip install -e ".[embeddings]"  # sentence-transformers
pip install -e ".[llm]"         # anthropic + openai
pip install -e ".[pii]"         # presidio + spacy (PII anonymization)
pip install -e ".[dev]"         # pytest
```

The embedder and LLM providers are entirely optional. With the default
configuration, `rag-engine` runs fully offline using only numpy and the standard
library.

## Logging

The engine logs the shape of every operation and none of its content. A record
says how many chunks were indexed, how many results were retrieved, the best
similarity score, whether the grounding gate refused, and how long it took.
Questions, answers, document text and detected PII values never appear in a log
line, at any level: a log file must not become a second, unprotected copy of the
corpus.

The library follows the usual rule and configures nothing: it logs through
`logging.getLogger(__name__)` under a `rag_engine` logger carrying a
`NullHandler`, so importing the package prints nothing and your own logging
configuration wins. Applications opt in:

```bash
rag --verbose ingest data/sample     # INFO trace on stderr
rag -vv query "When is recycling collected?"   # DEBUG
RAG_LOG_LEVEL=INFO rag ingest data/sample      # same, via the environment
```

```python
from rag_engine.observability import configure_logging

configure_logging("INFO")  # or use your application's own handlers
```

## Scale, and where this design stops

Measured on an ordinary laptop, with chunks of about 600 characters at the
default `dim=256`:

| Chunks | Memory | On disk | Query |
| --- | --- | --- | --- |
| 10 000 | ~51 MB | 17 MB | 0.1 ms |
| 100 000 | ~480 MB | 173 MB | 1.9 ms |
| 500 000 | ~2.2 GB | 864 MB | 9.5 ms |

Each row was measured, not extrapolated: peak resident memory while building the
index, the size of the two files on disk, and the mean of twenty queries.

Everything is linear in the number of chunks, in both memory and query time: a
query is an exact scan of the whole matrix, never an approximation. Two limits
follow, and neither is a bug to be reported:

- **The index lives in memory.** A corpus that does not fit is out of scope for
  this design. An approximate index (FAISS) or a vector database is the right
  answer there, and the `add`/`search`/`save`/`load` interface is deliberately
  the shape one of those would expose, so the swap stays local.
- **Ingestion rebuilds the index.** `ingest()` re-reads and re-embeds the whole
  folder; there is no incremental update. For a corpus of this size that is
  seconds, and it keeps the index a pure function of the corpus on disk, which
  is what makes rebuilding a safe recovery from any bad state.

## Development

```bash
pip install -e ".[dev]"

pytest --cov          # the suite, offline, with branch coverage
ruff check .          # lint
ruff format .         # format
mypy                  # strict type check over src/
bandit -r src -q      # static analysis
pip-audit             # known vulnerabilities in dependencies
```

CI runs all of it on every push and pull request, and the suite on Python 3.10
through 3.13. It installs no optional dependency, which is how the offline-first
promise stays true rather than aspirational.

## Project structure

```
rag-engine/
  pyproject.toml            # packaging, console_scripts entry point `rag`
  README.md
  LICENSE                   # MIT
  .env.example              # config placeholders (no secrets)
  .gitignore
  data/sample/*.md          # bundled synthetic sample documents
  evals/sample_eval.json    # example evaluation cases
  src/rag_engine/
    config.py               # env-driven configuration, validated at the boundary
    observability.py        # package logger + application-side configuration
    ingestion.py            # load + clean + chunk documents
    anonymizer.py           # PII redaction: regex (offline) + presidio (optional)
    embeddings.py           # hashing (default) and sentence-transformers embedders
    vectorstore.py          # numpy vector store with cosine similarity + persistence
    retriever.py            # query -> top-k chunks
    llm.py                  # extractive (default), anthropic, openai providers
    guardrails.py           # grounding gate, citations, safe refusal
    pipeline.py             # RagPipeline tying it all together
    evaluation.py           # recall@k + keyword-score harness
    cli.py                  # `rag ingest|query|eval|anonymize`
  tests/                    # pytest suite (passes offline)
```

## License

MIT. Copyright (c) 2026 Olivier Dehareng. See [LICENSE](LICENSE).
