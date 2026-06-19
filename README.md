# rag-engine

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
- **Grounding guardrails.** Retrieval scores gate generation: if no chunk clears
  a similarity threshold, the engine refuses instead of guessing. When it does
  answer, every supporting chunk is attached as a citation, so answers are
  auditable.
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

## Configuration

All settings are read from environment variables (see `.env.example`). Secrets
are never read at import time: API keys are read lazily, only when an
API-backed provider is actually called.

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_EMBEDDER` | `hashing` | `hashing` (offline) or `sentence-transformers` (optional). |
| `RAG_LLM_PROVIDER` | `extractive` | `extractive` (offline), `anthropic`, or `openai`. |
| `ANTHROPIC_API_KEY` | (unset) | Optional. Only used by the `anthropic` provider. |
| `OPENAI_API_KEY` | (unset) | Optional. Only used by the `openai` provider. |
| `RAG_TOP_K` | `4` | Number of chunks retrieved per query. |
| `RAG_CHUNK_SIZE` | `600` | Target chunk size in characters. |
| `RAG_CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks, in characters. |
| `RAG_SIMILARITY_THRESHOLD` | `0.15` | Minimum cosine similarity for a chunk to count as evidence. |
| `RAG_EMBEDDING_DIM` | `256` | Vector dimension for the hashing embedder. |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name for the sentence-transformers embedder. |
| `RAG_LLM_MODEL` | `claude-opus-4-8` | Model name for the anthropic/openai providers. |
| `RAG_INDEX_DIR` | `.rag_index` | Directory where the vector index is saved. |

Optional extras (only needed if you switch providers):

```bash
pip install -e ".[embeddings]"  # sentence-transformers
pip install -e ".[llm]"         # anthropic + openai
pip install -e ".[dev]"         # pytest
```

The embedder and LLM providers are entirely optional. With the default
configuration, `rag-engine` runs fully offline using only numpy and the standard
library.

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
    config.py               # env-driven configuration dataclass
    ingestion.py            # load + clean + chunk documents
    embeddings.py           # hashing (default) and sentence-transformers embedders
    vectorstore.py          # numpy vector store with cosine similarity + persistence
    retriever.py            # query -> top-k chunks
    llm.py                  # extractive (default), anthropic, openai providers
    guardrails.py           # grounding gate, citations, safe refusal
    pipeline.py             # RagPipeline tying it all together
    evaluation.py           # recall@k + keyword-score harness
    cli.py                  # `rag ingest|query|eval`
  tests/                    # pytest suite (passes offline)
```

## License

MIT. Copyright (c) 2026 Olivier Dehareng. See [LICENSE](LICENSE).
