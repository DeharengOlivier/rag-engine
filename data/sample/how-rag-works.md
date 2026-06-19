# How Retrieval-Augmented Generation Works

Retrieval-Augmented Generation, often shortened to RAG, is a pattern for
answering questions over a private collection of documents. Instead of relying
only on what a language model memorized during training, a RAG system fetches
relevant passages from your own documents and uses them as evidence.

## The main steps

A RAG pipeline usually has the same stages. First, documents are split into
small overlapping chunks. Each chunk is turned into a numeric vector by an
embedding model. The vectors are stored in a vector index. When a question
arrives, it is embedded the same way, and the index returns the chunks whose
vectors are most similar to the question.

## Grounding the answer

The retrieved chunks are passed to a generator together with the question. The
generator is instructed to answer using only the retrieved evidence. A good RAG
system also adds guardrails: if no chunk is similar enough to the question, the
system refuses to answer rather than inventing a response. Every answer should
carry citations pointing back to the source chunks, so a reader can verify it.

## Why it helps

Grounding answers in retrieved text reduces hallucination, keeps answers up to
date as documents change, and makes the system auditable because each claim can
be traced to a source.
