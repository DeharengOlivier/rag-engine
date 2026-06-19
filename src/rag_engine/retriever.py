"""Retrieval: turn a question into the top-k most relevant chunks.

The retriever is a thin coordinator over an :class:`~rag_engine.embeddings.Embedder`
and a :class:`~rag_engine.vectorstore.VectorStore`: it embeds the query with the
*same* embedder used at ingestion time, then asks the store for nearest neighbors.

Keeping this as its own component (rather than folding it into the pipeline)
makes it trivial to test retrieval quality in isolation and to reuse it for
evaluation (recall@k) without invoking an LLM.
"""

from __future__ import annotations

from rag_engine.embeddings import Embedder
from rag_engine.vectorstore import SearchResult, VectorStore


class Retriever:
    """Embed a query and fetch the most similar chunks from the vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` chunks most relevant to ``query``.

        Args:
            query: The user's question.
            top_k: Maximum number of chunks to return.

        Returns:
            Search results (chunk + score) sorted by descending similarity.
            Empty for a blank query or empty store.
        """
        if not query or not query.strip():
            return []
        query_vector = self._embedder.embed([query])[0]
        return self._store.search(query_vector, top_k=top_k)
