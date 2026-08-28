"""Document loading and chunking.

Loads ``.txt`` and ``.md`` files from a folder, lightly cleans their text, and
splits them into overlapping chunks suitable for embedding and retrieval.

Why character-based chunking with overlap?

- It is dependency-free and deterministic (no tokenizer needed), which keeps the
  default path fully offline.
- Overlap preserves context that would otherwise be cut mid-sentence at a chunk
  boundary, improving retrieval recall for facts that straddle the boundary.

A real production system might use a token-aware splitter; the boundary logic
here is intentionally simple but tries to break on whitespace rather than mid-word.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# File extensions we know how to read as plain text.
SUPPORTED_EXTENSIONS = (".txt", ".md")


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance.

    Attributes:
        text: The chunk's cleaned text content.
        source: Path (as a string) of the document the chunk came from.
        chunk_index: 0-based position of this chunk within its source document.
        metadata: Free-form provenance attached by later stages, and serialized
            to JSON with the index. The engine itself only ever writes a ``pii``
            entry holding per-type redaction counts.
    """

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure.

    Collapses runs of spaces/tabs, trims trailing spaces on each line, and
    reduces 3+ consecutive newlines to a maximum of two (one blank line).
    """
    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces and tabs (but not newlines) into a single space.
    text = re.sub(r"[ \t]+", " ", text)
    # Trim trailing spaces per line.
    text = re.sub(r" *\n", "\n", text)
    # Collapse 3+ newlines into a paragraph break (two newlines).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` characters.

    The splitter advances by ``chunk_size - chunk_overlap`` each step so that
    consecutive chunks share ``chunk_overlap`` characters of context. Where
    possible, it nudges the cut point back to the last whitespace inside the
    window so chunks do not end mid-word.

    Args:
        text: The (already cleaned) text to split.
        chunk_size: Target maximum chunk length in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of non-empty chunk strings. Returns ``[]`` for empty input.

    Raises:
        ValueError: If ``chunk_size <= 0`` or ``chunk_overlap >= chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # Try to avoid cutting mid-word: back up to the last whitespace in the
        # window, as long as that does not shrink the chunk too aggressively.
        if end < n:
            window = text[start:end]
            last_space = window.rfind(" ")
            last_newline = window.rfind("\n")
            cut = max(last_space, last_newline)
            # Only honor the break if it keeps at least half the target size,
            # otherwise we'd produce tiny chunks for space-poor text.
            if cut >= step // 2:
                end = start + cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break
        start = end - chunk_overlap if end - chunk_overlap > start else end

    return chunks


def iter_document_paths(folder: str | Path) -> Iterable[Path]:
    """Yield supported document paths under ``folder``, sorted for determinism.

    Args:
        folder: Directory to scan recursively.

    Raises:
        FileNotFoundError: If the folder does not exist.
        NotADirectoryError: If the path exists but is not a directory.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    paths = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def load_and_chunk(
    folder: str | Path,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Load every supported document in ``folder`` and return their chunks.

    Args:
        folder: Directory containing ``.txt`` / ``.md`` files (scanned recursively).
        chunk_size: Target chunk length in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        A flat list of :class:`Chunk` objects across all documents.
    """
    chunks: list[Chunk] = []
    for path in iter_document_paths(folder):
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw)
        for i, piece in enumerate(
            chunk_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                Chunk(text=piece, source=str(path), chunk_index=i)
            )
    return chunks
