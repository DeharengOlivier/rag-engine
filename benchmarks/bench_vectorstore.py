"""Measure what the in-memory index costs, so the README table can be re-run.

A performance claim without a number is not allowed, and a number nobody can
reproduce is barely better. This produces the table in the README's "Scale, and
where this design stops" section:

    python benchmarks/bench_vectorstore.py
    python benchmarks/bench_vectorstore.py 1000 10000

Memory is the peak resident set size of a process that builds one index, minus
its own baseline. That is what decides whether a corpus fits: it includes the
transient copy `add` makes, not only the size of the final matrix. Peak RSS is a
high-water mark for the whole process, so each size is measured in a process of
its own, which is what the internal ``--single`` mode is for.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from rag_engine.ingestion import Chunk
from rag_engine.vectorstore import VectorStore

DEFAULT_SIZES = (10_000, 100_000, 500_000)
CHUNK_CHARACTERS = 596
QUERIES = 20
SEED = 0


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in megabytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def build_corpus(n: int, dim: int) -> tuple[np.ndarray, list[Chunk]]:
    """Return ``n`` normalized vectors and ``n`` distinct chunks of fixed length."""
    rng = np.random.default_rng(SEED)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    filler = "abcdefghij" * ((CHUNK_CHARACTERS - 6) // 10)
    chunks = [
        # Distinct texts: identical ones would be shared by the interpreter and
        # the memory figure would be a fiction.
        Chunk(text=f"{i:06d}{filler}", source=f"doc{i % 500}.md", chunk_index=i)
        for i in range(n)
    ]
    return vectors, chunks


def measure(n: int, dim: int) -> dict[str, float]:
    """Build an index of ``n`` chunks and measure memory, disk and query time.

    Must run in a process that has not already built one: peak RSS is a
    high-water mark and never goes back down.

    Complexity: O(n * dim) to build and O(n * dim) per query, both linear in n.
    """
    before = peak_rss_mb()
    vectors, chunks = build_corpus(n, dim)

    store = VectorStore(dim=dim)
    store.add(vectors, chunks)
    memory_mb = peak_rss_mb() - before
    del vectors

    query = np.random.default_rng(SEED + 1).standard_normal(dim).astype(np.float32)
    store.search(query, top_k=4)  # warm the caches
    started_at = time.perf_counter()
    for _ in range(QUERIES):
        store.search(query, top_k=4)
    query_ms = (time.perf_counter() - started_at) / QUERIES * 1000

    directory = Path(tempfile.mkdtemp()) / "index"
    try:
        store.save(directory)
        disk_mb = sum(f.stat().st_size for f in directory.iterdir()) / 1e6
    finally:
        shutil.rmtree(directory.parent, ignore_errors=True)

    return {"n": n, "memory_mb": memory_mb, "disk_mb": disk_mb, "query_ms": query_ms}


def measure_in_a_fresh_process(n: int, dim: int) -> dict[str, float]:
    """Run :func:`measure` for one size in a subprocess and return its result."""
    completed = subprocess.run(
        [sys.executable, __file__, "--single", str(n), "--dim", str(dim)],
        capture_output=True,
        text=True,
        check=True,
    )
    result: dict[str, float] = json.loads(completed.stdout)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sizes",
        nargs="*",
        type=int,
        default=list(DEFAULT_SIZES),
        help="Index sizes to measure, in chunks.",
    )
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument(
        "--single",
        type=int,
        default=None,
        help="Measure one size and emit JSON. Used internally: one size per process.",
    )
    args = parser.parse_args(argv)

    if args.single is not None:
        print(json.dumps(measure(args.single, args.dim)))
        return 0

    print("| Chunks | Memory | On disk | Query |")
    print("| --- | --- | --- | --- |")
    for n in sorted(args.sizes):
        row = measure_in_a_fresh_process(n, args.dim)
        print(
            f"| {row['n']:,} | ~{row['memory_mb']:.0f} MB | "
            f"{row['disk_mb']:.0f} MB | {row['query_ms']:.1f} ms |"
        )
    print(
        f"\n{CHUNK_CHARACTERS}-character chunks, dim {args.dim}, "
        f"mean of {QUERIES} queries, seed {SEED}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - a script, not an import
    raise SystemExit(main())
