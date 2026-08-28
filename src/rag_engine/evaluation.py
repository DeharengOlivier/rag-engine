"""A small evaluation harness for retrieval and answer quality.

Given a list of labeled questions, it computes two cheap, interpretable metrics:

- **Retrieval recall@k**: did the expected source document appear among the
  top-k retrieved chunks? This measures the retriever in isolation.
- **Answer keyword score**: does the generated answer contain the expected
  keywords? A crude but useful proxy for groundedness/correctness that needs no
  second model to grade it.

These are intentionally simple. A production system would add semantic graders
or human review, but recall@k and keyword coverage catch the most common
regressions (retrieval breaking, or answers drifting off-topic) with zero extra
dependencies, which fits the offline-first design.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_engine.pipeline import RagPipeline


@dataclass
class EvalCase:
    """One labeled evaluation example.

    Attributes:
        question: The question to ask.
        expected_source: Optional filename (basename) expected in retrieval.
        answer_keywords: Keywords expected to appear in the generated answer.
    """

    question: str
    expected_source: str | None = None
    answer_keywords: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregate metrics across all evaluation cases."""

    num_cases: int
    recall_at_k: float
    keyword_score: float
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def format(self) -> str:
        """Render a short human-readable report."""
        lines = [
            "Evaluation report",
            "=================",
            f"Cases:          {self.num_cases}",
            f"Recall@k:       {self.recall_at_k:.2%}",
            f"Keyword score:  {self.keyword_score:.2%}",
            "",
            "Per-case:",
        ]
        for case in self.per_case:
            status = "ok " if case["retrieval_hit"] else "MISS"
            lines.append(
                f"  [{status}] kw={case['keyword_hit']:.0%} "
                f"q={case['question'][:60]!r}"
            )
        return "\n".join(lines)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file.

    The file must be a JSON array of objects with keys ``question`` and
    optionally ``expected_source`` and ``answer_keywords``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=item["question"],
            expected_source=item.get("expected_source"),
            answer_keywords=item.get("answer_keywords", []),
        )
        for item in raw
    ]


def evaluate(pipeline: RagPipeline, cases: list[EvalCase]) -> EvalReport:
    """Run ``cases`` through ``pipeline`` and compute aggregate metrics.

    Args:
        pipeline: A pipeline whose corpus has already been ingested.
        cases: The labeled evaluation cases.

    Returns:
        An :class:`EvalReport` with recall@k, keyword score, and per-case detail.
    """
    if not cases:
        return EvalReport(num_cases=0, recall_at_k=0.0, keyword_score=0.0)

    recall_hits = 0
    keyword_total = 0.0
    per_case: list[dict[str, Any]] = []

    for case in cases:
        result = pipeline.answer(case.question)

        # Retrieval recall@k: was the expected source retrieved?
        retrieved_sources = {
            os.path.basename(r.chunk.source) for r in result.used_chunks
        }
        if case.expected_source is None:
            # No expected source given: count as a hit only if anything retrieved.
            retrieval_hit = len(result.used_chunks) > 0
        else:
            retrieval_hit = case.expected_source in retrieved_sources
        recall_hits += int(retrieval_hit)

        # Keyword score: fraction of expected keywords present in the answer.
        if case.answer_keywords:
            answer_lower = result.answer.lower()
            found = sum(
                1 for kw in case.answer_keywords if kw.lower() in answer_lower
            )
            keyword_hit = found / len(case.answer_keywords)
        else:
            keyword_hit = 1.0  # nothing to check
        keyword_total += keyword_hit

        per_case.append(
            {
                "question": case.question,
                "retrieval_hit": retrieval_hit,
                "keyword_hit": keyword_hit,
                "refused": result.refused,
            }
        )

    n = len(cases)
    return EvalReport(
        num_cases=n,
        recall_at_k=recall_hits / n,
        keyword_score=keyword_total / n,
        per_case=per_case,
    )
