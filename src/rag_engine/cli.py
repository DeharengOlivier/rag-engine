"""Command-line interface for the RAG engine.

Subcommands:

- ``rag ingest <folder>``    : index a folder of documents.
- ``rag query "<question>"`` : ask a question against the index.
- ``rag eval <evalfile>``    : run the evaluation harness.
- ``rag anonymize "<text>"`` : show the PII an anonymizer would redact.

The CLI reads configuration from environment variables (see ``RagConfig``), so
the same defaults that make the library run offline also apply here.
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_engine.anonymizer import build_anonymizer
from rag_engine.config import RagConfig
from rag_engine.evaluation import evaluate, load_eval_cases
from rag_engine.pipeline import RagPipeline


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Handle ``rag ingest``."""
    pipeline = RagPipeline(RagConfig.from_env())
    count = pipeline.ingest(args.folder)
    print(f"Indexed {count} chunk(s) from '{args.folder}'.")
    print(f"Index saved to '{pipeline.config.index_dir}'.")
    report = pipeline.last_pii_report
    if report:
        total = sum(report.values())
        breakdown = ", ".join(f"{t}={n}" for t, n in sorted(report.items()))
        print(
            f"Anonymized {total} PII entit{'y' if total == 1 else 'ies'} "
            f"before indexing ({breakdown})."
        )
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    """Handle ``rag query``."""
    pipeline = RagPipeline(RagConfig.from_env())
    try:
        result = pipeline.answer(args.question)
    except FileNotFoundError:
        print(
            "No index found. Run 'rag ingest <folder>' first.",
            file=sys.stderr,
        )
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(result.answer)
    if result.refused:
        return 0
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(f"  [{c.index}] {c.source} (score={c.score}) {c.snippet}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Handle ``rag eval``."""
    pipeline = RagPipeline(RagConfig.from_env())
    cases = load_eval_cases(args.evalfile)
    report = evaluate(pipeline, cases)
    print(report.format())
    return 0


def _cmd_anonymize(args: argparse.Namespace) -> int:
    """Handle ``rag anonymize``: show what an anonymizer redacts in a text."""
    config = RagConfig.from_env()
    # Default to the offline regex backend for this command unless the user has
    # explicitly selected one, so the demo works with no extra dependencies.
    if config.anonymizer == "none":
        config.anonymizer = "regex"
    anonymizer = build_anonymizer(config)
    result = anonymizer.anonymize(args.text)

    if args.json:
        print(
            json.dumps(
                {
                    "anonymizer": anonymizer.name,
                    "anonymized_text": result.text,
                    "entities": [
                        {
                            "type": e.entity_type,
                            "text": e.text,
                            "start": e.start,
                            "end": e.end,
                            "score": round(e.score, 4),
                        }
                        for e in result.entities
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"Backend: {anonymizer.name}")
    print(f"Anonymized: {result.text}")
    if result.entities:
        print("\nDetected:")
        for e in result.entities:
            print(f"  {e.entity_type}: '{e.text}' (score={round(e.score, 2)})")
    else:
        print("\nNo PII detected.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ``rag`` command."""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="A generic, offline-first Retrieval-Augmented Generation engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a folder of .txt/.md documents.")
    p_ingest.add_argument("folder", help="Folder containing documents to index.")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="Ask a question against the index.")
    p_query.add_argument("question", help="The question to ask.")
    p_query.add_argument(
        "--json", action="store_true", help="Emit the full result as JSON."
    )
    p_query.set_defaults(func=_cmd_query)

    p_eval = sub.add_parser("eval", help="Run the evaluation harness.")
    p_eval.add_argument("evalfile", help="Path to a JSON file of eval cases.")
    p_eval.set_defaults(func=_cmd_eval)

    p_anon = sub.add_parser(
        "anonymize", help="Show the PII an anonymizer would redact in a text."
    )
    p_anon.add_argument("text", help="The text to scan for PII.")
    p_anon.add_argument(
        "--json", action="store_true", help="Emit the full result as JSON."
    )
    p_anon.set_defaults(func=_cmd_anonymize)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rag`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
