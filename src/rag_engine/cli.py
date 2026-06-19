"""Command-line interface for the RAG engine.

Subcommands:

- ``rag ingest <folder>``   : index a folder of documents.
- ``rag query "<question>"``: ask a question against the index.
- ``rag eval <evalfile>``   : run the evaluation harness.

The CLI reads configuration from environment variables (see ``RagConfig``), so
the same defaults that make the library run offline also apply here.
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_engine.config import RagConfig
from rag_engine.evaluation import evaluate, load_eval_cases
from rag_engine.pipeline import RagPipeline


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Handle ``rag ingest``."""
    pipeline = RagPipeline(RagConfig.from_env())
    count = pipeline.ingest(args.folder)
    print(f"Indexed {count} chunk(s) from '{args.folder}'.")
    print(f"Index saved to '{pipeline.config.index_dir}'.")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``rag`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
