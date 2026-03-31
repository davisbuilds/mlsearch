from __future__ import annotations

import argparse
from typing import Callable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arxiv-cslg-search",
        description="Local-first semantic paper search over arXiv cs.LG.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_corpus_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_index_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_train_parser(subparsers)
    _add_search_parser(subparsers)

    return parser


def _add_corpus_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("corpus", help="Build and validate the paper corpus.")
    nested = parser.add_subparsers(dest="corpus_command", required=True)

    build = nested.add_parser("build", help="Build the local cs.LG corpus.")
    build.add_argument("--limit", type=int, default=5000, help="Target paper count.")
    build.set_defaults(handler=_placeholder_handler("corpus build"))

    validate = nested.add_parser("validate", help="Validate the local corpus artifact.")
    validate.set_defaults(handler=_placeholder_handler("corpus validate"))


def _add_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("benchmark", help="Generate and review benchmark queries.")
    nested = parser.add_subparsers(dest="benchmark_command", required=True)

    generate = nested.add_parser("generate", help="Generate synthetic benchmark queries.")
    generate.set_defaults(handler=_placeholder_handler("benchmark generate"))

    review = nested.add_parser("sample-review", help="Sample queries for manual review.")
    review.add_argument("--count", type=int, default=30, help="Review sample size.")
    review.set_defaults(handler=_placeholder_handler("benchmark sample-review"))


def _add_index_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("index", help="Build or inspect the retrieval index.")
    nested = parser.add_subparsers(dest="index_command", required=True)

    build = nested.add_parser("build", help="Build the local retrieval index.")
    build.set_defaults(handler=_placeholder_handler("index build"))


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Run baseline and comparison evaluations.")
    nested = parser.add_subparsers(dest="eval_command", required=True)

    baseline = nested.add_parser("baseline", help="Run the zero-shot baseline evaluation.")
    baseline.set_defaults(handler=_placeholder_handler("eval baseline"))

    compare = nested.add_parser("compare", help="Compare a trained model to baseline.")
    compare.add_argument("--model", default="latest", help="Model checkpoint alias.")
    compare.add_argument(
        "--record-results",
        action="store_true",
        help="Append the comparison outcome to results.tsv.",
    )
    compare.set_defaults(handler=_placeholder_handler("eval compare"))


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train a local retriever.")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config path.")
    parser.set_defaults(handler=_placeholder_handler("train"))


def _add_search_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("search", help="Search the local paper index.")
    parser.add_argument("query", help="Free-form semantic search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--format",
        default="table",
        choices=("table", "json"),
        help="Output format.",
    )
    parser.set_defaults(handler=_placeholder_handler("search"))


def _placeholder_handler(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(_: argparse.Namespace) -> int:
        raise SystemExit(f"{name} is not implemented yet.")

    return handler


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
