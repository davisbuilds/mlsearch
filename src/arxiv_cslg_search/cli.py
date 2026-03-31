from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from arxiv_cslg_search.paths import PATHS
from arxiv_cslg_search.pipelines.build_corpus import build_corpus
from arxiv_cslg_search.pipelines.finalize_review_set import finalize_review_set
from arxiv_cslg_search.pipelines.generate_queries import generate_queries
from arxiv_cslg_search.pipelines.review_workflow import review_next, review_stats
from arxiv_cslg_search.pipelines.sample_review_set import sample_review_set
from arxiv_cslg_search.pipelines.validate_corpus import validate_corpus


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
    build.set_defaults(handler=_handle_corpus_build)

    validate = nested.add_parser("validate", help="Validate the local corpus artifact.")
    validate.set_defaults(handler=_handle_corpus_validate)


def _add_benchmark_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("benchmark", help="Generate and review benchmark queries.")
    nested = parser.add_subparsers(dest="benchmark_command", required=True)

    generate = nested.add_parser("generate", help="Generate synthetic benchmark queries.")
    generate.set_defaults(handler=_handle_benchmark_generate)

    review = nested.add_parser("sample-review", help="Sample queries for manual review.")
    review.add_argument("--count", type=int, default=30, help="Review sample size.")
    review.set_defaults(handler=_handle_benchmark_sample_review)

    finalize = nested.add_parser("finalize-review", help="Finalize the reviewed held-out eval split.")
    finalize.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "reviewed" / "review_sample.csv"),
        help="Reviewed CSV to import into the held-out eval split.",
    )
    finalize.set_defaults(handler=_handle_benchmark_finalize_review)

    stats = nested.add_parser("review-stats", help="Show progress counts for the review CSV.")
    stats.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "reviewed" / "review_sample.csv"),
        help="Reviewed CSV to summarize.",
    )
    stats.set_defaults(handler=_handle_benchmark_review_stats)

    next_item = nested.add_parser("review-next", help="Show the next pending review row with source-paper context.")
    next_item.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "reviewed" / "review_sample.csv"),
        help="Reviewed CSV to inspect.",
    )
    next_item.add_argument(
        "--query-id",
        default=None,
        help="Inspect a specific query id instead of the next pending row.",
    )
    next_item.set_defaults(handler=_handle_benchmark_review_next)


def _add_index_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("index", help="Build or inspect the retrieval index.")
    nested = parser.add_subparsers(dest="index_command", required=True)

    build = nested.add_parser("build", help="Build the local retrieval index.")
    build.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformer model name.",
    )
    build.set_defaults(handler=_handle_index_build)


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval", help="Run baseline and comparison evaluations.")
    nested = parser.add_subparsers(dest="eval_command", required=True)

    baseline = nested.add_parser("baseline", help="Run the zero-shot baseline evaluation.")
    baseline.set_defaults(handler=_handle_eval_baseline)

    compare = nested.add_parser("compare", help="Compare a trained model to baseline.")
    compare.add_argument("--model", default="latest", help="Model checkpoint alias.")
    compare.add_argument(
        "--record-results",
        action="store_true",
        help="Append the comparison outcome to results.tsv.",
    )
    compare.set_defaults(handler=_handle_eval_compare)


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train a local retriever.")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config path.")
    parser.set_defaults(handler=_handle_train)


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
    parser.set_defaults(handler=_handle_search)


def _placeholder_handler(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(_: argparse.Namespace) -> int:
        raise SystemExit(f"{name} is not implemented yet.")

    return handler


def _handle_corpus_build(args: argparse.Namespace) -> int:
    report = build_corpus(
        config_path=PATHS.configs / "corpus.yaml",
        limit_override=args.limit,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_corpus_validate(_: argparse.Namespace) -> int:
    report = validate_corpus(config_path=PATHS.configs / "corpus.yaml")
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0 if report.valid else 1


def _handle_benchmark_generate(_: argparse.Namespace) -> int:
    report = generate_queries(config_path=PATHS.configs / "benchmark.yaml")
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_sample_review(args: argparse.Namespace) -> int:
    report = sample_review_set(
        config_path=PATHS.configs / "benchmark.yaml",
        count=args.count,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_finalize_review(args: argparse.Namespace) -> int:
    report = finalize_review_set(review_path=PATHS.root / args.input if not Path(args.input).is_absolute() else Path(args.input))
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_review_stats(args: argparse.Namespace) -> int:
    review_path = PATHS.root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    report = review_stats(review_path=review_path)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_review_next(args: argparse.Namespace) -> int:
    review_path = PATHS.root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    try:
        report = review_next(review_path=review_path, query_id=args.query_id)
    except ValueError as exc:
        raise SystemExit(f"Review lookup failed: {exc}") from exc
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_index_build(args: argparse.Namespace) -> int:
    from arxiv_cslg_search.retrieval.index import build_index

    report = build_index(model_name=args.model)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_eval_baseline(_: argparse.Namespace) -> int:
    from arxiv_cslg_search.eval.run_eval import run_baseline_eval

    try:
        report = run_baseline_eval()
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing retrieval artifacts: {exc}") from exc
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_eval_compare(args: argparse.Namespace) -> int:
    from arxiv_cslg_search.eval.run_eval import run_compare_eval

    report = run_compare_eval(model_ref=args.model, record_results=args.record_results)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    from arxiv_cslg_search.training.train_retriever import train_retriever

    report = train_retriever(config_path=PATHS.root / args.config)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    from arxiv_cslg_search.present.search_output import render_hits
    from arxiv_cslg_search.retrieval.search import search_index

    try:
        hits = search_index(args.query, top_k=args.top_k)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing retrieval artifacts: {exc}") from exc
    print(render_hits(hits, output_format=args.format))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
