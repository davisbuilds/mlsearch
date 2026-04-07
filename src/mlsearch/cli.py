from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from mlsearch.paths import PATHS
from mlsearch.pipelines.archive_review_artifacts import archive_review_artifacts
from mlsearch.pipelines.build_corpus import build_corpus
from mlsearch.pipelines.finalize_review_set import finalize_review_set
from mlsearch.pipelines.generate_queries import compute_query_diagnostics, generate_queries, load_query_candidates
from mlsearch.pipelines.review_workflow import review_loop, review_next, review_stats
from mlsearch.pipelines.sample_review_set import sample_review_set
from mlsearch.pipelines.validate_corpus import validate_corpus


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlsearch",
        description="Local-first semantic paper search over arXiv cs.LG.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_corpus_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_index_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_experiment_parser(subparsers)
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
    review.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to sample into.")
    review.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Allow queries that already appear in archived review batches or the current held-out eval.",
    )
    review.set_defaults(handler=_handle_benchmark_sample_review)

    finalize = nested.add_parser("finalize-review", help="Finalize the reviewed held-out eval split.")
    finalize.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "reviewed" / "review_sample.csv"),
        help="Reviewed CSV to import into the held-out eval split.",
    )
    finalize.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to finalize into.")
    finalize.set_defaults(handler=_handle_benchmark_finalize_review)

    archive = nested.add_parser("archive-reviewed", help="Archive the current review artifacts under a label.")
    archive.add_argument("--label", default=None, help="Archive label. Defaults to a UTC timestamp.")
    archive.set_defaults(handler=_handle_benchmark_archive_reviewed)

    diagnostics = nested.add_parser("diagnostics", help="Summarize overlap diagnostics for generated benchmark queries.")
    diagnostics.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "generated" / "query_candidates.jsonl"),
        help="Generated query candidate JSONL to inspect.",
    )
    diagnostics.set_defaults(handler=_handle_benchmark_diagnostics)

    stats = nested.add_parser("review-stats", help="Show progress counts for the review CSV.")
    stats.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "reviewed" / "review_sample.csv"),
        help="Reviewed CSV to summarize.",
    )
    stats.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to inspect.")
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
    next_item.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to inspect.")
    next_item.set_defaults(handler=_handle_benchmark_review_next)

    review_loop_parser = nested.add_parser(
        "review-loop",
        help="Interactively review rows and persist decisions back to the CSV.",
    )
    review_loop_parser.add_argument(
        "--input",
        default=str(PATHS.data_benchmark / "reviewed" / "review_sample.csv"),
        help="Reviewed CSV to edit in place.",
    )
    review_loop_parser.add_argument(
        "--query-id",
        default=None,
        help="Start from a specific query id instead of the next pending row.",
    )
    review_loop_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process before exiting.",
    )
    review_loop_parser.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to edit.")
    review_loop_parser.set_defaults(handler=_handle_benchmark_review_loop)


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
    baseline.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to evaluate.")
    baseline.set_defaults(handler=_handle_eval_baseline)

    baseline_rerank = nested.add_parser("baseline-rerank", help="Run baseline retrieval plus second-stage reranking.")
    baseline_rerank.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name or local path.",
    )
    baseline_rerank.add_argument(
        "--rerank-depth",
        type=int,
        default=10,
        help="How many retrieved papers to rerank per query.",
    )
    baseline_rerank.add_argument("--top-k", type=int, default=10, help="Eval cutoff after reranking.")
    baseline_rerank.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to evaluate.")
    baseline_rerank.set_defaults(handler=_handle_eval_baseline_rerank)

    compare = nested.add_parser("compare", help="Compare a trained model to baseline.")
    compare.add_argument("--model", default="latest", help="Model checkpoint alias.")
    compare.add_argument("--split", choices=("dev", "test"), default="dev", help="Reviewed split to evaluate.")
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


def _add_experiment_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("experiment", help="Run constrained local autoresearch loops.")
    nested = parser.add_subparsers(dest="experiment_command", required=True)

    sweep = nested.add_parser("sweep", help="Sweep safe training hyperparameters against the reviewed eval split.")
    sweep.add_argument("--config", default="configs/train.yaml", help="Base training config path.")
    sweep.add_argument(
        "--reference-model",
        default="baseline",
        help="Starting champion for the sweep: baseline, latest, or a model checkpoint name.",
    )
    sweep.add_argument("--learning-rate", dest="learning_rates", type=float, nargs="+", default=None)
    sweep.add_argument("--num-epochs", dest="num_epochs", type=int, nargs="+", default=None)
    sweep.add_argument("--batch-size", dest="batch_sizes", type=int, nargs="+", default=None)
    sweep.add_argument("--max-examples", dest="max_examples", type=int, nargs="+", default=None)
    sweep.add_argument("--seed", dest="seeds", type=int, nargs="+", default=None)
    sweep.add_argument(
        "--record-results",
        action="store_true",
        help="Append each sweep run to results.tsv.",
    )
    sweep.set_defaults(handler=_handle_experiment_sweep)

    rerank = nested.add_parser("rerank", help="Run a second-stage reranking experiment on the reviewed eval split.")
    rerank.add_argument("--retriever-model", default="latest", help="Retriever checkpoint alias for first-stage recall.")
    rerank.add_argument(
        "--reference-model",
        default="latest",
        help="Reference system to compare against: baseline, latest, or a model checkpoint name.",
    )
    rerank.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name or local path.",
    )
    rerank.add_argument("--rerank-depth", type=int, default=10, help="How many retrieved papers to rerank per query.")
    rerank.add_argument("--top-k", type=int, default=10, help="Eval cutoff after reranking.")
    rerank.add_argument(
        "--record-results",
        action="store_true",
        help="Append the rerank outcome to results.tsv.",
    )
    rerank.set_defaults(handler=_handle_experiment_rerank)


def _add_search_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("search", help="Search the local paper index.")
    parser.add_argument("query", help="Free-form semantic search query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply the default second-stage cross-encoder reranker to the retrieved shortlist.",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name or local path used when --rerank is enabled.",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=10,
        help="How many retrieved papers to rerank when --rerank is enabled.",
    )
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
        include_reviewed=args.include_reviewed,
        split=args.split,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _resolve_review_input_path(input_value: str, *, split: str) -> Path | None:
    default_dev_path = str(PATHS.data_benchmark / "reviewed" / "review_sample.csv")
    if input_value == default_dev_path and split != "dev":
        return None
    return PATHS.root / input_value if not Path(input_value).is_absolute() else Path(input_value)


def _handle_benchmark_finalize_review(args: argparse.Namespace) -> int:
    report = finalize_review_set(
        review_path=_resolve_review_input_path(args.input, split=args.split),
        split=args.split,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_archive_reviewed(args: argparse.Namespace) -> int:
    report = archive_review_artifacts(label=args.label)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_diagnostics(args: argparse.Namespace) -> int:
    input_path = PATHS.root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    candidates = load_query_candidates(input_path)
    report = compute_query_diagnostics(candidates)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_review_stats(args: argparse.Namespace) -> int:
    review_path = _resolve_review_input_path(args.input, split=args.split)
    report = review_stats(review_path=review_path, split=args.split)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_review_next(args: argparse.Namespace) -> int:
    review_path = _resolve_review_input_path(args.input, split=args.split)
    try:
        report = review_next(review_path=review_path, query_id=args.query_id, split=args.split)
    except ValueError as exc:
        raise SystemExit(f"Review lookup failed: {exc}") from exc
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_benchmark_review_loop(args: argparse.Namespace) -> int:
    review_path = _resolve_review_input_path(args.input, split=args.split)
    try:
        report = review_loop(review_path=review_path, query_id=args.query_id, limit=args.limit, split=args.split)
    except ValueError as exc:
        raise SystemExit(f"Interactive review failed: {exc}") from exc
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_index_build(args: argparse.Namespace) -> int:
    from mlsearch.retrieval.index import build_index

    report = build_index(model_name=args.model)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_eval_baseline(args: argparse.Namespace) -> int:
    from mlsearch.eval.run_eval import run_baseline_eval

    try:
        report = run_baseline_eval(split=args.split)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing retrieval artifacts: {exc}") from exc
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_eval_baseline_rerank(args: argparse.Namespace) -> int:
    from mlsearch.eval.run_eval import run_baseline_rerank_eval

    try:
        report = run_baseline_rerank_eval(
            reranker_model_name=args.reranker_model,
            rerank_depth=args.rerank_depth,
            top_k=args.top_k,
            split=args.split,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing retrieval artifacts: {exc}") from exc
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_eval_compare(args: argparse.Namespace) -> int:
    from mlsearch.eval.run_eval import run_compare_eval

    report = run_compare_eval(model_ref=args.model, record_results=args.record_results, split=args.split)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    from mlsearch.training.train_retriever import train_retriever

    report = train_retriever(config_path=PATHS.root / args.config)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_experiment_sweep(args: argparse.Namespace) -> int:
    from mlsearch.experiments.sweep import run_experiment_sweep

    report = run_experiment_sweep(
        config_path=PATHS.root / args.config,
        reference_model=args.reference_model,
        learning_rates=args.learning_rates,
        num_epochs=args.num_epochs,
        batch_sizes=args.batch_sizes,
        max_examples=args.max_examples,
        seeds=args.seeds,
        record_results=args.record_results,
    )
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))
    return 0


def _handle_experiment_rerank(args: argparse.Namespace) -> int:
    from mlsearch.eval.run_eval import run_rerank_experiment

    report = run_rerank_experiment(
        retriever_model_ref=args.retriever_model,
        reference_model_ref=args.reference_model,
        reranker_model_name=args.reranker_model,
        rerank_depth=args.rerank_depth,
        top_k=args.top_k,
        record_results=args.record_results,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    from mlsearch.present.search_output import render_hits
    from mlsearch.retrieval.search import search_index

    try:
        hits = search_index(
            args.query,
            top_k=args.top_k,
            reranker_model_name=args.reranker_model if args.rerank else None,
            rerank_depth=args.rerank_depth,
        )
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
