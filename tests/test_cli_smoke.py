from __future__ import annotations

import pytest

from mlsearch import cli


def test_help_renders(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "corpus" in output
    assert "benchmark" in output
    assert "experiment" in output
    assert "search" in output


def test_search_parser_accepts_arguments() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["search", "test query", "--top-k", "3", "--format", "json"])
    assert args.command == "search"
    assert args.query == "test query"
    assert args.top_k == 3
    assert args.format == "json"


def test_benchmark_finalize_review_parser_accepts_input_override() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["benchmark", "finalize-review", "--input", "data/benchmark/reviewed/review_sample.csv"])
    assert args.command == "benchmark"
    assert args.benchmark_command == "finalize-review"
    assert args.input == "data/benchmark/reviewed/review_sample.csv"


def test_benchmark_review_stats_parser_accepts_input_override() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["benchmark", "review-stats", "--input", "data/benchmark/reviewed/review_sample.csv"])
    assert args.command == "benchmark"
    assert args.benchmark_command == "review-stats"
    assert args.input == "data/benchmark/reviewed/review_sample.csv"


def test_benchmark_review_next_parser_accepts_query_id_override() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "review-next",
            "--input",
            "data/benchmark/reviewed/review_sample.csv",
            "--query-id",
            "paper-1-question",
        ]
    )
    assert args.command == "benchmark"
    assert args.benchmark_command == "review-next"
    assert args.input == "data/benchmark/reviewed/review_sample.csv"
    assert args.query_id == "paper-1-question"


def test_benchmark_review_loop_parser_accepts_limit_and_query_id() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "benchmark",
            "review-loop",
            "--input",
            "data/benchmark/reviewed/review_sample.csv",
            "--query-id",
            "paper-1-question",
            "--limit",
            "3",
        ]
    )
    assert args.command == "benchmark"
    assert args.benchmark_command == "review-loop"
    assert args.input == "data/benchmark/reviewed/review_sample.csv"
    assert args.query_id == "paper-1-question"
    assert args.limit == 3


def test_benchmark_archive_reviewed_parser_accepts_label() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["benchmark", "archive-reviewed", "--label", "pre-hardening"])
    assert args.command == "benchmark"
    assert args.benchmark_command == "archive-reviewed"
    assert args.label == "pre-hardening"


def test_benchmark_diagnostics_parser_accepts_input_override() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["benchmark", "diagnostics", "--input", "data/benchmark/generated/query_candidates.jsonl"])
    assert args.command == "benchmark"
    assert args.benchmark_command == "diagnostics"
    assert args.input == "data/benchmark/generated/query_candidates.jsonl"


def test_experiment_sweep_parser_accepts_grid_arguments() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "sweep",
            "--config",
            "configs/train.yaml",
            "--reference-model",
            "latest",
            "--learning-rate",
            "1e-5",
            "2e-5",
            "--num-epochs",
            "1",
            "2",
            "--batch-size",
            "4",
            "--max-examples",
            "1000",
            "2000",
            "--seed",
            "42",
            "43",
            "--record-results",
        ]
    )
    assert args.command == "experiment"
    assert args.experiment_command == "sweep"
    assert args.reference_model == "latest"
    assert args.learning_rates == [1e-5, 2e-5]
    assert args.num_epochs == [1, 2]
    assert args.batch_sizes == [4]
    assert args.max_examples == [1000, 2000]
    assert args.seeds == [42, 43]
    assert args.record_results is True


def test_experiment_rerank_parser_accepts_reference_and_reranker_options() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "rerank",
            "--retriever-model",
            "latest",
            "--reference-model",
            "latest",
            "--reranker-model",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--rerank-depth",
            "10",
            "--top-k",
            "10",
            "--record-results",
        ]
    )
    assert args.command == "experiment"
    assert args.experiment_command == "rerank"
    assert args.retriever_model == "latest"
    assert args.reference_model == "latest"
    assert args.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert args.rerank_depth == 10
    assert args.top_k == 10
    assert args.record_results is True
