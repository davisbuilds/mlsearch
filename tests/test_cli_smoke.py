from __future__ import annotations

import pytest

from arxiv_cslg_search import cli


def test_help_renders(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "corpus" in output
    assert "benchmark" in output
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
