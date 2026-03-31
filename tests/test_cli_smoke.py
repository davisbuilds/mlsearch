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
