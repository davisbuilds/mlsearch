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


def test_search_placeholder_fails_cleanly() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["search", "test query"])

    assert str(exc.value).startswith("Missing retrieval artifacts:")
