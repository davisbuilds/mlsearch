from __future__ import annotations

from pathlib import Path

from arxiv_cslg_search.benchmark.review import REVIEW_COLUMNS, write_review_csv
from arxiv_cslg_search.benchmark.schema import QueryCandidate


def test_write_review_csv_creates_expected_columns(tmp_path: Path) -> None:
    candidate = QueryCandidate(
        query_id="paper-1-keyword",
        query_text="few-shot meta learning",
        style="keyword",
        source_paper_id="paper-1",
        source_title="Meta-learning for few-shot classification",
        source_published="2020-01-01T00:00:00Z",
        positive_ids=("paper-1",),
        hard_negative_ids=("paper-2",),
    )
    path = tmp_path / "review.csv"
    write_review_csv(path, [candidate])
    content = path.read_text(encoding="utf-8")
    for column in REVIEW_COLUMNS:
        assert column in content
