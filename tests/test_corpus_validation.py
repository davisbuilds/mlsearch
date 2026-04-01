from __future__ import annotations

from mlsearch.data.models import ArxivPaper
from mlsearch.pipelines.validate_corpus import validate_papers


def _paper(arxiv_id: str, *, abstract: str = "Abstract") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title="Title",
        abstract=abstract,
        authors=("Alice",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published="2020-01-01T00:00:00Z",
        updated="2020-01-01T00:00:00Z",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=None,
    )


def test_validate_papers_accepts_valid_records() -> None:
    errors = validate_papers(
        [_paper("1234.5678v1"), _paper("1234.5679v1")],
        "cs.LG",
        "2016-04-01",
        "2026-03-31",
    )
    assert errors == []


def test_validate_papers_reports_duplicates_and_empty_fields() -> None:
    errors = validate_papers(
        [_paper("1234.5678v1"), _paper("1234.5678v1", abstract="")],
        "cs.LG",
        "2016-04-01",
        "2026-03-31",
    )
    assert "Duplicate arXiv id: 1234.5678v1" in errors
    assert "Empty abstract: 1234.5678v1" in errors
