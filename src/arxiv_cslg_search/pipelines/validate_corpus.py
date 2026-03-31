from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from arxiv_cslg_search.config import load_corpus_config
from arxiv_cslg_search.data.models import ArxivPaper
from arxiv_cslg_search.paths import PATHS


@dataclass(frozen=True)
class CorpusValidationReport:
    corpus_path: str
    count: int
    valid: bool
    errors: list[str]


def validate_corpus(*, config_path: Path, corpus_path: Path | None = None) -> CorpusValidationReport:
    config = load_corpus_config(config_path)
    resolved_path = corpus_path or (PATHS.data_processed / "corpus.jsonl")
    papers = load_corpus(resolved_path)
    errors = validate_papers(papers, config.category, config.start_date, config.end_date)
    return CorpusValidationReport(
        corpus_path=str(resolved_path),
        count=len(papers),
        valid=not errors,
        errors=errors,
    )


def load_corpus(path: Path) -> list[ArxivPaper]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    papers: list[ArxivPaper] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            papers.append(ArxivPaper.from_dict(record))
    return papers


def validate_papers(
    papers: list[ArxivPaper],
    expected_category: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    for paper in papers:
        if paper.arxiv_id in seen_ids:
            errors.append(f"Duplicate arXiv id: {paper.arxiv_id}")
        seen_ids.add(paper.arxiv_id)

        if not paper.title.strip():
            errors.append(f"Empty title: {paper.arxiv_id}")
        if not paper.abstract.strip():
            errors.append(f"Empty abstract: {paper.arxiv_id}")
        if expected_category not in paper.categories:
            errors.append(f"Missing expected category {expected_category}: {paper.arxiv_id}")

        published_date = date.fromisoformat(paper.published[:10])
        if published_date < start or published_date > end:
            errors.append(
                f"Published date out of range for {paper.arxiv_id}: {paper.published[:10]}"
            )
    return errors
