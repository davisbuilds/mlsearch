from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path

from arxiv_cslg_search.config import CorpusConfig, load_corpus_config
from arxiv_cslg_search.data.arxiv_client import ArxivClient, DateWindow, build_search_query
from arxiv_cslg_search.data.models import ArxivPaper
from arxiv_cslg_search.paths import PATHS


@dataclass(frozen=True)
class CorpusBuildReport:
    corpus_path: str
    manifest_path: str
    count: int
    windows: list[dict[str, object]]
    selection_strategy: str


def build_corpus(*, config_path: Path, limit_override: int | None = None) -> CorpusBuildReport:
    config = load_corpus_config(config_path)
    if limit_override is not None:
        config = replace(config, target_size=limit_override)

    raw_root = PATHS.data_raw / "arxiv"
    processed_root = PATHS.data_processed
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    windows = build_year_windows(config.start_date, config.end_date)
    quotas = allocate_quotas(config.target_size, len(windows))
    client = ArxivClient(page_size=config.page_size, delay_seconds=config.delay_seconds)

    papers: list[ArxivPaper] = []
    seen_ids: set[str] = set()
    window_reports: list[dict[str, object]] = []

    for window, quota in zip(windows, quotas, strict=True):
        query = build_search_query(config.category, window.start_date, window.end_date)
        fetched = client.fetch_papers(
            search_query=query,
            limit=quota,
            raw_dir=raw_root / window.label,
            sort_by="submittedDate",
            sort_order="descending",
        )
        unique_window_papers: list[ArxivPaper] = []
        for paper in fetched:
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)
            unique_window_papers.append(paper)
        papers.extend(unique_window_papers)
        window_reports.append(
            {
                "label": window.label,
                "start_date": window.start_date,
                "end_date": window.end_date,
                "requested": quota,
                "fetched": len(unique_window_papers),
            }
        )

    papers = sorted(papers, key=lambda paper: (paper.published, paper.arxiv_id), reverse=True)
    corpus_path = processed_root / "corpus.jsonl"
    manifest_path = processed_root / "corpus_manifest.json"
    write_corpus(corpus_path, papers)
    manifest_path.write_text(
        json.dumps(
            {
                "category": config.category,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "target_size": config.target_size,
                "actual_size": len(papers),
                "selection_strategy": config.selection_strategy,
                "windows": window_reports,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return CorpusBuildReport(
        corpus_path=str(corpus_path),
        manifest_path=str(manifest_path),
        count=len(papers),
        windows=window_reports,
        selection_strategy=config.selection_strategy,
    )


def build_year_windows(start_date: str, end_date: str) -> list[DateWindow]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows: list[DateWindow] = []
    current = start
    while current <= end:
        next_year = current.replace(year=current.year + 1)
        window_end = min(next_year - timedelta(days=1), end)
        windows.append(
            DateWindow(
                label=f"{current.isoformat()}_to_{window_end.isoformat()}",
                start_date=current.isoformat(),
                end_date=window_end.isoformat(),
            )
        )
        current = next_year
    return windows


def allocate_quotas(total: int, count: int) -> list[int]:
    base = total // count
    remainder = total % count
    quotas = [base] * count
    for index in range(remainder):
        quotas[index] += 1
    return quotas


def write_corpus(path: Path, papers: list[ArxivPaper]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.to_dict(), sort_keys=True) + "\n")
