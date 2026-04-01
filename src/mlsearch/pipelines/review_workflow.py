from __future__ import annotations

from pathlib import Path

from mlsearch.benchmark.review import (
    load_next_review_item,
    run_review_loop,
    summarize_review_progress,
)
from mlsearch.paths import PATHS


def review_stats(*, review_path: Path | None = None):
    resolved_review_path = review_path or (PATHS.data_benchmark / "reviewed" / "review_sample.csv")
    return summarize_review_progress(resolved_review_path)


def review_next(*, review_path: Path | None = None, query_id: str | None = None):
    resolved_review_path = review_path or (PATHS.data_benchmark / "reviewed" / "review_sample.csv")
    return load_next_review_item(
        review_path=resolved_review_path,
        corpus_path=PATHS.data_processed / "corpus.jsonl",
        query_id=query_id,
    )


def review_loop(*, review_path: Path | None = None, query_id: str | None = None, limit: int | None = None):
    resolved_review_path = review_path or (PATHS.data_benchmark / "reviewed" / "review_sample.csv")
    return run_review_loop(
        review_path=resolved_review_path,
        corpus_path=PATHS.data_processed / "corpus.jsonl",
        query_id=query_id,
        limit=limit,
    )
