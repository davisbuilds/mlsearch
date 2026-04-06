from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mlsearch.benchmark.review import load_reviewed_queries, write_review_csv
from mlsearch.benchmark.schema import QueryCandidate
from mlsearch.config import load_benchmark_config
from mlsearch.paths import PATHS
from mlsearch.pipelines.generate_queries import load_query_candidates


@dataclass(frozen=True)
class ReviewSampleReport:
    review_path: str
    count: int
    excluded_count: int
    styles: dict[str, int]


def sample_review_set(*, config_path: Path, count: int, include_reviewed: bool = False) -> ReviewSampleReport:
    config = load_benchmark_config(config_path)
    candidates_path = PATHS.data_benchmark / "generated" / "query_candidates.jsonl"
    candidates = load_query_candidates(candidates_path)
    excluded_query_ids = set() if include_reviewed else load_reviewed_query_ids()
    excluded_source_paper_ids = set() if include_reviewed else load_reviewed_source_paper_ids()
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.query_id not in excluded_query_ids
        and candidate.source_paper_id not in excluded_source_paper_ids
    ]
    selected = stratified_sample(eligible_candidates, count=min(count, len(eligible_candidates)), seed=config.seed)

    review_path = PATHS.data_benchmark / "reviewed" / "review_sample.csv"
    write_review_csv(review_path, selected)

    styles: dict[str, int] = defaultdict(int)
    for candidate in selected:
        styles[candidate.style] += 1
    return ReviewSampleReport(
        review_path=str(review_path),
        count=len(selected),
        excluded_count=len(excluded_query_ids) + len(excluded_source_paper_ids),
        styles=dict(sorted(styles.items())),
    )


def stratified_sample(candidates: list[QueryCandidate], *, count: int, seed: int) -> list[QueryCandidate]:
    grouped: dict[str, list[QueryCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.style].append(candidate)

    if not grouped:
        return []

    rng = random.Random(seed)
    styles = sorted(grouped)
    quotas = {style: count // len(styles) for style in styles}
    for index in range(count % len(styles)):
        quotas[styles[index]] += 1

    selected: list[QueryCandidate] = []
    for style in styles:
        pool = grouped[style]
        if len(pool) <= quotas[style]:
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, quotas[style]))
    return sorted(selected, key=lambda candidate: candidate.query_id)


def load_reviewed_query_ids() -> set[str]:
    reviewed_ids: set[str] = set()

    held_out_path = PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl"
    if held_out_path.exists():
        reviewed_ids.update(query.query_id for query in load_reviewed_queries(held_out_path))

    archive_dir = PATHS.data_benchmark / "reviewed" / "archive"
    if archive_dir.exists():
        for path in archive_dir.glob("*/review_sample.csv"):
            reviewed_ids.update(_load_review_ids_from_csv(path))

    current_review_path = PATHS.data_benchmark / "reviewed" / "review_sample.csv"
    if current_review_path.exists():
        reviewed_ids.update(_load_review_ids_from_csv(current_review_path))

    return reviewed_ids


def load_reviewed_source_paper_ids() -> set[str]:
    reviewed_paper_ids: set[str] = set()

    held_out_path = PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl"
    if held_out_path.exists():
        reviewed_paper_ids.update(query.source_paper_id for query in load_reviewed_queries(held_out_path))

    archive_dir = PATHS.data_benchmark / "reviewed" / "archive"
    if archive_dir.exists():
        for path in archive_dir.glob("*/review_sample.csv"):
            reviewed_paper_ids.update(_load_source_paper_ids_from_csv(path))

    current_review_path = PATHS.data_benchmark / "reviewed" / "review_sample.csv"
    if current_review_path.exists():
        reviewed_paper_ids.update(_load_source_paper_ids_from_csv(current_review_path))

    return reviewed_paper_ids


def _load_review_ids_from_csv(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["query_id"]) for row in reader if row.get("query_id")}


def _load_source_paper_ids_from_csv(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["source_paper_id"]) for row in reader if row.get("source_paper_id")}
