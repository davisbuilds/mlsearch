from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from arxiv_cslg_search.benchmark.review import write_review_csv
from arxiv_cslg_search.benchmark.schema import QueryCandidate
from arxiv_cslg_search.config import load_benchmark_config
from arxiv_cslg_search.paths import PATHS
from arxiv_cslg_search.pipelines.generate_queries import load_query_candidates


@dataclass(frozen=True)
class ReviewSampleReport:
    review_path: str
    count: int
    styles: dict[str, int]


def sample_review_set(*, config_path: Path, count: int) -> ReviewSampleReport:
    config = load_benchmark_config(config_path)
    candidates_path = PATHS.data_benchmark / "generated" / "query_candidates.jsonl"
    candidates = load_query_candidates(candidates_path)
    selected = stratified_sample(candidates, count=count, seed=config.seed)

    review_path = PATHS.data_benchmark / "reviewed" / "review_sample.csv"
    write_review_csv(review_path, selected)

    styles: dict[str, int] = defaultdict(int)
    for candidate in selected:
        styles[candidate.style] += 1
    return ReviewSampleReport(
        review_path=str(review_path),
        count=len(selected),
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
