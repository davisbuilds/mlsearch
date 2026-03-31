from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from arxiv_cslg_search.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from arxiv_cslg_search.paths import PATHS
from arxiv_cslg_search.pipelines.generate_queries import load_query_candidates
from arxiv_cslg_search.retrieval.search import search_many


@dataclass(frozen=True)
class BaselineEvalReport:
    report_path: str
    query_count: int
    metrics: dict[str, float]


def run_baseline_eval(
    *,
    candidates_path: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 10,
) -> BaselineEvalReport:
    candidate_path = candidates_path or (PATHS.data_benchmark / "generated" / "query_candidates.jsonl")
    candidates = load_query_candidates(candidate_path)
    hits_per_query = search_many([candidate.query_text for candidate in candidates], top_k=top_k)

    metrics = aggregate_metrics(candidates, hits_per_query, top_k=top_k)
    output_dir = output_dir or PATHS.artifacts_results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"baseline-{timestamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "query_count": len(candidates),
                "metrics": metrics,
                "top_k": top_k,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return BaselineEvalReport(
        report_path=str(report_path),
        query_count=len(candidates),
        metrics=metrics,
    )


def aggregate_metrics(candidates, hits_per_query, *, top_k: int) -> dict[str, float]:
    recall = 0.0
    mrr = 0.0
    ndcg = 0.0
    for candidate, hits in zip(candidates, hits_per_query, strict=True):
        result_ids = [hit.arxiv_id for hit in hits]
        relevant_ids = set(candidate.positive_ids)
        recall += recall_at_k(result_ids, relevant_ids, top_k)
        mrr += reciprocal_rank(result_ids, relevant_ids)
        ndcg += ndcg_at_k(result_ids, relevant_ids, top_k)
    total = len(candidates) or 1
    return {
        f"recall@{top_k}": recall / total,
        "mrr": mrr / total,
        f"ndcg@{top_k}": ndcg / total,
    }
