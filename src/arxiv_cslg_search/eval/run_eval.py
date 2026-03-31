from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from arxiv_cslg_search.experiments.compare import compare_metric_sets
from arxiv_cslg_search.experiments.logging import append_result
from arxiv_cslg_search.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from arxiv_cslg_search.paths import PATHS
from arxiv_cslg_search.pipelines.generate_queries import load_query_candidates
from arxiv_cslg_search.retrieval.index import build_index
from arxiv_cslg_search.retrieval.search import search_many
from arxiv_cslg_search.training.checkpoints import latest_checkpoint


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
    metrics, report_path = _run_eval(
        candidates_path=candidate_path,
        index_dir=PATHS.artifacts_index,
        output_dir=output_dir or PATHS.artifacts_results,
        report_prefix="baseline",
        top_k=top_k,
    )
    return BaselineEvalReport(
        report_path=str(report_path),
        query_count=len(candidates),
        metrics=metrics,
    )


def run_compare_eval(*, model_ref: str, record_results: bool) -> dict[str, object]:
    checkpoint = latest_checkpoint() if model_ref == "latest" else (PATHS.artifacts_models / model_ref)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    compare_index_dir = PATHS.artifacts_index / checkpoint.name
    build_index(output_dir=compare_index_dir, model_name=str(checkpoint))
    candidate_path = PATHS.data_benchmark / "generated" / "query_candidates.jsonl"
    candidate_metrics, report_path = _run_eval(
        candidates_path=candidate_path,
        index_dir=compare_index_dir,
        output_dir=PATHS.artifacts_results,
        report_prefix="compare",
        top_k=10,
    )
    baseline_metrics = load_latest_metrics("baseline")
    comparison = compare_metric_sets(candidate_metrics, baseline_metrics)
    payload = {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "comparison": comparison,
        "model_ref": checkpoint.name,
        "report_path": str(report_path),
    }
    if record_results:
        append_result(
            PATHS.root / "results.tsv",
            model_ref=checkpoint.name,
            metrics=candidate_metrics,
            status=str(comparison["status"]),
            description="local fine-tuned compare run",
        )
    return payload


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


def _run_eval(*, candidates_path: Path, index_dir: Path, output_dir: Path, report_prefix: str, top_k: int) -> tuple[dict[str, float], Path]:
    candidates = load_query_candidates(candidates_path)
    hits_per_query = search_many(
        [candidate.query_text for candidate in candidates],
        top_k=top_k,
        index_dir=index_dir,
    )
    metrics = aggregate_metrics(candidates, hits_per_query, top_k=top_k)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"{report_prefix}-{timestamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "query_count": len(candidates),
                "metrics": metrics,
                "top_k": top_k,
                "index_dir": str(index_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return metrics, report_path


def load_latest_metrics(prefix: str) -> dict[str, float]:
    reports = sorted(PATHS.artifacts_results.glob(f"{prefix}-*.json"))
    if not reports:
        raise FileNotFoundError(f"No {prefix} reports found in {PATHS.artifacts_results}")
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    return payload["metrics"]
