from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlsearch.benchmark.review import load_reviewed_queries
from mlsearch.benchmark.schema import QueryCandidate, ReviewedQuery
from mlsearch.experiments.compare import compare_metric_sets
from mlsearch.experiments.logging import append_result
from mlsearch.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from mlsearch.paths import PATHS
from mlsearch.pipelines.generate_queries import load_query_candidates
from mlsearch.retrieval.index import build_index
from mlsearch.retrieval.search import search_many
from mlsearch.training.checkpoints import latest_checkpoint


@dataclass(frozen=True)
class BaselineEvalReport:
    report_path: str
    candidates_path: str
    query_count: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class ModelEvalReport:
    report_path: str
    candidates_path: str
    index_dir: str
    model_ref: str
    query_count: int
    metrics: dict[str, float]


def run_baseline_eval(
    *,
    candidates_path: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 10,
) -> BaselineEvalReport:
    selected_candidates_path, candidates = resolve_eval_candidates(
        generated_candidates_path=candidates_path or (PATHS.data_benchmark / "generated" / "query_candidates.jsonl"),
        reviewed_eval_path=PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl",
    )
    metrics, report_path = _run_eval(
        candidates=candidates,
        candidates_path=selected_candidates_path,
        index_dir=PATHS.artifacts_index,
        output_dir=output_dir or PATHS.artifacts_results,
        report_prefix="baseline",
        top_k=top_k,
    )
    return BaselineEvalReport(
        report_path=str(report_path),
        candidates_path=str(selected_candidates_path),
        query_count=len(candidates),
        metrics=metrics,
    )


def run_model_eval(*, model_ref: str | Path, top_k: int = 10) -> ModelEvalReport:
    checkpoint = _resolve_checkpoint(model_ref)
    compare_index_dir = PATHS.artifacts_index / checkpoint.name
    build_index(output_dir=compare_index_dir, model_name=str(checkpoint))
    candidate_path, candidates = resolve_eval_candidates(
        generated_candidates_path=PATHS.data_benchmark / "generated" / "query_candidates.jsonl",
        reviewed_eval_path=PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl",
    )
    candidate_metrics, report_path = _run_eval(
        candidates=candidates,
        candidates_path=candidate_path,
        index_dir=compare_index_dir,
        output_dir=PATHS.artifacts_results,
        report_prefix="compare",
        top_k=top_k,
    )
    return ModelEvalReport(
        report_path=str(report_path),
        candidates_path=str(candidate_path),
        index_dir=str(compare_index_dir),
        model_ref=checkpoint.name,
        query_count=len(candidates),
        metrics=candidate_metrics,
    )


def run_compare_eval(*, model_ref: str, record_results: bool) -> dict[str, object]:
    model_report = run_model_eval(model_ref=model_ref, top_k=10)
    baseline_report = load_latest_report("baseline")
    ensure_baseline_compatible(baseline_report, candidates_path=Path(model_report.candidates_path))
    baseline_metrics = baseline_report["metrics"]
    comparison = compare_metric_sets(model_report.metrics, baseline_metrics)
    payload = {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": model_report.metrics,
        "comparison": comparison,
        "model_ref": model_report.model_ref,
        "report_path": model_report.report_path,
    }
    if record_results:
        append_result(
            PATHS.root / "results.tsv",
            model_ref=model_report.model_ref,
            metrics=model_report.metrics,
            status=str(comparison["status"]),
            description="local fine-tuned compare run",
        )
    return payload


def aggregate_metrics(candidates: list[QueryCandidate | ReviewedQuery], hits_per_query, *, top_k: int) -> dict[str, float]:
    recall = 0.0
    mrr = 0.0
    ndcg = 0.0
    for candidate, hits in zip(candidates, hits_per_query, strict=True):
        result_ids = [hit.arxiv_id for hit in hits]
        relevant_ids = set(_relevant_ids(candidate))
        recall += recall_at_k(result_ids, relevant_ids, top_k)
        mrr += reciprocal_rank(result_ids, relevant_ids)
        ndcg += ndcg_at_k(result_ids, relevant_ids, top_k)
    total = len(candidates) or 1
    return {
        f"recall@{top_k}": recall / total,
        "mrr": mrr / total,
        f"ndcg@{top_k}": ndcg / total,
    }


def _run_eval(
    *,
    candidates: list[QueryCandidate | ReviewedQuery],
    candidates_path: Path,
    index_dir: Path,
    output_dir: Path,
    report_prefix: str,
    top_k: int,
) -> tuple[dict[str, float], Path]:
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
                "candidates_path": str(candidates_path),
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
    payload = load_latest_report(prefix)
    return payload["metrics"]


def load_latest_report(prefix: str) -> dict[str, object]:
    reports = sorted(PATHS.artifacts_results.glob(f"{prefix}-*.json"))
    if not reports:
        raise FileNotFoundError(f"No {prefix} reports found in {PATHS.artifacts_results}")
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def resolve_eval_candidates(
    *,
    generated_candidates_path: Path,
    reviewed_eval_path: Path,
) -> tuple[Path, list[QueryCandidate | ReviewedQuery]]:
    if reviewed_eval_path.exists():
        reviewed_queries = load_reviewed_queries(reviewed_eval_path)
        if reviewed_queries:
            return reviewed_eval_path, reviewed_queries
    return generated_candidates_path, load_query_candidates(generated_candidates_path)


def _relevant_ids(candidate: QueryCandidate | ReviewedQuery) -> tuple[str, ...]:
    if isinstance(candidate, ReviewedQuery):
        return candidate.relevant_paper_ids
    return candidate.positive_ids


def ensure_baseline_compatible(baseline_report: dict[str, object], *, candidates_path: Path) -> None:
    baseline_candidates_path = baseline_report.get("candidates_path")
    if baseline_candidates_path is None:
        raise ValueError("Latest baseline report is missing candidates_path; rerun `eval baseline`.")
    if Path(str(baseline_candidates_path)) != candidates_path:
        raise ValueError(
            "Latest baseline report targets a different benchmark split; rerun `eval baseline` before compare."
        )


def _resolve_checkpoint(model_ref: str | Path) -> Path:
    if isinstance(model_ref, Path):
        checkpoint = model_ref
    elif model_ref == "latest":
        checkpoint = latest_checkpoint()
    else:
        checkpoint = PATHS.artifacts_models / model_ref
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    return checkpoint
