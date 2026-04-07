from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlsearch.benchmark.review import load_reviewed_queries
from mlsearch.benchmark.splits import DEFAULT_REVIEW_SPLIT, held_out_eval_path
from mlsearch.benchmark.schema import QueryCandidate, ReviewedQuery
from mlsearch.experiments.compare import compare_metric_sets
from mlsearch.experiments.logging import append_result
from mlsearch.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from mlsearch.paths import PATHS
from mlsearch.pipelines.generate_queries import load_query_candidates
from mlsearch.retrieval.index import build_index
from mlsearch.retrieval.rerank import DEFAULT_RERANKER_MODEL_NAME, RerankerConfig, rerank_hit_lists
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
    split: str = DEFAULT_REVIEW_SPLIT,
) -> BaselineEvalReport:
    selected_candidates_path, candidates = resolve_eval_candidates(
        generated_candidates_path=candidates_path or (PATHS.data_benchmark / "generated" / "query_candidates.jsonl"),
        reviewed_eval_path=held_out_eval_path(split=split),
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


def run_baseline_rerank_eval(
    *,
    candidates_path: Path | None = None,
    output_dir: Path | None = None,
    top_k: int = 10,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    rerank_depth: int = 10,
    split: str = DEFAULT_REVIEW_SPLIT,
) -> BaselineEvalReport:
    selected_candidates_path, candidates = resolve_eval_candidates(
        generated_candidates_path=candidates_path or (PATHS.data_benchmark / "generated" / "query_candidates.jsonl"),
        reviewed_eval_path=held_out_eval_path(split=split),
    )
    metrics, report_path = _run_eval(
        candidates=candidates,
        candidates_path=selected_candidates_path,
        index_dir=PATHS.artifacts_index,
        output_dir=output_dir or PATHS.artifacts_results,
        report_prefix="baseline-rerank",
        top_k=top_k,
        reranker_model_name=reranker_model_name,
        rerank_depth=rerank_depth,
    )
    return BaselineEvalReport(
        report_path=str(report_path),
        candidates_path=str(selected_candidates_path),
        query_count=len(candidates),
        metrics=metrics,
    )


def run_model_eval(*, model_ref: str | Path, top_k: int = 10, split: str = DEFAULT_REVIEW_SPLIT) -> ModelEvalReport:
    return _run_checkpoint_eval(
        model_ref=model_ref,
        top_k=top_k,
        report_prefix="compare",
        split=split,
    )


def run_rerank_experiment(
    *,
    retriever_model_ref: str,
    reference_model_ref: str,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    rerank_depth: int = 10,
    top_k: int = 10,
    record_results: bool,
    split: str = DEFAULT_REVIEW_SPLIT,
) -> dict[str, object]:
    candidate_report = _run_checkpoint_eval(
        model_ref=retriever_model_ref,
        top_k=top_k,
        report_prefix="rerank",
        reranker_model_name=reranker_model_name,
        rerank_depth=rerank_depth,
        split=split,
    )
    reference_name, reference_report = load_reference_report(reference_model_ref, top_k=top_k)
    ensure_report_compatible(reference_report, candidates_path=Path(candidate_report.candidates_path), report_label=reference_name)
    reference_metrics = reference_report["metrics"]
    comparison = compare_metric_sets(candidate_report.metrics, reference_metrics)
    candidate_report_payload = json.loads(Path(candidate_report.report_path).read_text(encoding="utf-8"))
    query_deltas = build_query_delta_report(
        baseline_queries=list(reference_report.get("per_query", [])),
        candidate_queries=list(candidate_report_payload.get("per_query", [])),
    )
    candidate_report_payload.update(
        {
            "comparison": comparison,
            "reference_metrics": reference_metrics,
            "reference_model_ref": reference_name,
            "rerank_depth": rerank_depth,
            "reranker_model_name": reranker_model_name,
            "retriever_model_ref": candidate_report.model_ref,
            "query_deltas": query_deltas,
        }
    )
    Path(candidate_report.report_path).write_text(
        json.dumps(candidate_report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "candidate_metrics": candidate_report.metrics,
        "comparison": comparison,
        "query_delta_count": len(query_deltas),
        "reference_metrics": reference_metrics,
        "reference_model_ref": reference_name,
        "report_path": candidate_report.report_path,
        "rerank_depth": rerank_depth,
        "reranker_model_name": reranker_model_name,
        "retriever_model_ref": candidate_report.model_ref,
    }
    if record_results:
        append_result(
            PATHS.root / "results.tsv",
            model_ref=f"{candidate_report.model_ref}+rerank",
            metrics=candidate_report.metrics,
            status=str(comparison["status"]),
            description=(
                f"rerank experiment vs={reference_name} "
                f"reranker={reranker_model_name} depth={rerank_depth}"
            ),
        )
    return payload


def _run_checkpoint_eval(
    *,
    model_ref: str | Path,
    top_k: int,
    report_prefix: str,
    reranker_model_name: str | None = None,
    rerank_depth: int = 10,
    split: str = DEFAULT_REVIEW_SPLIT,
) -> ModelEvalReport:
    checkpoint = _resolve_checkpoint(model_ref)
    compare_index_dir = PATHS.artifacts_index / checkpoint.name
    build_index(output_dir=compare_index_dir, model_name=str(checkpoint))
    candidate_path, candidates = resolve_eval_candidates(
        generated_candidates_path=PATHS.data_benchmark / "generated" / "query_candidates.jsonl",
        reviewed_eval_path=held_out_eval_path(split=split),
    )
    candidate_metrics, report_path = _run_eval(
        candidates=candidates,
        candidates_path=candidate_path,
        index_dir=compare_index_dir,
        output_dir=PATHS.artifacts_results,
        report_prefix=report_prefix,
        top_k=top_k,
        reranker_model_name=reranker_model_name,
        rerank_depth=rerank_depth,
    )
    return ModelEvalReport(
        report_path=str(report_path),
        candidates_path=str(candidate_path),
        index_dir=str(compare_index_dir),
        model_ref=checkpoint.name,
        query_count=len(candidates),
        metrics=candidate_metrics,
    )


def run_compare_eval(*, model_ref: str, record_results: bool, split: str = DEFAULT_REVIEW_SPLIT) -> dict[str, object]:
    model_report = run_model_eval(model_ref=model_ref, top_k=10, split=split)
    baseline_report = load_latest_report("baseline")
    ensure_baseline_compatible(baseline_report, candidates_path=Path(model_report.candidates_path))
    baseline_metrics = baseline_report["metrics"]
    comparison = compare_metric_sets(model_report.metrics, baseline_metrics)
    candidate_report_payload = json.loads(Path(model_report.report_path).read_text(encoding="utf-8"))
    query_deltas = build_query_delta_report(
        baseline_queries=list(baseline_report.get("per_query", [])),
        candidate_queries=list(candidate_report_payload.get("per_query", [])),
    )
    candidate_report_payload.update(
        {
            "baseline_metrics": baseline_metrics,
            "comparison": comparison,
            "model_ref": model_report.model_ref,
            "query_deltas": query_deltas,
        }
    )
    Path(model_report.report_path).write_text(
        json.dumps(candidate_report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": model_report.metrics,
        "comparison": comparison,
        "model_ref": model_report.model_ref,
        "report_path": model_report.report_path,
        "query_delta_count": len(query_deltas),
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


def build_query_breakdowns(
    candidates: list[QueryCandidate | ReviewedQuery],
    hits_per_query: list[list[object]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    breakdowns: list[dict[str, Any]] = []
    for candidate, hits in zip(candidates, hits_per_query, strict=True):
        result_ids = [hit.arxiv_id for hit in hits]
        relevant_ids = set(_relevant_ids(candidate))
        top_hit = hits[0] if hits else None
        breakdowns.append(
            {
                "query_id": candidate.query_id,
                "query_text": candidate.query_text,
                "style": candidate.style,
                "source_paper_id": candidate.source_paper_id,
                "relevant_paper_ids": sorted(relevant_ids),
                "top_hit_arxiv_id": top_hit.arxiv_id if top_hit is not None else None,
                "top_hit_title": top_hit.title if top_hit is not None else None,
                "top_hit_score": top_hit.score if top_hit is not None else None,
                "relevant_rank": _relevant_rank(result_ids, relevant_ids),
                "recall_at_k": recall_at_k(result_ids, relevant_ids, top_k),
                "reciprocal_rank": reciprocal_rank(result_ids, relevant_ids),
                "ndcg_at_k": ndcg_at_k(result_ids, relevant_ids, top_k),
            }
        )
    return breakdowns


def _run_eval(
    *,
    candidates: list[QueryCandidate | ReviewedQuery],
    candidates_path: Path,
    index_dir: Path,
    output_dir: Path,
    report_prefix: str,
    top_k: int,
    reranker_model_name: str | None = None,
    rerank_depth: int = 10,
) -> tuple[dict[str, float], Path]:
    hits_per_query = search_many(
        [candidate.query_text for candidate in candidates],
        top_k=top_k,
        index_dir=index_dir,
    )
    if reranker_model_name is not None:
        hits_per_query = rerank_hit_lists(
            [candidate.query_text for candidate in candidates],
            hits_per_query,
            index_dir=index_dir,
            config=RerankerConfig(model_name=reranker_model_name, rerank_depth=rerank_depth),
        )
    metrics = aggregate_metrics(candidates, hits_per_query, top_k=top_k)
    query_breakdowns = build_query_breakdowns(candidates, hits_per_query, top_k=top_k)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"{report_prefix}-{timestamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "candidates_path": str(candidates_path),
                "query_count": len(candidates),
                "metrics": metrics,
                "per_query": query_breakdowns,
                "top_k": top_k,
                "index_dir": str(index_dir),
                "rerank_depth": rerank_depth if reranker_model_name is not None else None,
                "reranker_model_name": reranker_model_name,
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


def ensure_report_compatible(
    report_payload: dict[str, object],
    *,
    candidates_path: Path,
    report_label: str,
) -> None:
    report_candidates_path = report_payload.get("candidates_path")
    if report_candidates_path is None:
        raise ValueError(f"{report_label} report is missing candidates_path; rerun the reference evaluation.")
    if Path(str(report_candidates_path)) != candidates_path:
        raise ValueError(
            f"{report_label} report targets a different benchmark split; rerun the reference evaluation first."
        )


def build_query_delta_report(
    baseline_queries: list[dict[str, Any]],
    candidate_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {str(item["query_id"]): item for item in baseline_queries}
    deltas: list[dict[str, Any]] = []
    for candidate_item in candidate_queries:
        query_id = str(candidate_item["query_id"])
        baseline_item = baseline_by_id.get(query_id)
        if baseline_item is None:
            continue
        baseline_rank = _maybe_int(baseline_item.get("relevant_rank"))
        candidate_rank = _maybe_int(candidate_item.get("relevant_rank"))
        deltas.append(
            {
                "query_id": query_id,
                "query_text": candidate_item["query_text"],
                "style": candidate_item["style"],
                "source_paper_id": candidate_item["source_paper_id"],
                "baseline_relevant_rank": baseline_rank,
                "candidate_relevant_rank": candidate_rank,
                "delta_relevant_rank": _delta_rank(baseline_rank, candidate_rank),
                "baseline_reciprocal_rank": baseline_item["reciprocal_rank"],
                "candidate_reciprocal_rank": candidate_item["reciprocal_rank"],
                "delta_reciprocal_rank": candidate_item["reciprocal_rank"] - baseline_item["reciprocal_rank"],
                "baseline_top_hit_arxiv_id": baseline_item.get("top_hit_arxiv_id"),
                "candidate_top_hit_arxiv_id": candidate_item.get("top_hit_arxiv_id"),
                "baseline_top_hit_title": baseline_item.get("top_hit_title"),
                "candidate_top_hit_title": candidate_item.get("top_hit_title"),
                "top_hit_changed": baseline_item.get("top_hit_arxiv_id") != candidate_item.get("top_hit_arxiv_id"),
            }
        )
    return sorted(
        deltas,
        key=lambda item: (
            -item["delta_reciprocal_rank"],
            -(item["delta_relevant_rank"] or 0),
            item["query_id"],
        ),
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


def load_reference_report(reference_model_ref: str, *, top_k: int) -> tuple[str, dict[str, object]]:
    if reference_model_ref == "baseline":
        return "baseline", load_latest_report("baseline")
    reference_eval = _run_checkpoint_eval(
        model_ref=reference_model_ref,
        top_k=top_k,
        report_prefix="reference",
    )
    report = json.loads(Path(reference_eval.report_path).read_text(encoding="utf-8"))
    return reference_eval.model_ref, report


def _relevant_rank(result_ids: list[str], relevant_ids: set[str]) -> int | None:
    for index, result_id in enumerate(result_ids, start=1):
        if result_id in relevant_ids:
            return index
    return None


def _maybe_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _delta_rank(baseline_rank: int | None, candidate_rank: int | None) -> int | None:
    if baseline_rank is None or candidate_rank is None:
        return None
    return baseline_rank - candidate_rank
