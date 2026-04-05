from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlsearch.benchmark.review import finalize_review_csv, load_reviewed_queries
from mlsearch.benchmark.schema import ReviewedQuery
from mlsearch.paths import PATHS


@dataclass(frozen=True)
class FinalizeReviewPipelineReport:
    output_path: str
    manifest_path: str
    accepted_count: int
    merged_count: int
    added_count: int
    rejected_count: int
    styles: dict[str, int]


def finalize_review_set(*, review_path: Path | None = None) -> FinalizeReviewPipelineReport:
    resolved_review_path = review_path or (PATHS.data_benchmark / "reviewed" / "review_sample.csv")
    output_path = PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl"
    manifest_path = PATHS.data_benchmark / "reviewed" / "held_out_eval_manifest.json"
    previous_queries = load_reviewed_queries(output_path) if output_path.exists() else []
    temp_output_path = output_path.with_suffix(".new.jsonl")
    report = finalize_review_csv(
        resolved_review_path,
        output_path=temp_output_path,
        manifest_path=manifest_path,
    )
    new_queries = load_reviewed_queries(temp_output_path)
    merged_queries = merge_reviewed_queries(previous_queries, new_queries)
    write_reviewed_queries(output_path, merged_queries)
    temp_output_path.unlink(missing_ok=True)
    manifest_payload = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    manifest_payload.update(
        {
            "accepted_count": len(new_queries),
            "merged_count": len(merged_queries),
            "added_count": sum(1 for query in new_queries if query.query_id not in {item.query_id for item in previous_queries}),
            "previous_count": len(previous_queries),
            "styles": _count_styles(merged_queries),
        }
    )
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    previous_ids = {query.query_id for query in previous_queries}
    return FinalizeReviewPipelineReport(
        output_path=str(output_path),
        manifest_path=str(manifest_path),
        accepted_count=report.accepted_count,
        merged_count=len(merged_queries),
        added_count=sum(1 for query in new_queries if query.query_id not in previous_ids),
        rejected_count=report.rejected_count,
        styles=_count_styles(merged_queries),
    )


def merge_reviewed_queries(previous: list[ReviewedQuery], current: list[ReviewedQuery]) -> list[ReviewedQuery]:
    merged: dict[str, ReviewedQuery] = {query.query_id: query for query in previous}
    for query in current:
        merged[query.query_id] = query
    return [merged[query_id] for query_id in sorted(merged)]


def write_reviewed_queries(path: Path, queries: list[ReviewedQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.to_dict(), sort_keys=True) + "\n")


def _count_styles(queries: list[ReviewedQuery]) -> dict[str, int]:
    styles: dict[str, int] = {}
    for query in queries:
        styles[query.style] = styles.get(query.style, 0) + 1
    return dict(sorted(styles.items()))
