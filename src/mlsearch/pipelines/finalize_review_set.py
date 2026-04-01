from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlsearch.benchmark.review import finalize_review_csv
from mlsearch.paths import PATHS


@dataclass(frozen=True)
class FinalizeReviewPipelineReport:
    output_path: str
    manifest_path: str
    accepted_count: int
    rejected_count: int
    styles: dict[str, int]


def finalize_review_set(*, review_path: Path | None = None) -> FinalizeReviewPipelineReport:
    resolved_review_path = review_path or (PATHS.data_benchmark / "reviewed" / "review_sample.csv")
    output_path = PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl"
    manifest_path = PATHS.data_benchmark / "reviewed" / "held_out_eval_manifest.json"
    report = finalize_review_csv(
        resolved_review_path,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    return FinalizeReviewPipelineReport(
        output_path=report.output_path,
        manifest_path=report.manifest_path,
        accepted_count=report.accepted_count,
        rejected_count=report.rejected_count,
        styles=report.styles,
    )
