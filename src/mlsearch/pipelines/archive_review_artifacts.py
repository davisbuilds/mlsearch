from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from mlsearch.paths import PATHS


@dataclass(frozen=True)
class ArchiveReviewArtifactsReport:
    archive_dir: str
    copied_files: list[str]


def archive_review_artifacts(*, reviewed_dir: Path | None = None, label: str | None = None) -> ArchiveReviewArtifactsReport:
    source_dir = reviewed_dir or (PATHS.data_benchmark / "reviewed")
    archive_label = label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = source_dir / "archive" / archive_label
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for name in ("review_sample.csv", "held_out_eval.jsonl", "held_out_eval_manifest.json"):
        source_path = source_dir / name
        if not source_path.exists():
            continue
        target_path = archive_dir / name
        shutil.copy2(source_path, target_path)
        copied_files.append(str(target_path))

    return ArchiveReviewArtifactsReport(
        archive_dir=str(archive_dir),
        copied_files=copied_files,
    )
