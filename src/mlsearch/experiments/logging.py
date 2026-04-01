from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

RESULTS_HEADER = "timestamp\tmodel_ref\trecall@10\tmrr\tndcg@10\tstatus\tdescription\n"


def ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text(RESULTS_HEADER, encoding="utf-8")


def append_result(
    path: Path,
    *,
    model_ref: str,
    metrics: dict[str, float],
    status: str,
    description: str,
) -> None:
    ensure_results_file(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = "\t".join(
        [
            timestamp,
            model_ref,
            f"{metrics['recall@10']:.6f}",
            f"{metrics['mrr']:.6f}",
            f"{metrics['ndcg@10']:.6f}",
            status,
            description,
        ]
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")
