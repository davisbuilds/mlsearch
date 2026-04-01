from __future__ import annotations

from pathlib import Path

from mlsearch.experiments.compare import compare_metric_sets
from mlsearch.experiments.logging import append_result, ensure_results_file


def test_compare_metric_sets_prefers_higher_recall_then_mrr() -> None:
    comparison = compare_metric_sets(
        {"recall@10": 0.8, "mrr": 0.6, "ndcg@10": 0.7},
        {"recall@10": 0.7, "mrr": 0.9, "ndcg@10": 0.9},
    )
    assert comparison["status"] == "keep"


def test_append_result_creates_results_tsv(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    ensure_results_file(path)
    append_result(
        path,
        model_ref="checkpoint-1",
        metrics={"recall@10": 1.0, "mrr": 1.0, "ndcg@10": 1.0},
        status="keep",
        description="test run",
    )
    content = path.read_text(encoding="utf-8")
    assert "checkpoint-1" in content
    assert "recall@10" in content.splitlines()[0]
