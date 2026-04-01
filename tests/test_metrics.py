from __future__ import annotations

from mlsearch.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_metrics_handle_binary_relevance() -> None:
    results = ["a", "b", "c"]
    relevant = {"b"}
    assert recall_at_k(results, relevant, 2) == 1.0
    assert reciprocal_rank(results, relevant) == 0.5
    assert 0.0 < ndcg_at_k(results, relevant, 3) <= 1.0
