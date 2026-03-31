from __future__ import annotations


def compare_metric_sets(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float | str]:
    recall_key = next(key for key in candidate if key.startswith("recall@"))
    ndcg_key = next(key for key in candidate if key.startswith("ndcg@"))
    candidate_score = (candidate[recall_key], candidate["mrr"], candidate[ndcg_key])
    baseline_score = (baseline[recall_key], baseline["mrr"], baseline[ndcg_key])
    status = "keep" if candidate_score > baseline_score else "discard"
    return {
        "status": status,
        "delta_recall": candidate[recall_key] - baseline[recall_key],
        "delta_mrr": candidate["mrr"] - baseline["mrr"],
        "delta_ndcg": candidate[ndcg_key] - baseline[ndcg_key],
    }
