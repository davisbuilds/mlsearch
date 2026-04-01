from __future__ import annotations

import math


def recall_at_k(result_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for result_id in result_ids[:k] if result_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(result_ids: list[str], relevant_ids: set[str]) -> float:
    for index, result_id in enumerate(result_ids, start=1):
        if result_id in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(result_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for index, result_id in enumerate(result_ids[:k], start=1):
        if result_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg
