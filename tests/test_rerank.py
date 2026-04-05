from __future__ import annotations

from mlsearch.retrieval.rerank import rerank_hits_with_scores
from mlsearch.retrieval.search import SearchHit


def test_rerank_hits_with_scores_reorders_hits_by_cross_encoder_score() -> None:
    hits = [
        SearchHit(arxiv_id="a", title="Paper A", published="2024-01-01", score=0.9),
        SearchHit(arxiv_id="b", title="Paper B", published="2024-01-01", score=0.8),
        SearchHit(arxiv_id="c", title="Paper C", published="2024-01-01", score=0.7),
    ]

    reranked = rerank_hits_with_scores(hits, [0.1, 0.9, 0.4])

    assert [hit.arxiv_id for hit in reranked] == ["b", "c", "a"]
    assert [hit.score for hit in reranked] == [0.9, 0.4, 0.1]
