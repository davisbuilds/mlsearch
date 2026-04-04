from __future__ import annotations

from mlsearch.benchmark.schema import ReviewedQuery
from mlsearch.eval.run_eval import build_query_breakdowns, build_query_delta_report
from mlsearch.retrieval.search import SearchHit


def test_build_query_breakdowns_tracks_rank_and_top_hit() -> None:
    query = ReviewedQuery(
        query_id="q1",
        query_text="traffic routing ml",
        style="keyword",
        source_paper_id="p1",
        relevant_paper_ids=("p1",),
        review_status="accept",
    )
    hits = [
        SearchHit(arxiv_id="p2", title="Other Paper", published="2024-01-01", score=0.9),
        SearchHit(arxiv_id="p1", title="Target Paper", published="2024-01-01", score=0.8),
    ]

    breakdowns = build_query_breakdowns([query], [hits], top_k=10)

    assert breakdowns[0]["query_id"] == "q1"
    assert breakdowns[0]["top_hit_arxiv_id"] == "p2"
    assert breakdowns[0]["relevant_rank"] == 2
    assert breakdowns[0]["reciprocal_rank"] == 0.5


def test_build_query_delta_report_sorts_biggest_improvements_first() -> None:
    baseline_queries = [
        {
            "query_id": "q1",
            "query_text": "first query",
            "style": "keyword",
            "source_paper_id": "p1",
            "relevant_rank": 3,
            "reciprocal_rank": 1 / 3,
            "top_hit_arxiv_id": "x",
            "top_hit_title": "Old Top",
        },
        {
            "query_id": "q2",
            "query_text": "second query",
            "style": "question",
            "source_paper_id": "p2",
            "relevant_rank": 1,
            "reciprocal_rank": 1.0,
            "top_hit_arxiv_id": "p2",
            "top_hit_title": "Stable Top",
        },
    ]
    candidate_queries = [
        {
            "query_id": "q1",
            "query_text": "first query",
            "style": "keyword",
            "source_paper_id": "p1",
            "relevant_rank": 1,
            "reciprocal_rank": 1.0,
            "top_hit_arxiv_id": "p1",
            "top_hit_title": "New Top",
        },
        {
            "query_id": "q2",
            "query_text": "second query",
            "style": "question",
            "source_paper_id": "p2",
            "relevant_rank": 2,
            "reciprocal_rank": 0.5,
            "top_hit_arxiv_id": "x",
            "top_hit_title": "Wrong Top",
        },
    ]

    deltas = build_query_delta_report(baseline_queries, candidate_queries)

    assert deltas[0]["query_id"] == "q1"
    assert deltas[0]["delta_relevant_rank"] == 2
    assert deltas[0]["delta_reciprocal_rank"] > 0
    assert deltas[-1]["query_id"] == "q2"
    assert deltas[-1]["delta_relevant_rank"] == -1
    assert deltas[-1]["top_hit_changed"] is True
