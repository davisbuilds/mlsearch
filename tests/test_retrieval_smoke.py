from __future__ import annotations

from pathlib import Path

import numpy as np

from mlsearch.data.models import ArxivPaper
from mlsearch.retrieval.search import SearchHit, rank_hits, search_index


def _paper(arxiv_id: str, title: str) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="",
        authors=("Alice",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published="2020-01-01T00:00:00Z",
        updated="2020-01-01T00:00:00Z",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=None,
    )


def test_rank_hits_returns_highest_scoring_papers_first() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    documents = np.array(
        [
            [0.1, 0.0],
            [0.9, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float32,
    )
    hits = rank_hits(
        query,
        documents,
        [_paper("a", "first"), _paper("b", "second"), _paper("c", "third")],
        top_k=2,
    )
    assert [hit.arxiv_id for hit in hits] == ["b", "c"]


def test_search_index_can_apply_reranking(monkeypatch) -> None:
    papers = [_paper("a", "first"), _paper("b", "second")]
    embeddings = np.array([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32)
    index_dir = Path("/tmp/test-index")

    class FakeEmbedder:
        def __init__(self, _config) -> None:
            pass

        def embed_queries(self, _queries):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    def fake_load_index(_index_dir=None):
        return papers, embeddings, {"model_name": "fake-model"}

    captured = {}

    def fake_rerank_hit_lists(queries, hits_per_query, *, index_dir, config):
        captured["queries"] = queries
        captured["index_dir"] = index_dir
        captured["model_name"] = config.model_name
        captured["rerank_depth"] = config.rerank_depth
        return [[hits_per_query[0][1], hits_per_query[0][0]]]

    monkeypatch.setattr("mlsearch.retrieval.search.load_index", fake_load_index)
    monkeypatch.setattr("mlsearch.retrieval.search.TextEmbedder", FakeEmbedder)
    monkeypatch.setattr("mlsearch.retrieval.rerank.rerank_hit_lists", fake_rerank_hit_lists)

    hits = search_index(
        "test query",
        top_k=2,
        index_dir=index_dir,
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_depth=5,
    )

    assert [hit.arxiv_id for hit in hits] == ["a", "b"]
    assert captured == {
        "queries": ["test query"],
        "index_dir": index_dir,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_depth": 5,
    }
