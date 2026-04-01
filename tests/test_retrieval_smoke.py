from __future__ import annotations

import numpy as np

from mlsearch.data.models import ArxivPaper
from mlsearch.retrieval.search import rank_hits


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
