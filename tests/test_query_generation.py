from __future__ import annotations

from arxiv_cslg_search.benchmark.schema import QueryCandidate
from arxiv_cslg_search.config import BenchmarkConfig
from arxiv_cslg_search.data.models import ArxivPaper
from arxiv_cslg_search.pipelines.generate_queries import build_query_candidates


def _paper(arxiv_id: str, title: str) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="This paper studies generalization and optimization.",
        authors=("Alice",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published="2020-01-01T00:00:00Z",
        updated="2020-01-01T00:00:00Z",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=None,
    )


def test_build_query_candidates_emits_keyword_and_question_styles() -> None:
    candidates = build_query_candidates(
        [
            _paper("1", "Meta-learning for few-shot classification"),
            _paper("2", "Few-shot optimization for meta-learning systems"),
        ],
        BenchmarkConfig(max_candidates=10, negatives_per_query=1),
    )
    assert len(candidates) == 4
    styles = {candidate.style for candidate in candidates}
    assert styles == {"keyword", "question"}

    keyword = next(candidate for candidate in candidates if candidate.query_id == "1-keyword")
    assert keyword.positive_ids == ("1",)
    assert keyword.hard_negative_ids == ("2",)


def test_question_query_looks_human_readable() -> None:
    candidates = build_query_candidates(
        [_paper("1", "Uncertainty estimation in graph neural networks")],
        BenchmarkConfig(max_candidates=10, negatives_per_query=1),
    )
    question = next(candidate for candidate in candidates if candidate.style == "question")
    assert question.query_text.startswith("what papers study ")
    assert question.query_text.endswith("?")
