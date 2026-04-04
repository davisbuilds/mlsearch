from __future__ import annotations

import json
from pathlib import Path

from mlsearch.benchmark.schema import QueryCandidate
from mlsearch.config import BenchmarkConfig
from mlsearch.data.models import ArxivPaper
from mlsearch.pipelines.generate_queries import (
    build_query_candidates,
    compute_query_diagnostics,
    title_overlap_ratio,
    write_query_candidates,
)


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
    assert question.query_text.startswith(("papers on ", "research on ", "work on "))
    assert question.query_text == question.query_text.lower()


def test_build_query_candidates_uses_abstract_signal_to_reduce_title_copy() -> None:
    paper = ArxivPaper(
        arxiv_id="1",
        title="ViSioRed: A Visualisation Tool for Interpretable Predictive Process Monitoring",
        abstract=(
            "We study interpretable predictive process monitoring through interactive "
            "visualization dashboards for operational workflows."
        ),
        authors=("Alice",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published="2020-01-01T00:00:00Z",
        updated="2020-01-01T00:00:00Z",
        abs_url="https://arxiv.org/abs/1",
        pdf_url=None,
    )

    candidates = build_query_candidates([paper], BenchmarkConfig(max_candidates=10, negatives_per_query=1))
    keyword = next(candidate for candidate in candidates if candidate.style == "keyword")
    question = next(candidate for candidate in candidates if candidate.style == "question")

    assert "visualisation tool" not in keyword.query_text
    assert "visiored" not in keyword.query_text
    assert "interactive visualization dashboards" in question.query_text
    assert title_overlap_ratio(question.query_text, paper.title) < 0.7


def test_build_query_candidates_prefers_title_topics_over_abstract_intro_debris() -> None:
    paper = ArxivPaper(
        arxiv_id="2",
        title="MLSys: The New Frontier of Machine Learning Systems",
        abstract=(
            "Machine learning techniques are enjoying rapidly increasing adoption. "
            "However, designing and implementing the systems that support ML models "
            "in real-world deployments remains a significant obstacle."
        ),
        authors=("Alice",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published="2020-01-01T00:00:00Z",
        updated="2020-01-01T00:00:00Z",
        abs_url="https://arxiv.org/abs/2",
        pdf_url=None,
    )

    candidates = build_query_candidates([paper], BenchmarkConfig(max_candidates=10, negatives_per_query=1))
    keyword = next(candidate for candidate in candidates if candidate.style == "keyword")
    question = next(candidate for candidate in candidates if candidate.style == "question")

    assert "rapidly increasing adoption" not in keyword.query_text
    assert "machine learning systems" in keyword.query_text
    assert question.query_text.startswith(("papers on ", "research on ", "work on "))


def test_compute_query_diagnostics_summarizes_overlap_by_style() -> None:
    candidates = [
        QueryCandidate(
            query_id="1-keyword",
            query_text="few-shot classification optimization",
            style="keyword",
            source_paper_id="1",
            source_title="Meta-learning for few-shot classification",
            source_published="2020-01-01T00:00:00Z",
            positive_ids=("1",),
            hard_negative_ids=(),
        ),
        QueryCandidate(
            query_id="1-question",
            query_text="what papers study interactive visualization dashboards for process monitoring?",
            style="question",
            source_paper_id="1",
            source_title="ViSioRed: A Visualisation Tool for Interpretable Predictive Process Monitoring",
            source_published="2020-01-01T00:00:00Z",
            positive_ids=("1",),
            hard_negative_ids=(),
        ),
    ]

    diagnostics = compute_query_diagnostics(candidates)

    assert diagnostics["count"] == 2
    assert diagnostics["style_overlap"]["keyword"]["count"] == 1
    assert diagnostics["style_overlap"]["question"]["count"] == 1
    assert diagnostics["style_overlap"]["question"]["mean_title_overlap"] < 1.0


def test_write_query_candidates_writes_diagnostics_to_manifest(tmp_path: Path) -> None:
    paper = ArxivPaper(
        arxiv_id="1",
        title="Masked Autoencoders for Longitudinal Records",
        abstract="We study masked autoencoder pretraining for longitudinal patient records.",
        authors=("Alice",),
        categories=("cs.LG",),
        primary_category="cs.LG",
        published="2020-01-01T00:00:00Z",
        updated="2020-01-01T00:00:00Z",
        abs_url="https://arxiv.org/abs/1",
        pdf_url=None,
    )
    candidates = build_query_candidates([paper], BenchmarkConfig(max_candidates=1, negatives_per_query=1))

    report = write_query_candidates(candidates, generated_dir=tmp_path, seed=42, max_candidates=1)
    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))

    assert "diagnostics" in manifest
    assert manifest["diagnostics"]["count"] == 2
