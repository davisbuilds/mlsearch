from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from mlsearch.benchmark.review import (
    apply_review_decision,
    load_next_review_item,
    load_review_rows,
    run_review_loop,
    summarize_review_progress,
)
from mlsearch.data.models import ArxivPaper


def _write_review_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "style",
                "source_paper_id",
                "query_text",
                "positive_ids",
                "hard_negative_ids",
                "review_status",
                "edited_query",
                "relevant_paper_ids",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "paper-1-question",
                "style": "question",
                "source_paper_id": "paper-1",
                "query_text": "what papers study few-shot classification?",
                "positive_ids": "paper-1",
                "hard_negative_ids": "paper-2",
                "review_status": "pending",
                "edited_query": "",
                "relevant_paper_ids": "",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "query_id": "paper-2-keyword",
                "style": "keyword",
                "source_paper_id": "paper-2",
                "query_text": "uncertainty graph neural networks",
                "positive_ids": "paper-2",
                "hard_negative_ids": "paper-1",
                "review_status": "accept",
                "edited_query": "",
                "relevant_paper_ids": "",
                "notes": "already reviewed",
            }
        )


def _write_corpus(path: Path) -> None:
    papers = [
        ArxivPaper(
            arxiv_id="paper-1",
            title="Meta-learning for few-shot classification",
            abstract="This paper studies few-shot classification with meta-learning.",
            authors=("Alice",),
            categories=("cs.LG",),
            primary_category="cs.LG",
            published="2020-01-01T00:00:00Z",
            updated="2020-01-01T00:00:00Z",
            abs_url="https://arxiv.org/abs/paper-1",
            pdf_url=None,
        ),
        ArxivPaper(
            arxiv_id="paper-2",
            title="Uncertainty estimation in graph neural networks",
            abstract="This paper studies uncertainty estimation for graph neural networks.",
            authors=("Bob",),
            categories=("cs.LG",),
            primary_category="cs.LG",
            published="2020-01-02T00:00:00Z",
            updated="2020-01-02T00:00:00Z",
            abs_url="https://arxiv.org/abs/paper-2",
            pdf_url=None,
        ),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.to_dict()) + "\n")


def test_summarize_review_progress_counts_statuses_and_styles(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    _write_review_csv(review_path)

    report = summarize_review_progress(review_path)

    assert report.total_count == 2
    assert report.pending_count == 1
    assert report.completed_count == 1
    assert report.status_counts == {"accept": 1, "pending": 1}
    assert report.style_counts == {"keyword": 1, "question": 1}


def test_load_next_review_item_returns_pending_query_with_source_context(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_review_csv(review_path)
    _write_corpus(corpus_path)

    report = load_next_review_item(review_path=review_path, corpus_path=corpus_path)

    assert report.query_id == "paper-1-question"
    assert report.review_row["review_status"] == "pending"
    assert report.source_paper["arxiv_id"] == "paper-1"
    assert "few-shot classification" in report.source_paper["title"].lower()
    assert [paper["arxiv_id"] for paper in report.positive_papers] == ["paper-1"]
    assert [paper["arxiv_id"] for paper in report.hard_negative_papers] == ["paper-2"]


def test_load_next_review_item_can_select_explicit_query_id(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_review_csv(review_path)
    _write_corpus(corpus_path)

    report = load_next_review_item(
        review_path=review_path,
        corpus_path=corpus_path,
        query_id="paper-2-keyword",
    )

    assert report.query_id == "paper-2-keyword"
    assert report.review_row["review_status"] == "accept"
    assert report.source_paper["arxiv_id"] == "paper-2"


def test_apply_review_decision_updates_row_for_accept(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    _write_review_csv(review_path)
    rows = load_review_rows(review_path)

    updated = apply_review_decision(
        rows[0],
        action="accept",
        notes="clear accept",
    )

    assert updated["review_status"] == "accept"
    assert updated["edited_query"] == ""
    assert updated["relevant_paper_ids"] == ""
    assert updated["notes"] == "clear accept"


def test_apply_review_decision_updates_row_for_edit(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    _write_review_csv(review_path)
    rows = load_review_rows(review_path)

    updated = apply_review_decision(
        rows[0],
        action="edit",
        edited_query="few-shot classification meta-learning",
        relevant_paper_ids="paper-1|paper-2",
        notes="better wording",
    )

    assert updated["review_status"] == "edit"
    assert updated["edited_query"] == "few-shot classification meta-learning"
    assert updated["relevant_paper_ids"] == "paper-1|paper-2"
    assert updated["notes"] == "better wording"


def test_run_review_loop_accepts_pending_row_and_persists_csv(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_review_csv(review_path)
    _write_corpus(corpus_path)

    stdin = io.StringIO("a\nlooks good\nq\n")
    stdout = io.StringIO()

    report = run_review_loop(
        review_path=review_path,
        corpus_path=corpus_path,
        input_stream=stdin,
        output_stream=stdout,
        limit=1,
    )

    assert report.updated_count == 1
    updated_rows = load_review_rows(review_path)
    assert updated_rows[0]["review_status"] == "accept"
    assert updated_rows[0]["notes"] == "looks good"
    output = stdout.getvalue()
    assert "paper-1-question" in output
    assert "Accepted." in output
