from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mlsearch.benchmark.review import (
    REVIEW_COLUMNS,
    finalize_review_csv,
    write_review_csv,
)
from mlsearch.benchmark.schema import QueryCandidate, ReviewedQuery


def test_write_review_csv_creates_expected_columns(tmp_path: Path) -> None:
    candidate = QueryCandidate(
        query_id="paper-1-keyword",
        query_text="few-shot meta learning",
        style="keyword",
        source_paper_id="paper-1",
        source_title="Meta-learning for few-shot classification",
        source_published="2020-01-01T00:00:00Z",
        positive_ids=("paper-1",),
        hard_negative_ids=("paper-2",),
    )
    path = tmp_path / "review.csv"
    write_review_csv(path, [candidate])
    content = path.read_text(encoding="utf-8")
    for column in REVIEW_COLUMNS:
        assert column in content


def test_finalize_review_csv_materializes_only_accepted_queries(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "paper-1-question",
                "style": "question",
                "source_paper_id": "paper-1",
                "query_text": "what papers study few-shot classification?",
                "positive_ids": "paper-1",
                "hard_negative_ids": "paper-2",
                "review_status": "accept",
                "edited_query": "",
                "relevant_paper_ids": "",
                "notes": "looks good",
            }
        )
        writer.writerow(
            {
                "query_id": "paper-2-keyword",
                "style": "keyword",
                "source_paper_id": "paper-2",
                "query_text": "uncertainty graph neural networks",
                "positive_ids": "paper-2",
                "hard_negative_ids": "paper-3",
                "review_status": "edit",
                "edited_query": "uncertainty estimation graph neural networks",
                "relevant_paper_ids": "paper-2|paper-9",
                "notes": "add broader relevant set",
            }
        )
        writer.writerow(
            {
                "query_id": "paper-3-keyword",
                "style": "keyword",
                "source_paper_id": "paper-3",
                "query_text": "bad synthetic query",
                "positive_ids": "paper-3",
                "hard_negative_ids": "",
                "review_status": "reject",
                "edited_query": "",
                "relevant_paper_ids": "",
                "notes": "drop this one",
            }
        )

    output_path = tmp_path / "held_out_eval.jsonl"
    manifest_path = tmp_path / "held_out_eval_manifest.json"
    report = finalize_review_csv(review_path, output_path=output_path, manifest_path=manifest_path)

    assert report.accepted_count == 2
    assert report.rejected_count == 1
    assert output_path.exists()
    payloads = [ReviewedQuery.from_dict(json.loads(line)) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [item.query_id for item in payloads] == ["paper-1-question", "paper-2-keyword"]
    assert payloads[0].query_text == "what papers study few-shot classification?"
    assert payloads[0].relevant_paper_ids == ("paper-1",)
    assert payloads[1].query_text == "uncertainty estimation graph neural networks"
    assert payloads[1].relevant_paper_ids == ("paper-2", "paper-9")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["accepted_count"] == 2
    assert manifest["rejected_count"] == 1
    assert manifest["styles"] == {"keyword": 1, "question": 1}


def test_finalize_review_csv_rejects_invalid_edit_rows(tmp_path: Path) -> None:
    review_path = tmp_path / "review.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "paper-1-question",
                "style": "question",
                "source_paper_id": "paper-1",
                "query_text": "what papers study few-shot classification?",
                "positive_ids": "paper-1",
                "hard_negative_ids": "",
                "review_status": "edit",
                "edited_query": "",
                "relevant_paper_ids": "paper-1",
                "notes": "",
            }
        )

    with pytest.raises(ValueError, match="edited_query"):
        finalize_review_csv(review_path, output_path=tmp_path / "held_out_eval.jsonl")
