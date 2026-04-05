from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from mlsearch.benchmark.review import write_review_csv
from mlsearch.benchmark.schema import QueryCandidate, ReviewedQuery
from mlsearch.pipelines.finalize_review_set import finalize_review_set
from mlsearch.pipelines.sample_review_set import load_reviewed_query_ids, load_reviewed_source_paper_ids


def test_load_reviewed_query_ids_includes_archived_and_held_out(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    reviewed_dir = root / "data" / "benchmark" / "reviewed"
    archive_dir = reviewed_dir / "archive" / "batch-1"
    archive_dir.mkdir(parents=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    archive_csv = archive_dir / "review_sample.csv"
    archive_csv.write_text(
        "query_id,style,source_paper_id,query_text,positive_ids,hard_negative_ids,review_status,edited_query,relevant_paper_ids,notes\n"
        "paper-1-keyword,keyword,paper-1,query,paper-1,,accept,,paper-1,\n",
        encoding="utf-8",
    )
    held_out = reviewed_dir / "held_out_eval.jsonl"
    held_out.write_text(
        json.dumps(
            ReviewedQuery(
                query_id="paper-2-question",
                query_text="query",
                style="question",
                source_paper_id="paper-2",
                relevant_paper_ids=("paper-2",),
                review_status="accept",
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )
    current_csv = reviewed_dir / "review_sample.csv"
    current_csv.write_text(
        "query_id,style,source_paper_id,query_text,positive_ids,hard_negative_ids,review_status,edited_query,relevant_paper_ids,notes\n"
        "paper-3-keyword,keyword,paper-3,query,paper-3,,pending,,,\n",
        encoding="utf-8",
    )

    fake_paths = SimpleNamespace(data_benchmark=root / "data" / "benchmark")
    monkeypatch.setattr("mlsearch.pipelines.sample_review_set.PATHS", fake_paths)

    reviewed_ids = load_reviewed_query_ids()

    assert reviewed_ids == {"paper-1-keyword", "paper-2-question", "paper-3-keyword"}


def test_load_reviewed_source_paper_ids_includes_archived_and_held_out(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    reviewed_dir = root / "data" / "benchmark" / "reviewed"
    archive_dir = reviewed_dir / "archive" / "batch-1"
    archive_dir.mkdir(parents=True)
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    archive_csv = archive_dir / "review_sample.csv"
    archive_csv.write_text(
        "query_id,style,source_paper_id,query_text,positive_ids,hard_negative_ids,review_status,edited_query,relevant_paper_ids,notes\n"
        "paper-1-keyword,keyword,paper-1,query,paper-1,,accept,,paper-1,\n",
        encoding="utf-8",
    )
    held_out = reviewed_dir / "held_out_eval.jsonl"
    held_out.write_text(
        json.dumps(
            ReviewedQuery(
                query_id="paper-2-question",
                query_text="query",
                style="question",
                source_paper_id="paper-2",
                relevant_paper_ids=("paper-2",),
                review_status="accept",
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )
    current_csv = reviewed_dir / "review_sample.csv"
    current_csv.write_text(
        "query_id,style,source_paper_id,query_text,positive_ids,hard_negative_ids,review_status,edited_query,relevant_paper_ids,notes\n"
        "paper-3-keyword,keyword,paper-3,query,paper-3,,pending,,,\n",
        encoding="utf-8",
    )

    fake_paths = SimpleNamespace(data_benchmark=root / "data" / "benchmark")
    monkeypatch.setattr("mlsearch.pipelines.sample_review_set.PATHS", fake_paths)

    reviewed_source_paper_ids = load_reviewed_source_paper_ids()

    assert reviewed_source_paper_ids == {"paper-1", "paper-2", "paper-3"}


def test_finalize_review_set_merges_existing_held_out_eval(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    reviewed_dir = root / "data" / "benchmark" / "reviewed"
    reviewed_dir.mkdir(parents=True)
    review_path = reviewed_dir / "review_sample.csv"
    write_review_csv(
        review_path,
        [
            QueryCandidate(
                query_id="paper-2-keyword",
                query_text="paper two",
                style="keyword",
                source_paper_id="paper-2",
                source_title="Paper Two",
                source_published="2020-01-01T00:00:00Z",
                positive_ids=("paper-2",),
                hard_negative_ids=(),
            )
        ],
    )
    rows = list(csv.DictReader(review_path.open()))
    rows[0]["review_status"] = "accept"
    rows[0]["relevant_paper_ids"] = "paper-2"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    existing_path = reviewed_dir / "held_out_eval.jsonl"
    existing_path.write_text(
        json.dumps(
            ReviewedQuery(
                query_id="paper-1-keyword",
                query_text="paper one",
                style="keyword",
                source_paper_id="paper-1",
                relevant_paper_ids=("paper-1",),
                review_status="accept",
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )

    fake_paths = SimpleNamespace(data_benchmark=root / "data" / "benchmark")
    monkeypatch.setattr("mlsearch.pipelines.finalize_review_set.PATHS", fake_paths)

    report = finalize_review_set(review_path=review_path)

    merged_rows = [json.loads(line) for line in existing_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert report.accepted_count == 1
    assert report.added_count == 1
    assert report.merged_count == 2
    assert report.source_paper_count == 2
    assert {row["query_id"] for row in merged_rows} == {"paper-1-keyword", "paper-2-keyword"}
