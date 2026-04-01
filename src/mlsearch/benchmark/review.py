from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from mlsearch.benchmark.schema import QueryCandidate, ReviewedQuery
from mlsearch.data.models import ArxivPaper
from mlsearch.pipelines.validate_corpus import load_corpus


REVIEW_COLUMNS = [
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
]
FINAL_REVIEW_STATUSES = frozenset({"accept", "edit", "reject"})


def write_review_csv(path: Path, candidates: list[QueryCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "query_id": candidate.query_id,
                    "style": candidate.style,
                    "source_paper_id": candidate.source_paper_id,
                    "query_text": candidate.query_text,
                    "positive_ids": "|".join(candidate.positive_ids),
                    "hard_negative_ids": "|".join(candidate.hard_negative_ids),
                    "review_status": "pending",
                    "edited_query": "",
                    "relevant_paper_ids": "",
                    "notes": "",
                }
            )


@dataclass(frozen=True)
class FinalizeReviewReport:
    output_path: str
    manifest_path: str
    accepted_count: int
    rejected_count: int
    styles: dict[str, int]


@dataclass(frozen=True)
class ReviewStatsReport:
    review_path: str
    total_count: int
    pending_count: int
    completed_count: int
    status_counts: dict[str, int]
    style_counts: dict[str, int]


@dataclass(frozen=True)
class ReviewNextReport:
    review_path: str
    query_id: str
    review_row: dict[str, str]
    source_paper: dict[str, object]
    positive_papers: list[dict[str, object]]
    hard_negative_papers: list[dict[str, object]]


@dataclass(frozen=True)
class ReviewLoopReport:
    review_path: str
    updated_count: int
    processed_count: int
    stopped_reason: str


def finalize_review_csv(
    review_path: Path,
    *,
    output_path: Path,
    manifest_path: Path | None = None,
) -> FinalizeReviewReport:
    reviewed_queries = load_review_decisions(review_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query in reviewed_queries:
            handle.write(json.dumps(query.to_dict(), sort_keys=True) + "\n")

    rejected_count = _count_rejected_rows(review_path)
    styles = _count_styles(reviewed_queries)
    resolved_manifest_path = manifest_path or output_path.with_name(f"{output_path.stem}_manifest.json")
    resolved_manifest_path.write_text(
        json.dumps(
            {
                "accepted_count": len(reviewed_queries),
                "rejected_count": rejected_count,
                "review_csv": str(review_path),
                "review_csv_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                "styles": styles,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return FinalizeReviewReport(
        output_path=str(output_path),
        manifest_path=str(resolved_manifest_path),
        accepted_count=len(reviewed_queries),
        rejected_count=rejected_count,
        styles=styles,
    )


def load_reviewed_queries(path: Path) -> list[ReviewedQuery]:
    if not path.exists():
        raise FileNotFoundError(f"Reviewed eval file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [ReviewedQuery.from_dict(json.loads(line)) for line in handle if line.strip()]


def load_review_decisions(review_path: Path) -> list[ReviewedQuery]:
    if not review_path.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_path}")

    reviewed_queries: list[ReviewedQuery] = []
    reviewed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            decision = _row_to_reviewed_query(row, review_path=review_path, reviewed_at=reviewed_at)
            if decision is not None:
                reviewed_queries.append(decision)
    return reviewed_queries


def summarize_review_progress(review_path: Path) -> ReviewStatsReport:
    rows = load_review_rows(review_path)
    status_counts: dict[str, int] = defaultdict(int)
    style_counts: dict[str, int] = defaultdict(int)
    pending_count = 0
    for row in rows:
        status = _normalize_review_status(row.get("review_status", ""))
        style = (row.get("style") or "").strip()
        status_counts[status] += 1
        if style:
            style_counts[style] += 1
        if status == "pending":
            pending_count += 1
    return ReviewStatsReport(
        review_path=str(review_path),
        total_count=len(rows),
        pending_count=pending_count,
        completed_count=len(rows) - pending_count,
        status_counts=dict(sorted(status_counts.items())),
        style_counts=dict(sorted(style_counts.items())),
    )


def load_next_review_item(
    *,
    review_path: Path,
    corpus_path: Path,
    query_id: str | None = None,
) -> ReviewNextReport:
    rows = load_review_rows(review_path)
    row = _select_review_row(rows, query_id=query_id)
    papers_by_id = {paper.arxiv_id: paper for paper in load_corpus(corpus_path)}
    source_paper_id = _required_value(row, "source_paper_id")
    source_paper = _paper_payload(papers_by_id[source_paper_id])
    positive_papers = _resolve_papers(_split_pipe_values(row.get("positive_ids", "")), papers_by_id)
    hard_negative_papers = _resolve_papers(_split_pipe_values(row.get("hard_negative_ids", "")), papers_by_id)
    return ReviewNextReport(
        review_path=str(review_path),
        query_id=_required_value(row, "query_id"),
        review_row={key: (value or "") for key, value in row.items()},
        source_paper=source_paper,
        positive_papers=positive_papers,
        hard_negative_papers=hard_negative_papers,
    )


def load_review_rows(review_path: Path) -> list[dict[str, str]]:
    if not review_path.exists():
        raise FileNotFoundError(f"Review CSV not found: {review_path}")
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def save_review_rows(review_path: Path, rows: list[dict[str, str]]) -> None:
    review_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = review_path.with_suffix(f"{review_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REVIEW_COLUMNS})
    temp_path.replace(review_path)


def apply_review_decision(
    row: dict[str, str],
    *,
    action: str,
    edited_query: str = "",
    relevant_paper_ids: str = "",
    notes: str = "",
) -> dict[str, str]:
    normalized_action = action.strip().lower()
    if normalized_action not in {"accept", "edit", "reject", "pending"}:
        raise ValueError(f"Unsupported review action: {action!r}")

    updated = dict(row)
    updated["review_status"] = normalized_action
    updated["notes"] = notes

    if normalized_action == "accept":
        updated["edited_query"] = ""
        updated["relevant_paper_ids"] = ""
        return updated
    if normalized_action == "edit":
        if not edited_query.strip():
            raise ValueError("edited_query is required for edit")
        if not relevant_paper_ids.strip():
            raise ValueError("relevant_paper_ids is required for edit")
        updated["edited_query"] = edited_query.strip()
        updated["relevant_paper_ids"] = relevant_paper_ids.strip()
        return updated

    updated["edited_query"] = ""
    updated["relevant_paper_ids"] = ""
    return updated


def run_review_loop(
    *,
    review_path: Path,
    corpus_path: Path,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    query_id: str | None = None,
    limit: int | None = None,
) -> ReviewLoopReport:
    stdin = input_stream or sys.stdin
    stdout = output_stream or sys.stdout
    processed_count = 0
    updated_count = 0
    stopped_reason = "completed"
    selected_query_id = query_id

    while limit is None or processed_count < limit:
        rows = load_review_rows(review_path)
        try:
            row = _select_review_row(rows, query_id=selected_query_id)
        except ValueError as exc:
            stopped_reason = "no_pending" if query_id is None else "not_found"
            if processed_count == 0:
                raise
            break

        report = load_next_review_item(
            review_path=review_path,
            corpus_path=corpus_path,
            query_id=row["query_id"],
        )
        _render_review_item(report, stdout)
        command = _prompt(stdin, stdout, "Action [a]ccept [e]dit [r]eject [s]kip [q]uit: ").strip().lower()
        if command in {"q", "quit"}:
            stopped_reason = "quit"
            break
        if command in {"s", "skip"}:
            stdout.write("Skipped.\n")
            processed_count += 1
            selected_query_id = None
            continue
        if command in {"a", "accept"}:
            notes = _prompt(stdin, stdout, "Notes (optional): ")
            _update_row_in_file(
                review_path,
                query_id=row["query_id"],
                updated_row=apply_review_decision(row, action="accept", notes=notes.strip()),
            )
            stdout.write("Accepted.\n")
            processed_count += 1
            updated_count += 1
            selected_query_id = None
            continue
        if command in {"r", "reject"}:
            notes = _prompt(stdin, stdout, "Notes (optional): ")
            _update_row_in_file(
                review_path,
                query_id=row["query_id"],
                updated_row=apply_review_decision(row, action="reject", notes=notes.strip()),
            )
            stdout.write("Rejected.\n")
            processed_count += 1
            updated_count += 1
            selected_query_id = None
            continue
        if command in {"e", "edit"}:
            edited_query = _prompt(stdin, stdout, "Edited query: ").strip()
            relevant_paper_ids = _prompt(
                stdin,
                stdout,
                f"Relevant paper ids (pipe-separated) [{row.get('positive_ids', '')}]: ",
            ).strip() or row.get("positive_ids", "")
            notes = _prompt(stdin, stdout, "Notes (optional): ").strip()
            _update_row_in_file(
                review_path,
                query_id=row["query_id"],
                updated_row=apply_review_decision(
                    row,
                    action="edit",
                    edited_query=edited_query,
                    relevant_paper_ids=relevant_paper_ids,
                    notes=notes,
                ),
            )
            stdout.write("Edited.\n")
            processed_count += 1
            updated_count += 1
            selected_query_id = None
            continue

        stdout.write("Unknown action. Use a, e, r, s, or q.\n")

    return ReviewLoopReport(
        review_path=str(review_path),
        updated_count=updated_count,
        processed_count=processed_count,
        stopped_reason=stopped_reason,
    )


def _row_to_reviewed_query(
    row: dict[str, str | None],
    *,
    review_path: Path,
    reviewed_at: str,
) -> ReviewedQuery | None:
    query_id = _required_value(row, "query_id")
    query_text = _required_value(row, "query_text")
    style = _required_value(row, "style")
    source_paper_id = _required_value(row, "source_paper_id")
    positive_ids = _split_pipe_values(row.get("positive_ids", ""))
    review_status = _normalize_review_status(_required_value(row, "review_status"))

    if review_status not in FINAL_REVIEW_STATUSES:
        allowed = ", ".join(sorted(FINAL_REVIEW_STATUSES))
        raise ValueError(f"Row {query_id} has invalid review_status {review_status!r}; expected one of {allowed}")
    if review_status == "reject":
        return None
    if review_status == "accept":
        if not positive_ids:
            raise ValueError(f"Row {query_id} must include positive_ids for accept")
        return ReviewedQuery(
            query_id=query_id,
            query_text=query_text,
            style=style,
            source_paper_id=source_paper_id,
            relevant_paper_ids=tuple(positive_ids),
            review_status=review_status,
            notes=(row.get("notes") or "").strip(),
            review_source_csv=str(review_path),
            reviewed_at=reviewed_at,
        )

    edited_query = (row.get("edited_query") or "").strip()
    relevant_paper_ids = _split_pipe_values(row.get("relevant_paper_ids", ""))
    if not edited_query:
        raise ValueError(f"Row {query_id} with status 'edit' must provide edited_query")
    if not relevant_paper_ids:
        raise ValueError(f"Row {query_id} with status 'edit' must provide relevant_paper_ids")
    return ReviewedQuery(
        query_id=query_id,
        query_text=edited_query,
        style=style,
        source_paper_id=source_paper_id,
        relevant_paper_ids=tuple(relevant_paper_ids),
        review_status=review_status,
        notes=(row.get("notes") or "").strip(),
        review_source_csv=str(review_path),
        reviewed_at=reviewed_at,
    )


def _required_value(row: dict[str, str | None], key: str) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Review row is missing required field {key!r}")
    return value


def _split_pipe_values(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def _normalize_review_status(value: str | None) -> str:
    return (value or "").strip().lower()


def _select_review_row(rows: list[dict[str, str]], *, query_id: str | None) -> dict[str, str]:
    if query_id is not None:
        for row in rows:
            if row.get("query_id") == query_id:
                return row
        raise ValueError(f"Review row not found for query_id {query_id!r}")
    for row in rows:
        if _normalize_review_status(row.get("review_status")) == "pending":
            return row
    raise ValueError("No pending review rows found")


def _update_row_in_file(review_path: Path, *, query_id: str, updated_row: dict[str, str]) -> None:
    rows = load_review_rows(review_path)
    for index, row in enumerate(rows):
        if row.get("query_id") == query_id:
            rows[index] = updated_row
            save_review_rows(review_path, rows)
            return
    raise ValueError(f"Review row not found for query_id {query_id!r}")


def _prompt(stdin: TextIO, stdout: TextIO, message: str) -> str:
    stdout.write(message)
    stdout.flush()
    line = stdin.readline()
    if line == "":
        raise ValueError("Interactive review input ended unexpectedly")
    return line.rstrip("\n")


def _render_review_item(report: ReviewNextReport, stdout: TextIO) -> None:
    row = report.review_row
    stdout.write("\n")
    stdout.write(f"Query: {report.query_id}\n")
    stdout.write(f"Style: {row.get('style', '')}\n")
    stdout.write(f"Current status: {row.get('review_status', '')}\n")
    stdout.write(f"Synthetic query: {row.get('query_text', '')}\n")
    if row.get("edited_query"):
        stdout.write(f"Edited query: {row['edited_query']}\n")
    stdout.write(f"Source paper: {report.source_paper['title']}\n")
    stdout.write(f"Abstract: {_truncate_text(str(report.source_paper['abstract']), 320)}\n")
    if report.positive_papers:
        stdout.write("Positives:\n")
        for paper in report.positive_papers:
            stdout.write(f"- {paper['arxiv_id']}: {paper['title']}\n")
    if report.hard_negative_papers:
        stdout.write("Hard negatives:\n")
        for paper in report.hard_negative_papers:
            stdout.write(f"- {paper['arxiv_id']}: {paper['title']}\n")


def _truncate_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _resolve_papers(ids: list[str], papers_by_id: dict[str, ArxivPaper]) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for paper_id in ids:
        paper = papers_by_id.get(paper_id)
        if paper is None:
            continue
        resolved.append(_paper_payload(paper))
    return resolved


def _paper_payload(paper: ArxivPaper) -> dict[str, object]:
    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "published": paper.published,
        "categories": list(paper.categories),
    }


def _count_rejected_rows(review_path: Path) -> int:
    count = 0
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("review_status") or "").strip().lower() == "reject":
                count += 1
    return count


def _count_styles(queries: list[ReviewedQuery]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for query in queries:
        counts[query.style] += 1
    return dict(sorted(counts.items()))
