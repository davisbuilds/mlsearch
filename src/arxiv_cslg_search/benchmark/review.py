from __future__ import annotations

import csv
from pathlib import Path

from arxiv_cslg_search.benchmark.schema import QueryCandidate


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
