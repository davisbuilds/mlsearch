from __future__ import annotations

import json
from pathlib import Path

from arxiv_cslg_search.benchmark.schema import QueryCandidate, ReviewedQuery
from arxiv_cslg_search.eval.run_eval import ensure_baseline_compatible, resolve_eval_candidates


def test_resolve_eval_candidates_prefers_reviewed_split(tmp_path: Path) -> None:
    generated_path = tmp_path / "query_candidates.jsonl"
    generated_path.write_text(
        json.dumps(
            QueryCandidate(
                query_id="paper-1-keyword",
                query_text="few-shot meta learning",
                style="keyword",
                source_paper_id="paper-1",
                source_title="Meta-learning for few-shot classification",
                source_published="2020-01-01T00:00:00Z",
                positive_ids=("paper-1",),
                hard_negative_ids=("paper-2",),
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )
    reviewed_path = tmp_path / "held_out_eval.jsonl"
    reviewed_path.write_text(
        json.dumps(
            ReviewedQuery(
                query_id="paper-9-question",
                query_text="what papers study distribution shift?",
                style="question",
                source_paper_id="paper-9",
                relevant_paper_ids=("paper-9",),
                review_status="accept",
                notes="",
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )

    selected_path, selected_queries = resolve_eval_candidates(
        generated_candidates_path=generated_path,
        reviewed_eval_path=reviewed_path,
    )

    assert selected_path == reviewed_path
    assert [query.query_id for query in selected_queries] == ["paper-9-question"]


def test_ensure_baseline_compatible_rejects_mismatched_split() -> None:
    baseline_report = {
        "candidates_path": "/tmp/generated/query_candidates.jsonl",
        "metrics": {"recall@10": 0.5, "mrr": 0.5, "ndcg@10": 0.5},
    }

    try:
        ensure_baseline_compatible(
            baseline_report,
            candidates_path=Path("/tmp/reviewed/held_out_eval.jsonl"),
        )
    except ValueError as exc:
        assert "rerun `eval baseline`" in str(exc)
    else:
        raise AssertionError("expected baseline compatibility check to fail")
