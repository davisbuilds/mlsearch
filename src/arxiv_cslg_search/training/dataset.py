from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arxiv_cslg_search.data.models import ArxivPaper
from arxiv_cslg_search.pipelines.generate_queries import load_query_candidates
from arxiv_cslg_search.pipelines.validate_corpus import load_corpus
from arxiv_cslg_search.retrieval.index import format_document


@dataclass(frozen=True)
class TrainingExample:
    query_id: str
    query_text: str
    source_paper_id: str
    document_text: str


def build_training_examples(
    *,
    candidates_path: Path,
    corpus_path: Path,
    max_examples: int | None = None,
) -> list[TrainingExample]:
    candidates = load_query_candidates(candidates_path)
    papers = load_corpus(corpus_path)
    papers_by_id = {paper.arxiv_id: paper for paper in papers}

    examples: list[TrainingExample] = []
    for candidate in candidates:
        if not candidate.positive_ids:
            continue
        positive_id = candidate.positive_ids[0]
        paper = papers_by_id.get(positive_id)
        if paper is None:
            continue
        examples.append(
            TrainingExample(
                query_id=candidate.query_id,
                query_text=candidate.query_text,
                source_paper_id=positive_id,
                document_text=format_document(paper),
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples
