from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from mlsearch.benchmark.review import load_reviewed_queries
from mlsearch.data.models import ArxivPaper
from mlsearch.pipelines.generate_queries import QUERY_PREFIXES, load_query_candidates
from mlsearch.pipelines.validate_corpus import load_corpus
from mlsearch.retrieval.index import format_document


@dataclass(frozen=True)
class TrainingExample:
    query_id: str
    query_text: str
    source_paper_id: str
    document_text: str
    style: str


def build_training_examples(
    *,
    candidates_path: Path,
    corpus_path: Path,
    held_out_eval_path: Path | None = None,
    max_examples: int | None = None,
    seed: int = 0,
    question_prefix_augmentation: bool = False,
) -> list[TrainingExample]:
    candidates = load_query_candidates(candidates_path)
    papers = load_corpus(corpus_path)
    papers_by_id = {paper.arxiv_id: paper for paper in papers}
    held_out_query_ids = _load_held_out_query_ids(held_out_eval_path)

    examples: list[TrainingExample] = []
    for candidate in candidates:
        if candidate.query_id in held_out_query_ids:
            continue
        if not candidate.positive_ids:
            continue
        positive_id = candidate.positive_ids[0]
        paper = papers_by_id.get(positive_id)
        if paper is None:
            continue
        for variant_index, query_text in enumerate(
            expand_training_query_texts(
                candidate.query_text,
                style=candidate.style,
                question_prefix_augmentation=question_prefix_augmentation,
            )
        ):
            query_id = candidate.query_id if variant_index == 0 else f"{candidate.query_id}-aug{variant_index}"
            examples.append(
                TrainingExample(
                    query_id=query_id,
                    query_text=query_text,
                    source_paper_id=positive_id,
                    document_text=format_document(paper),
                    style=candidate.style,
                )
            )
    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        sampled_examples = list(examples)
        rng.shuffle(sampled_examples)
        return sampled_examples[:max_examples]
    return examples


def _load_held_out_query_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {query.query_id for query in load_reviewed_queries(path)}


def expand_training_query_texts(
    query_text: str,
    *,
    style: str,
    question_prefix_augmentation: bool,
) -> list[str]:
    variants = [query_text]
    if style != "question" or not question_prefix_augmentation:
        return variants

    topic = strip_question_prefix(query_text)
    if not topic:
        return variants

    seen = {query_text}
    for prefix in QUERY_PREFIXES:
        variant = f"{prefix} {topic}"
        if variant in seen:
            continue
        seen.add(variant)
        variants.append(variant)
    return variants


def strip_question_prefix(query_text: str) -> str:
    lowered = query_text.lower()
    for prefix in QUERY_PREFIXES:
        needle = f"{prefix} "
        if lowered.startswith(needle):
            return query_text[len(needle) :].strip()
    return ""
