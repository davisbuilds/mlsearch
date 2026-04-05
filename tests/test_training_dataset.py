from __future__ import annotations

import json
from pathlib import Path

from mlsearch.data.models import ArxivPaper
from mlsearch.training.dataset import (
    build_training_examples,
    compute_sampling_weight,
    expand_training_query_texts,
    sample_training_examples,
)


def test_build_training_examples_maps_query_to_positive_document(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            ArxivPaper(
                arxiv_id="paper-1",
                title="Meta-learning for few-shot classification",
                abstract="Abstract",
                authors=("Alice",),
                categories=("cs.LG",),
                primary_category="cs.LG",
                published="2020-01-01T00:00:00Z",
                updated="2020-01-01T00:00:00Z",
                abs_url="https://arxiv.org/abs/paper-1",
                pdf_url=None,
            ).to_dict()
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = tmp_path / "queries.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "query_id": "paper-1-question",
                "query_text": "what papers study few-shot classification?",
                "style": "question",
                "source_paper_id": "paper-1",
                "source_title": "Meta-learning for few-shot classification",
                "source_published": "2020-01-01T00:00:00Z",
                "positive_ids": ["paper-1"],
                "hard_negative_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = build_training_examples(candidates_path=candidates_path, corpus_path=corpus_path)
    assert len(examples) == 1
    assert examples[0].source_paper_id == "paper-1"
    assert "Meta-learning for few-shot classification" in examples[0].document_text


def test_build_training_examples_excludes_reviewed_eval_queries(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    ArxivPaper(
                        arxiv_id="paper-1",
                        title="Meta-learning for few-shot classification",
                        abstract="Abstract",
                        authors=("Alice",),
                        categories=("cs.LG",),
                        primary_category="cs.LG",
                        published="2020-01-01T00:00:00Z",
                        updated="2020-01-01T00:00:00Z",
                        abs_url="https://arxiv.org/abs/paper-1",
                        pdf_url=None,
                    ).to_dict()
                ),
                json.dumps(
                    ArxivPaper(
                        arxiv_id="paper-2",
                        title="Uncertainty estimation in graph neural networks",
                        abstract="Abstract",
                        authors=("Bob",),
                        categories=("cs.LG",),
                        primary_category="cs.LG",
                        published="2020-01-01T00:00:00Z",
                        updated="2020-01-01T00:00:00Z",
                        abs_url="https://arxiv.org/abs/paper-2",
                        pdf_url=None,
                    ).to_dict()
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = tmp_path / "queries.jsonl"
    candidates_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "paper-1-question",
                        "query_text": "what papers study few-shot classification?",
                        "style": "question",
                        "source_paper_id": "paper-1",
                        "source_title": "Meta-learning for few-shot classification",
                        "source_published": "2020-01-01T00:00:00Z",
                        "positive_ids": ["paper-1"],
                        "hard_negative_ids": [],
                    }
                ),
                json.dumps(
                    {
                        "query_id": "paper-2-question",
                        "query_text": "what papers study uncertainty estimation?",
                        "style": "question",
                        "source_paper_id": "paper-2",
                        "source_title": "Uncertainty estimation in graph neural networks",
                        "source_published": "2020-01-01T00:00:00Z",
                        "positive_ids": ["paper-2"],
                        "hard_negative_ids": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    held_out_eval_path = tmp_path / "held_out_eval.jsonl"
    held_out_eval_path.write_text(
        json.dumps(
            {
                "query_id": "paper-1-question",
                "query_text": "what papers study few-shot classification?",
                "style": "question",
                "source_paper_id": "paper-1",
                "relevant_paper_ids": ["paper-1"],
                "review_status": "accept",
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = build_training_examples(
        candidates_path=candidates_path,
        corpus_path=corpus_path,
        held_out_eval_path=held_out_eval_path,
    )

    assert len(examples) == 1
    assert examples[0].query_id == "paper-2-question"


def test_expand_training_query_texts_augment_question_prefixes() -> None:
    variants = expand_training_query_texts(
        "papers on graph neural networks for chemistry",
        style="question",
        question_prefix_augmentation=True,
    )

    assert variants == [
        "papers on graph neural networks for chemistry",
        "research on graph neural networks for chemistry",
        "work on graph neural networks for chemistry",
    ]


def test_build_training_examples_augment_question_queries_and_sample_deterministically(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    ArxivPaper(
                        arxiv_id="paper-1",
                        title="Meta-learning for few-shot classification",
                        abstract="Abstract",
                        authors=("Alice",),
                        categories=("cs.LG",),
                        primary_category="cs.LG",
                        published="2020-01-01T00:00:00Z",
                        updated="2020-01-01T00:00:00Z",
                        abs_url="https://arxiv.org/abs/paper-1",
                        pdf_url=None,
                    ).to_dict()
                ),
                json.dumps(
                    ArxivPaper(
                        arxiv_id="paper-2",
                        title="Uncertainty estimation in graph neural networks",
                        abstract="Abstract",
                        authors=("Bob",),
                        categories=("cs.LG",),
                        primary_category="cs.LG",
                        published="2020-01-01T00:00:00Z",
                        updated="2020-01-01T00:00:00Z",
                        abs_url="https://arxiv.org/abs/paper-2",
                        pdf_url=None,
                    ).to_dict()
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = tmp_path / "queries.jsonl"
    candidates_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "paper-1-question",
                        "query_text": "papers on few-shot classification",
                        "style": "question",
                        "source_paper_id": "paper-1",
                        "source_title": "Meta-learning for few-shot classification",
                        "source_published": "2020-01-01T00:00:00Z",
                        "positive_ids": ["paper-1"],
                        "hard_negative_ids": [],
                    }
                ),
                json.dumps(
                    {
                        "query_id": "paper-2-keyword",
                        "query_text": "uncertainty estimation graph neural networks",
                        "style": "keyword",
                        "source_paper_id": "paper-2",
                        "source_title": "Uncertainty estimation in graph neural networks",
                        "source_published": "2020-01-01T00:00:00Z",
                        "positive_ids": ["paper-2"],
                        "hard_negative_ids": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first = build_training_examples(
        candidates_path=candidates_path,
        corpus_path=corpus_path,
        max_examples=3,
        seed=7,
        question_prefix_augmentation=True,
    )
    second = build_training_examples(
        candidates_path=candidates_path,
        corpus_path=corpus_path,
        max_examples=3,
        seed=7,
        question_prefix_augmentation=True,
    )

    assert len(first) == 3
    assert [example.query_id for example in first] == [example.query_id for example in second]
    assert any(example.query_id.endswith("-aug1") for example in first)
    assert any(example.style == "question" for example in first)


def test_compute_sampling_weight_prefers_broader_question_queries() -> None:
    broad_question_weight = compute_sampling_weight(
        "papers on traffic routing ml",
        source_title="Online Learning for Traffic Routing under Unknown Preferences",
        style="question",
        hard_query_pattern_weighting=True,
    )
    title_like_keyword_weight = compute_sampling_weight(
        "online learning traffic routing unknown preferences",
        source_title="Online Learning for Traffic Routing under Unknown Preferences",
        style="keyword",
        hard_query_pattern_weighting=True,
    )

    assert broad_question_weight > title_like_keyword_weight


def test_sample_training_examples_is_deterministic_and_weighted() -> None:
    examples = [
        type("Example", (), {})(),
        type("Example", (), {})(),
        type("Example", (), {})(),
    ]
    examples[0].sampling_weight = 3.5
    examples[0].query_id = "high"
    examples[1].sampling_weight = 1.0
    examples[1].query_id = "mid"
    examples[2].sampling_weight = 0.5
    examples[2].query_id = "low"

    first = sample_training_examples(examples, max_examples=2, seed=11)
    second = sample_training_examples(examples, max_examples=2, seed=11)

    assert [example.query_id for example in first] == [example.query_id for example in second]
    assert "high" in {example.query_id for example in first}
