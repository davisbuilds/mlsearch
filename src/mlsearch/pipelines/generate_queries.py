from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mlsearch.benchmark.schema import QueryCandidate
from mlsearch.config import BenchmarkConfig, load_benchmark_config
from mlsearch.data.models import ArxivPaper
from mlsearch.paths import PATHS
from mlsearch.pipelines.validate_corpus import load_corpus

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "using",
    "with",
}


@dataclass(frozen=True)
class QueryGenerationReport:
    candidates_path: str
    manifest_path: str
    count: int
    styles: dict[str, int]


def generate_queries(*, config_path: Path, corpus_path: Path | None = None) -> QueryGenerationReport:
    config = load_benchmark_config(config_path)
    papers = load_corpus(corpus_path or (PATHS.data_processed / "corpus.jsonl"))
    candidates = build_query_candidates(papers, config)

    generated_dir = PATHS.data_benchmark / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = generated_dir / "query_candidates.jsonl"
    manifest_path = generated_dir / "benchmark_manifest.json"

    with candidates_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")

    styles = count_styles(candidates)
    manifest_path.write_text(
        json.dumps(
            {
                "count": len(candidates),
                "styles": styles,
                "seed": config.seed,
                "max_candidates": config.max_candidates,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return QueryGenerationReport(
        candidates_path=str(candidates_path),
        manifest_path=str(manifest_path),
        count=len(candidates),
        styles=styles,
    )


def build_query_candidates(papers: list[ArxivPaper], config: BenchmarkConfig) -> list[QueryCandidate]:
    sampled_papers = select_source_papers(papers, config.max_candidates, config.seed)
    title_tokens = {paper.arxiv_id: keyword_tokens(paper.title) for paper in papers}
    token_index = build_token_index(title_tokens)

    candidates: list[QueryCandidate] = []
    for paper in sampled_papers:
        negatives = lexical_hard_negatives(
            paper,
            title_tokens=title_tokens,
            token_index=token_index,
            limit=config.negatives_per_query,
        )
        keyword_query = build_keyword_query(paper)
        question_query = build_question_query(paper)

        if keyword_query:
            candidates.append(
                QueryCandidate(
                    query_id=f"{paper.arxiv_id}-keyword",
                    query_text=keyword_query,
                    style="keyword",
                    source_paper_id=paper.arxiv_id,
                    source_title=paper.title,
                    source_published=paper.published,
                    positive_ids=(paper.arxiv_id,),
                    hard_negative_ids=tuple(negatives),
                )
            )
        if question_query:
            candidates.append(
                QueryCandidate(
                    query_id=f"{paper.arxiv_id}-question",
                    query_text=question_query,
                    style="question",
                    source_paper_id=paper.arxiv_id,
                    source_title=paper.title,
                    source_published=paper.published,
                    positive_ids=(paper.arxiv_id,),
                    hard_negative_ids=tuple(negatives),
                )
            )
    return candidates


def load_query_candidates(path: Path) -> list[QueryCandidate]:
    with path.open("r", encoding="utf-8") as handle:
        return [QueryCandidate.from_dict(json.loads(line)) for line in handle]


def select_source_papers(papers: list[ArxivPaper], max_candidates: int, seed: int) -> list[ArxivPaper]:
    if len(papers) <= max_candidates:
        return list(papers)
    rng = random.Random(seed)
    sampled = rng.sample(papers, max_candidates)
    return sorted(sampled, key=lambda paper: (paper.published, paper.arxiv_id), reverse=True)


def build_token_index(title_tokens: dict[str, set[str]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for paper_id, tokens in title_tokens.items():
        for token in tokens:
            index[token].add(paper_id)
    return index


def lexical_hard_negatives(
    paper: ArxivPaper,
    *,
    title_tokens: dict[str, set[str]],
    token_index: dict[str, set[str]],
    limit: int,
) -> list[str]:
    tokens = title_tokens.get(paper.arxiv_id, set())
    candidate_ids: set[str] = set()
    for token in tokens:
        candidate_ids.update(token_index.get(token, set()))
    candidate_ids.discard(paper.arxiv_id)

    scored: list[tuple[float, str]] = []
    for candidate_id in candidate_ids:
        other_tokens = title_tokens[candidate_id]
        union = tokens | other_tokens
        score = len(tokens & other_tokens) / len(union) if union else 0.0
        scored.append((score, candidate_id))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [candidate_id for _, candidate_id in scored[:limit]]


def build_keyword_query(paper: ArxivPaper) -> str:
    tokens = ordered_keyword_tokens(paper.title)
    if not tokens:
        tokens = ordered_keyword_tokens(paper.abstract)
    return " ".join(tokens[:5])


def build_question_query(paper: ArxivPaper) -> str:
    topic = normalize_title_topic(paper.title)
    if not topic:
        return ""
    return f"what papers study {topic}?"


def keyword_tokens(text: str) -> set[str]:
    tokens = {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS}
    return tokens


def ordered_keyword_tokens(text: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def normalize_title_topic(title: str) -> str:
    topic = re.sub(r"[^a-z0-9\s-]", "", title.lower()).strip()
    return " ".join(topic.split())


def count_styles(candidates: list[QueryCandidate]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate.style] += 1
    return dict(sorted(counts.items()))
