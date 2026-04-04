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
ABSTRACT_NOISE = {
    "almost",
    "approach",
    "approaches",
    "application",
    "applications",
    "alternative",
    "alternatives",
    "are",
    "as",
    "asks",
    "been",
    "being",
    "but",
    "can",
    "class",
    "classes",
    "combine",
    "combines",
    "deserved",
    "digital",
    "enable",
    "enabled",
    "enjoying",
    "era",
    "especially",
    "every",
    "gained",
    "had",
    "happens",
    "has",
    "have",
    "important",
    "introduce",
    "introduces",
    "investigate",
    "investigates",
    "is",
    "method",
    "methods",
    "model",
    "models",
    "new",
    "novel",
    "our",
    "paper",
    "presents",
    "present",
    "problem",
    "problems",
    "occur",
    "occurs",
    "often",
    "offer",
    "offering",
    "offers",
    "propose",
    "proposes",
    "rapidly",
    "remains",
    "results",
    "short",
    "show",
    "shows",
    "studies",
    "study",
    "task",
    "tasks",
    "this",
    "too",
    "through",
    "technique",
    "techniques",
    "towards",
    "two",
    "use",
    "used",
    "well",
    "whether",
    "when",
    "which",
    "work",
    "achieves",
    "aimed",
    "evaluate",
    "implement",
    "implements",
    "long",
    "most",
    "or",
    "people",
    "related",
    "that",
}
TITLE_LEAD_NOISE = {
    "accurate",
    "efficient",
    "frontier",
    "improved",
    "new",
    "novel",
    "observations",
    "online",
    "robust",
    "towards",
}
TITLE_TAIL_NOISE = {
    "approach",
    "paper",
    "study",
    "tool",
    "tools",
}
TITLE_CONNECTORS = ("for", "using", "via", "with", "in")
QUERY_PREFIXES = ("papers on", "research on", "work on")
QUERY_NOISE = ABSTRACT_NOISE | {"tool", "tools", "visualisation", "visualization"}


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
    return write_query_candidates(
        candidates,
        generated_dir=PATHS.data_benchmark / "generated",
        seed=config.seed,
        max_candidates=config.max_candidates,
    )


def write_query_candidates(
    candidates: list[QueryCandidate],
    *,
    generated_dir: Path | None = None,
    seed: int = 0,
    max_candidates: int = 0,
) -> QueryGenerationReport:
    resolved_generated_dir = generated_dir or (PATHS.data_benchmark / "generated")
    resolved_generated_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = resolved_generated_dir / "query_candidates.jsonl"
    manifest_path = resolved_generated_dir / "benchmark_manifest.json"

    with candidates_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.to_dict(), sort_keys=True) + "\n")

    styles = count_styles(candidates)
    diagnostics = compute_query_diagnostics(candidates)
    manifest_path.write_text(
        json.dumps(
            {
                "count": len(candidates),
                "diagnostics": diagnostics,
                "styles": styles,
                "seed": seed,
                "max_candidates": max_candidates,
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
    title_tokens = descriptive_title_tokens(paper.title)
    abstract_tokens = ordered_abstract_tokens(clean_abstract_topic(paper.abstract))
    title_query_tokens = build_title_query_tokens(title_tokens)
    abstract_query_tokens = build_abstract_query_tokens(
        title_tokens=title_tokens,
        abstract_tokens=abstract_tokens,
    )

    if title_query_tokens and is_clean_title_query(title_query_tokens):
        tokens = title_query_tokens[:5]
    elif abstract_query_tokens:
        tokens = abstract_query_tokens[:5]
    else:
        tokens = select_best_query_tokens(
            candidates=[
                title_query_tokens,
                abstract_query_tokens,
                dedupe_tokens(abstract_tokens[:5] + title_tokens[-2:]),
            ],
            source_title=paper.title,
        )
    if len(tokens) < 3:
        tokens = title_tokens[:5] or abstract_tokens[:5]
    return " ".join(tokens[:5])


def build_question_query(paper: ArxivPaper) -> str:
    topic = build_keyword_query(paper)
    if not topic:
        return ""
    prefix = QUERY_PREFIXES[sum(ord(char) for char in paper.arxiv_id) % len(QUERY_PREFIXES)]
    return f"{prefix} {topic}"


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


def descriptive_title_tokens(title: str) -> list[str]:
    if ":" in title:
        _, descriptive = title.split(":", maxsplit=1)
        tokens = ordered_keyword_tokens(descriptive)
        if tokens:
            return trim_title_lead_noise(tokens)
    return trim_title_lead_noise(ordered_keyword_tokens(title))


def normalize_title_topic(title: str) -> str:
    topic = re.sub(r"[^a-z0-9\s-]", "", title.lower()).strip()
    return " ".join(topic.split())


def clean_abstract_topic(abstract: str) -> str:
    sentence = first_sentence(abstract)
    lowered = sentence.lower().strip()
    lowered = re.sub(
        r"^(objective|background|purpose|aim|aims|motivation|introduction|methods?|results?|conclusions?)\s*:\s*",
        "",
        lowered,
    )
    prefixes = (
        "this paper studies ",
        "this paper presents ",
        "this paper proposes ",
        "this paper introduces ",
        "this paper investigates ",
        "this study aimed to evaluate ",
        "this study aims to evaluate ",
        "this study aimed to ",
        "this study aims to ",
        "this study evaluates ",
        "this study investigates ",
        "in this work we ",
        "in this work, we ",
        "we study ",
        "we present ",
        "we propose ",
        "we introduce ",
        "we investigate ",
        "we evaluate ",
        "our work studies ",
        "our work proposes ",
        "our method studies ",
        "recent advances in ",
        "recent progress in ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    return lowered


def ordered_abstract_tokens(text: str) -> list[str]:
    return [token for token in ordered_keyword_tokens(text) if token not in ABSTRACT_NOISE]


def first_sentence(text: str) -> str:
    parts = re.split(r"[.!?]\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text.strip()


def title_overlap_ratio(query_text: str, source_title: str) -> float:
    query_tokens = keyword_tokens(query_text)
    title_tokens = keyword_tokens(source_title)
    if not query_tokens or not title_tokens:
        return 0.0
    return len(query_tokens & title_tokens) / len(query_tokens | title_tokens)


def select_salient_ngram(abstract_tokens: list[str], title_token_set: set[str]) -> list[str]:
    if not abstract_tokens:
        return []

    best_tokens: list[str] = []
    best_score = float("-inf")
    for width in range(2, min(4, len(abstract_tokens)) + 1):
        for start in range(0, len(abstract_tokens) - width + 1):
            tokens = abstract_tokens[start : start + width]
            novelty = sum(1 for token in tokens if token not in title_token_set)
            title_overlap = width - novelty
            score = (novelty * 3.0) + width - (title_overlap * 0.75)
            if novelty == 0:
                score -= 2.0
            if score > best_score:
                best_score = score
                best_tokens = tokens
    return best_tokens or abstract_tokens[:3]


def build_abstract_query_tokens(*, title_tokens: list[str], abstract_tokens: list[str]) -> list[str]:
    salient_tokens = select_salient_ngram(abstract_tokens, set(title_tokens))
    context_tokens = select_context_tokens(
        salient_tokens=salient_tokens,
        title_tokens=title_tokens,
        abstract_tokens=abstract_tokens,
    )
    tokens = dedupe_tokens(salient_tokens + context_tokens)
    if len(tokens) < 4:
        tokens = dedupe_tokens(abstract_tokens[:5] + title_tokens[-2:])
    return tokens[:5]


def build_title_query_tokens(title_tokens: list[str]) -> list[str]:
    if not title_tokens:
        return []

    connector_index = next((index for index, token in enumerate(title_tokens) if token in TITLE_CONNECTORS), -1)
    if connector_index == -1:
        return trim_title_tail_noise(trim_title_lead_noise(title_tokens))[:5]

    left = trim_title_tail_noise(title_tokens[:connector_index])
    right = trim_title_tail_noise(title_tokens[connector_index + 1 :])
    connector = title_tokens[connector_index]

    if connector == "for":
        if left and left[-1] in {"framework", "frameworks", "tool", "tools", "system", "systems"} and right:
            return dedupe_tokens(right[:4] + [left[-1]])
        if right:
            return dedupe_tokens(right[:4] + left[-2:])
    if connector in {"using", "via", "with"}:
        if left and right:
            return dedupe_tokens(left[:2] + right[:3])
        if left:
            return left[:5]
    if connector == "in":
        if left and right:
            return dedupe_tokens(left[:4] + right[:2])

    return trim_title_tail_noise((left + right)[:5])


def trim_title_tail_noise(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while trimmed and trimmed[-1] in TITLE_TAIL_NOISE:
        trimmed = trimmed[:-1]
    return trimmed or tokens


def select_best_query_tokens(*, candidates: list[list[str]], source_title: str) -> list[str]:
    best_tokens: list[str] = []
    best_score = float("-inf")
    for candidate in candidates:
        tokens = dedupe_tokens(candidate)[:5]
        if len(tokens) < 3:
            continue
        overlap = title_overlap_ratio(" ".join(tokens), source_title)
        score = len(tokens) * 1.5
        if 0.15 <= overlap <= 0.65:
            score += 2.5
        else:
            score -= abs(overlap - 0.4) * 4.0
        score -= sum(1.25 for token in tokens if token in QUERY_NOISE)
        if len(set(tokens)) != len(tokens):
            score -= 1.0
        if score > best_score:
            best_score = score
            best_tokens = tokens
    return best_tokens


def is_clean_title_query(tokens: list[str]) -> bool:
    if len(tokens) < 3:
        return False
    if any(token in {"tool", "tools", "visualisation", "visualization"} for token in tokens):
        return False
    noise_count = sum(1 for token in tokens if token in QUERY_NOISE)
    return noise_count <= 1


def select_context_tokens(
    *,
    salient_tokens: list[str],
    title_tokens: list[str],
    abstract_tokens: list[str],
) -> list[str]:
    salient_set = set(salient_tokens)
    abstract_set = set(abstract_tokens)
    title_context = [
        token
        for token in title_tokens
        if token in abstract_set and token not in salient_set
    ]
    if len(title_context) >= 2:
        return title_context[-2:]
    fallback_context = [
        token
        for token in abstract_tokens
        if token not in salient_set
    ]
    if title_context:
        return dedupe_tokens(fallback_context + title_context)[:2]
    return fallback_context[:2]


def trim_title_lead_noise(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while trimmed and trimmed[0] in TITLE_LEAD_NOISE:
        trimmed = trimmed[1:]
    return trimmed or tokens


def dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def compute_query_diagnostics(candidates: list[QueryCandidate]) -> dict[str, object]:
    by_style: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        by_style[candidate.style].append(title_overlap_ratio(candidate.query_text, candidate.source_title))

    style_overlap: dict[str, dict[str, float | int]] = {}
    for style, overlaps in sorted(by_style.items()):
        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        style_overlap[style] = {
            "count": len(overlaps),
            "max_title_overlap": max(overlaps) if overlaps else 0.0,
            "mean_title_overlap": mean_overlap,
            "queries_at_or_above_0_8_overlap": sum(1 for item in overlaps if item >= 0.8),
        }
    return {
        "count": len(candidates),
        "style_overlap": style_overlap,
    }


def count_styles(candidates: list[QueryCandidate]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate.style] += 1
    return dict(sorted(counts.items()))
