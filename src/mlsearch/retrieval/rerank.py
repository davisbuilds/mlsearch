from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import CrossEncoder

from mlsearch.retrieval.embedder import detect_device
from mlsearch.retrieval.index import format_document, load_index
from mlsearch.retrieval.search import SearchHit

DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = DEFAULT_RERANKER_MODEL_NAME
    batch_size: int = 32
    rerank_depth: int = 10


def rerank_hit_lists(
    queries: list[str],
    hits_per_query: list[list[SearchHit]],
    *,
    index_dir: Path,
    config: RerankerConfig | None = None,
) -> list[list[SearchHit]]:
    resolved_config = config or RerankerConfig()
    papers, _, _ = load_index(index_dir)
    documents_by_id = {paper.arxiv_id: format_document(paper) for paper in papers}
    reranker = CrossEncoder(resolved_config.model_name, device=detect_device())

    pairs: list[tuple[str, str]] = []
    depths: list[int] = []
    for query, hits in zip(queries, hits_per_query, strict=True):
        depth = min(resolved_config.rerank_depth, len(hits))
        depths.append(depth)
        for hit in hits[:depth]:
            document_text = documents_by_id.get(hit.arxiv_id)
            if document_text is None:
                raise KeyError(f"Missing document text for rerank candidate: {hit.arxiv_id}")
            pairs.append((query, document_text))

    if not pairs:
        return hits_per_query

    scores = list(
        reranker.predict(
            pairs,
            batch_size=resolved_config.batch_size,
            show_progress_bar=False,
        )
    )

    reranked: list[list[SearchHit]] = []
    offset = 0
    for hits, depth in zip(hits_per_query, depths, strict=True):
        head_scores = scores[offset : offset + depth]
        reranked_head = rerank_hits_with_scores(hits[:depth], head_scores)
        reranked.append(reranked_head + hits[depth:])
        offset += depth
    return reranked


def rerank_hits_with_scores(hits: list[SearchHit], scores: list[float]) -> list[SearchHit]:
    ranked = sorted(
        zip(hits, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        SearchHit(
            arxiv_id=hit.arxiv_id,
            title=hit.title,
            published=hit.published,
            score=float(score),
        )
        for hit, score in ranked
    ]
