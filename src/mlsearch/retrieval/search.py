from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlsearch.data.models import ArxivPaper
from mlsearch.paths import PATHS
from mlsearch.retrieval.embedder import EmbedderConfig, TextEmbedder
from mlsearch.retrieval.index import load_index


@dataclass(frozen=True)
class SearchHit:
    arxiv_id: str
    title: str
    published: str
    score: float


def search_index(
    query: str,
    *,
    top_k: int,
    index_dir: Path | None = None,
    reranker_model_name: str | None = None,
    rerank_depth: int = 10,
) -> list[SearchHit]:
    resolved_index_dir = index_dir or PATHS.artifacts_index
    papers, embeddings, manifest = load_index(resolved_index_dir)
    embedder = TextEmbedder(EmbedderConfig(model_name=str(manifest["model_name"])))
    query_embedding = embedder.embed_queries([query])[0]
    hits = rank_hits(query_embedding, embeddings, papers, top_k=top_k)
    if reranker_model_name is None:
        return hits

    from mlsearch.retrieval.rerank import RerankerConfig, rerank_hit_lists

    reranked = rerank_hit_lists(
        [query],
        [hits],
        index_dir=resolved_index_dir,
        config=RerankerConfig(model_name=reranker_model_name, rerank_depth=rerank_depth),
    )
    return reranked[0]


def search_many(
    queries: list[str],
    *,
    top_k: int,
    index_dir: Path | None = None,
) -> list[list[SearchHit]]:
    papers, embeddings, manifest = load_index(index_dir)
    embedder = TextEmbedder(EmbedderConfig(model_name=str(manifest["model_name"])))
    query_embeddings = embedder.embed_queries(queries)
    return [rank_hits(query_embedding, embeddings, papers, top_k=top_k) for query_embedding in query_embeddings]


def rank_hits(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    papers: list[ArxivPaper],
    *,
    top_k: int,
) -> list[SearchHit]:
    scores = document_embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    hits: list[SearchHit] = []
    for index in ranked_indices:
        paper = papers[int(index)]
        hits.append(
            SearchHit(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                published=paper.published,
                score=float(scores[int(index)]),
            )
        )
    return hits
