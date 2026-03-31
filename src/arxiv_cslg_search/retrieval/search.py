from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arxiv_cslg_search.data.models import ArxivPaper
from arxiv_cslg_search.retrieval.embedder import EmbedderConfig, TextEmbedder
from arxiv_cslg_search.retrieval.index import load_index


@dataclass(frozen=True)
class SearchHit:
    arxiv_id: str
    title: str
    published: str
    score: float


def search_index(query: str, *, top_k: int, index_dir: Path | None = None) -> list[SearchHit]:
    papers, embeddings, manifest = load_index(index_dir)
    embedder = TextEmbedder(EmbedderConfig(model_name=str(manifest["model_name"])))
    query_embedding = embedder.embed_queries([query])[0]
    return rank_hits(query_embedding, embeddings, papers, top_k=top_k)


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
