from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arxiv_cslg_search.data.models import ArxivPaper
from arxiv_cslg_search.paths import PATHS
from arxiv_cslg_search.pipelines.validate_corpus import load_corpus
from arxiv_cslg_search.retrieval.embedder import DEFAULT_MODEL_NAME, EmbedderConfig, TextEmbedder


@dataclass(frozen=True)
class IndexBuildReport:
    index_dir: str
    manifest_path: str
    count: int
    model_name: str
    embedding_dim: int


def build_index(
    *,
    corpus_path: Path | None = None,
    output_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> IndexBuildReport:
    papers = load_corpus(corpus_path or (PATHS.data_processed / "corpus.jsonl"))
    index_dir = output_dir or PATHS.artifacts_index
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(EmbedderConfig(model_name=model_name))
    document_texts = [format_document(paper) for paper in papers]
    embeddings = embedder.embed_documents(document_texts)

    embeddings_path = index_dir / "embeddings.npy"
    documents_path = index_dir / "documents.jsonl"
    manifest_path = index_dir / "index_manifest.json"

    np.save(embeddings_path, embeddings)
    with documents_path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper.to_dict(), sort_keys=True) + "\n")

    manifest_path.write_text(
        json.dumps(
            {
                "count": len(papers),
                "model_name": model_name,
                "embedding_dim": int(embeddings.shape[1]),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return IndexBuildReport(
        index_dir=str(index_dir),
        manifest_path=str(manifest_path),
        count=len(papers),
        model_name=model_name,
        embedding_dim=int(embeddings.shape[1]),
    )


def load_index(index_dir: Path | None = None) -> tuple[list[ArxivPaper], np.ndarray, dict[str, object]]:
    resolved_dir = index_dir or PATHS.artifacts_index
    documents_path = resolved_dir / "documents.jsonl"
    embeddings_path = resolved_dir / "embeddings.npy"
    manifest_path = resolved_dir / "index_manifest.json"

    papers = load_corpus(documents_path)
    embeddings = np.load(embeddings_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return papers, embeddings, manifest


def format_document(paper: ArxivPaper) -> str:
    return f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
