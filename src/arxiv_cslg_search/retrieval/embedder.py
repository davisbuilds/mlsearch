from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 32
    normalize_embeddings: bool = True


class TextEmbedder:
    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self.device = detect_device()
        self.model = SentenceTransformer(self.config.model_name, device=self.device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prepared = [DEFAULT_QUERY_PREFIX + text for text in texts]
        return self._encode(prepared)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
