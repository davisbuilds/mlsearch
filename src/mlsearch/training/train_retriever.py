from __future__ import annotations

import json
import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader

from mlsearch.config import TrainConfig, load_train_config
from mlsearch.paths import PATHS
from mlsearch.retrieval.embedder import detect_device
from mlsearch.training.checkpoints import create_run_dir
from mlsearch.training.dataset import build_training_examples


@dataclass(frozen=True)
class TrainReport:
    model_dir: str
    summary_path: str
    example_count: int
    base_model_name: str
    device: str
    seed: int


def train_retriever(*, config_path: Path | None = None, config: TrainConfig | None = None) -> TrainReport:
    if (config_path is None) == (config is None):
        raise ValueError("Provide exactly one of config_path or config.")
    resolved_config = load_train_config(config_path) if config_path is not None else config
    assert resolved_config is not None
    examples = build_training_examples(
        candidates_path=PATHS.data_benchmark / "generated" / "query_candidates.jsonl",
        corpus_path=PATHS.data_processed / "corpus.jsonl",
        held_out_eval_path=PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl",
        max_examples=resolved_config.max_examples,
        seed=resolved_config.seed,
        question_prefix_augmentation=resolved_config.question_prefix_augmentation,
    )
    if not examples:
        raise ValueError("No training examples available. Generate benchmark queries first.")

    device = resolve_train_device(resolved_config.device)
    _seed_training(resolved_config.seed)
    model = SentenceTransformer(resolved_config.base_model_name, device=device)
    train_examples = [InputExample(texts=[example.query_text, example.document_text]) for example in examples]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=resolved_config.batch_size,
        collate_fn=model.smart_batching_collate,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=resolved_config.learning_rate)

    run_dir = create_run_dir(resolved_config.run_name_prefix)
    model.train()
    losses_seen: list[float] = []
    for _ in range(resolved_config.num_epochs):
        for sentence_features, labels in train_dataloader:
            sentence_features = [batch_to_device(features, model.device) for features in sentence_features]
            if isinstance(labels, torch.Tensor):
                labels = labels.to(model.device)
            optimizer.zero_grad()
            loss_value = train_loss(sentence_features, labels)
            loss_value.backward()
            optimizer.step()
            losses_seen.append(float(loss_value.detach().cpu()))
    model.save(str(run_dir))

    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "base_model_name": resolved_config.base_model_name,
                "config": asdict(resolved_config),
                "device": device,
                "example_count": len(examples),
                "style_counts": {
                    style: sum(1 for example in examples if example.style == style)
                    for style in sorted({example.style for example in examples})
                },
                "num_epochs": resolved_config.num_epochs,
                "batch_size": resolved_config.batch_size,
                "learning_rate": resolved_config.learning_rate,
                "seed": resolved_config.seed,
                "warmup_ratio": resolved_config.warmup_ratio,
                "mean_train_loss": sum(losses_seen) / len(losses_seen),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TrainReport(
        model_dir=str(run_dir),
        summary_path=str(summary_path),
        example_count=len(examples),
        base_model_name=resolved_config.base_model_name,
        device=device,
        seed=resolved_config.seed,
    )


def resolve_train_device(configured_device: str) -> str:
    if configured_device == "auto":
        return detect_device()
    if configured_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Train config requested mps, but MPS is not available on this machine.")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Train config requested cuda, but CUDA is not available on this machine.")
    return configured_device


def _seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
