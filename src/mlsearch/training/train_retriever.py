from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader

from mlsearch.config import load_train_config
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


def train_retriever(*, config_path: Path) -> TrainReport:
    config = load_train_config(config_path)
    examples = build_training_examples(
        candidates_path=PATHS.data_benchmark / "generated" / "query_candidates.jsonl",
        corpus_path=PATHS.data_processed / "corpus.jsonl",
        held_out_eval_path=PATHS.data_benchmark / "reviewed" / "held_out_eval.jsonl",
        max_examples=config.max_examples,
    )
    if not examples:
        raise ValueError("No training examples available. Generate benchmark queries first.")

    device = resolve_train_device(config.device)
    model = SentenceTransformer(config.base_model_name, device=device)
    train_examples = [InputExample(texts=[example.query_text, example.document_text]) for example in examples]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=model.smart_batching_collate,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    run_dir = create_run_dir(config.run_name_prefix)
    model.train()
    losses_seen: list[float] = []
    for _ in range(config.num_epochs):
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
                "base_model_name": config.base_model_name,
                "device": device,
                "example_count": len(examples),
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "warmup_ratio": config.warmup_ratio,
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
        base_model_name=config.base_model_name,
        device=device,
    )


def resolve_train_device(configured_device: str) -> str:
    if configured_device == "auto":
        return detect_device()
    if configured_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Train config requested mps, but MPS is not available on this machine.")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Train config requested cuda, but CUDA is not available on this machine.")
    return configured_device
