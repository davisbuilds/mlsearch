from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CorpusConfig:
    category: str = "cs.LG"
    start_date: str = "2016-04-01"
    end_date: str = "2026-03-31"
    target_size: int = 5000
    page_size: int = 100
    delay_seconds: float = 3.0
    selection_strategy: str = "year_window_quota"


@dataclass(frozen=True)
class BenchmarkConfig:
    review_count_default: int = 30
    keyword_queries_per_paper: int = 1
    question_queries_per_paper: int = 1
    negatives_per_query: int = 3
    max_candidates: int = 2000
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    base_model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"
    seed: int = 42
    num_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_examples: int = 5000
    question_prefix_augmentation: bool = False
    run_name_prefix: str = "retriever"


@dataclass(frozen=True)
class RuntimeConfig:
    corpus: CorpusConfig = CorpusConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    train: TrainConfig = TrainConfig()


RUNTIME_CONFIG = RuntimeConfig()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_corpus_config(path: Path) -> CorpusConfig:
    raw = load_yaml(path)
    defaults = asdict(CorpusConfig())
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown corpus config keys: {names}")
    merged = {**defaults, **raw}
    merged["start_date"] = _normalize_date_value(merged["start_date"])
    merged["end_date"] = _normalize_date_value(merged["end_date"])
    return CorpusConfig(**merged)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    raw = load_yaml(path)
    defaults = asdict(BenchmarkConfig())
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown benchmark config keys: {names}")
    merged = {**defaults, **raw}
    return BenchmarkConfig(**merged)


def load_train_config(path: Path) -> TrainConfig:
    raw = load_yaml(path)
    defaults = asdict(TrainConfig())
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown train config keys: {names}")
    merged = {**defaults, **raw}
    if merged["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("Train config device must be one of: auto, cpu, mps, cuda")
    return TrainConfig(**merged)


def merge_train_config(config: TrainConfig, **overrides: Any) -> TrainConfig:
    unknown = sorted(set(overrides) - set(asdict(config)))
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown train config overrides: {names}")
    return replace(config, **overrides)


def _normalize_date_value(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
