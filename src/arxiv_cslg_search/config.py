from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusConfig:
    category: str = "cs.LG"
    start_date: str = "2016-04-01"
    end_date: str = "2026-03-31"
    target_size: int = 5000


@dataclass(frozen=True)
class BenchmarkConfig:
    review_count_default: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    corpus: CorpusConfig = CorpusConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()


RUNTIME_CONFIG = RuntimeConfig()
