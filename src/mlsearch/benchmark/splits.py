from __future__ import annotations

from pathlib import Path

from mlsearch.paths import PATHS

DEFAULT_REVIEW_SPLIT = "dev"
REVIEW_SPLITS = ("dev", "test")


def normalize_review_split(split: str | None) -> str:
    resolved = (split or DEFAULT_REVIEW_SPLIT).strip().lower()
    if resolved not in REVIEW_SPLITS:
        allowed = ", ".join(REVIEW_SPLITS)
        raise ValueError(f"Unsupported review split {split!r}; expected one of {allowed}.")
    return resolved


def review_sample_path(*, split: str = DEFAULT_REVIEW_SPLIT, reviewed_dir: Path | None = None) -> Path:
    resolved_dir = reviewed_dir or (PATHS.data_benchmark / "reviewed")
    resolved_split = normalize_review_split(split)
    name = "review_sample.csv" if resolved_split == "dev" else f"review_sample_{resolved_split}.csv"
    return resolved_dir / name


def held_out_eval_path(*, split: str = DEFAULT_REVIEW_SPLIT, reviewed_dir: Path | None = None) -> Path:
    resolved_dir = reviewed_dir or (PATHS.data_benchmark / "reviewed")
    resolved_split = normalize_review_split(split)
    name = "held_out_eval.jsonl" if resolved_split == "dev" else f"held_out_eval_{resolved_split}.jsonl"
    return resolved_dir / name


def held_out_eval_manifest_path(*, split: str = DEFAULT_REVIEW_SPLIT, reviewed_dir: Path | None = None) -> Path:
    resolved_dir = reviewed_dir or (PATHS.data_benchmark / "reviewed")
    resolved_split = normalize_review_split(split)
    name = "held_out_eval_manifest.json" if resolved_split == "dev" else f"held_out_eval_manifest_{resolved_split}.json"
    return resolved_dir / name


def all_review_sample_paths(*, reviewed_dir: Path | None = None) -> list[Path]:
    resolved_dir = reviewed_dir or (PATHS.data_benchmark / "reviewed")
    return sorted(path for path in resolved_dir.glob("review_sample*.csv") if path.is_file())


def all_held_out_eval_paths(*, reviewed_dir: Path | None = None) -> list[Path]:
    resolved_dir = reviewed_dir or (PATHS.data_benchmark / "reviewed")
    return sorted(path for path in resolved_dir.glob("held_out_eval*.jsonl") if path.is_file())
