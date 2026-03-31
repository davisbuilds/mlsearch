from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src: Path
    configs: Path
    docs: Path
    data: Path
    data_raw: Path
    data_processed: Path
    data_benchmark: Path
    artifacts: Path
    artifacts_index: Path
    artifacts_models: Path
    artifacts_results: Path
    plans: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        return cls(
            root=root,
            src=root / "src",
            configs=root / "configs",
            docs=root / "docs",
            data=root / "data",
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            data_benchmark=root / "data" / "benchmark",
            artifacts=root / "artifacts",
            artifacts_index=root / "artifacts" / "index",
            artifacts_models=root / "artifacts" / "models",
            artifacts_results=root / "artifacts" / "results",
            plans=root / "docs" / "plans",
        )


PATHS = ProjectPaths.discover()
