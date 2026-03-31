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
    artifacts: Path
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
            artifacts=root / "artifacts",
            plans=root / "docs" / "plans",
        )


PATHS = ProjectPaths.discover()
