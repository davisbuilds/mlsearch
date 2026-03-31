from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from arxiv_cslg_search.paths import PATHS


def create_run_dir(prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PATHS.artifacts_models / f"{prefix}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_checkpoint() -> Path:
    if not PATHS.artifacts_models.exists():
        raise FileNotFoundError("No model artifacts directory exists yet.")
    candidates = sorted(path for path in PATHS.artifacts_models.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError("No model checkpoints found.")
    return candidates[-1]
