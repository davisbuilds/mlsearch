from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mlsearch.config import load_train_config
from mlsearch.training.train_retriever import resolve_train_device


def test_load_train_config_accepts_device_override(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        textwrap.dedent(
            """
            base_model_name: BAAI/bge-small-en-v1.5
            device: mps
            num_epochs: 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_train_config(path)

    assert config.device == "mps"


def test_load_train_config_accepts_seed_override(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("seed: 7\n", encoding="utf-8")

    config = load_train_config(path)

    assert config.seed == 7


def test_load_train_config_rejects_unknown_device(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("device: tpu\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Train config device must be one of"):
        load_train_config(path)


def test_resolve_train_device_returns_explicit_cpu() -> None:
    assert resolve_train_device("cpu") == "cpu"
