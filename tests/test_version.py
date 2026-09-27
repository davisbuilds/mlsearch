"""Version provenance works in installed and source-only invocations."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("installed", [False, True])
def test_version_uses_distribution_metadata_or_source_project(tmp_path, installed):
    package = tmp_path / "src" / "mlsearch"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(Path("src/mlsearch/__init__.py").read_text())
    tmp_path.joinpath("pyproject.toml").write_text('[project]\nversion = "9.8.7"\n')
    if installed:
        distribution = tmp_path / "src" / "mlsearch-9.9.9.dist-info"
        distribution.mkdir()
        distribution.joinpath("METADATA").write_text("Name: mlsearch\nVersion: 9.9.9\n")
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import mlsearch; print(mlsearch.__version__)"],
        env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ("9.9.9" if installed else "9.8.7")
