"""arXiv cs.LG semantic search."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["__version__"]

try:
    __version__ = version("mlsearch")
except PackageNotFoundError:
    # Support PYTHONPATH=src and source-only test runs before installation.
    import tomllib

    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as project:
        __version__ = tomllib.load(project)["project"]["version"]
