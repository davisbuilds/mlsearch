# Architecture

## Intent

The repo is organized around a benchmark-first retrieval workflow:

- fixed data build path
- fixed evaluation harness
- narrow training and experiment surfaces
- shared CLI for both benchmarking and search

## Initial Layout

- `src/mlsearch/cli.py`: CLI entrypoint
- `src/mlsearch/config.py`: project-level runtime settings
- `src/mlsearch/paths.py`: canonical filesystem paths
- `configs/`: user-editable configuration files
- `data/`: local dataset and benchmark artifacts
- `artifacts/`: model outputs, reports, and indexes
- `docs/plans/`: brainstorm and implementation plans

## Design Principles

- Keep data and evaluation logic deterministic.
- Prefer simple local formats over heavy infrastructure.
- Avoid backend-specific optimization until the benchmark is stable.
- Shape the repo so future autoresearch-style loops can mutate a small surface without touching corpus and eval rules.
