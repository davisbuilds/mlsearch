# MLSearch

Local-first semantic retrieval project for arXiv `cs.LG` papers. MLSearch treats
paper search as a benchmarked ML system: fixed corpus, reviewed evaluation
splits, local embedding/reranking, and mechanical comparison through a results
ledger.

## Agent Setup

New here? Paste the prompt below into your coding agent (Claude Code, Codex, etc.) and it will install and verify the CLI, then walk you to the first smoke run.

```text
Set up the `mlsearch` repo for me. It's a local-first, benchmark-first retrieval
project for arXiv cs.LG papers (embeddings + optional reranking, judged on reviewed
held-out eval splits). Python + uv + PyTorch + sentence-transformers; optimized for
Apple Silicon. No secrets or API keys — everything runs locally.

Do this, in order:
1. Install deps. Ensure `uv` is installed (https://astral.sh/uv); run
   `uv sync --group dev` from the repo root. Clone
   https://github.com/davisbuilds/mlsearch.git and cd in first if needed.
2. Verify the CLI is wired up WITHOUT downloading anything heavy: run
   `uv run mlsearch --help` and `uv run ruff check .`. Both should succeed. If
   either fails, show me the error and stop. (The full pytest suite pulls in
   torch/sentence-transformers and is heavy — only run `uv run python -m pytest -q`
   if I ask.)
3. Report back: confirm the CLI runs and lint passed, then give me the first real
   smoke path from the Quickstart, which downloads a tiny corpus from arXiv:
   `uv run mlsearch corpus build --limit 10` → `uv run mlsearch corpus validate`.

Don't commit anything.
```

Prefer to do it yourself? The manual steps are below.

## What It Does

- Builds a reproducible arXiv `cs.LG` corpus for April 1, 2016 through March 31, 2026.
- Searches paper titles and abstracts with local embeddings.
- Generates mixed-style synthetic queries for manual review.
- Maintains reviewed `dev` and blind `test` evaluation splits.
- Excludes held-out reviewed source papers from training.
- Supports optional second-stage reranking over retriever shortlists.
- Records model comparisons through a results ledger.
- Keeps the workflow CLI-first and local-first.

## Quick Start

Requirements:

- Python `3.11+`
- `uv`
- Apple Silicon is the primary local target

```bash
uv sync --group dev
uv run mlsearch --help
uv run ruff check .
```

Tiny corpus smoke path:

```bash
uv run mlsearch corpus build --limit 10
uv run mlsearch corpus validate
```

For a real run, increase the corpus and review counts after the smoke path works
locally.

## Common Commands

```bash
uv run mlsearch benchmark generate
uv run mlsearch benchmark diagnostics
uv run mlsearch benchmark sample-review --count 4
uv run mlsearch benchmark review-stats
uv run mlsearch benchmark review-loop --limit 1
uv run mlsearch benchmark finalize-review
uv run mlsearch index build
uv run mlsearch eval baseline
uv run mlsearch eval baseline-rerank
uv run mlsearch eval baseline-rerank --split test
uv run mlsearch search "few-shot classification" --top-k 3
uv run mlsearch search "few-shot classification" --top-k 3 --rerank
uv run mlsearch train --config configs/train.yaml
uv run mlsearch eval compare --model latest --record-results
uv run mlsearch experiment sweep --reference-model latest --learning-rate 1e-5 2e-5 --num-epochs 1 2 --record-results
uv run mlsearch experiment rerank --retriever-model latest --reference-model latest --record-results

uv run ruff check .
uv run ruff format --check .
```

Top-level command groups are `corpus`, `benchmark`, `index`, `eval`,
`experiment`, `train`, and `search`.

## Review Workflow

1. Optionally archive current reviewed artifacts with `benchmark archive-reviewed --label <name>`.
2. Generate candidates with `benchmark generate`.
3. Inspect overlap with `benchmark diagnostics`.
4. Export a sample with `benchmark sample-review`; use `--split dev` for tuning and `--split test` for blind validation.
5. Track progress with `benchmark review-stats`.
6. Review rows with `benchmark review-loop` or `benchmark review-next`.
7. Mark each query as `accept`, `edit`, or `reject`.
8. Finalize with `benchmark finalize-review`.

Reviewed files are the eval source for `eval baseline`, `eval baseline-rerank`,
and `eval compare`.

## Code Layout

```text
configs/       training and experiment configs
data/          generated corpus, benchmark, index, and result artifacts
docs/          system, project, and plan docs
src/           mlsearch package and CLI
tests/         pytest suite and fixtures
```

## Documentation

- Agent guidance: [AGENTS.md](AGENTS.md)
- Architecture: [docs/system/ARCHITECTURE.md](docs/system/ARCHITECTURE.md)
- Benchmark workflow: [docs/system/BENCHMARK.md](docs/system/BENCHMARK.md)
- Training and experiments: [docs/system/TRAINING.md](docs/system/TRAINING.md)
- Operations: [docs/system/OPERATIONS.md](docs/system/OPERATIONS.md)
- Plans: [docs/plans/](docs/plans/)

## Current Boundaries

- Fully local and cheap by design.
- Apple Silicon is the primary target.
- v1 does not ingest full-text HTML/PDF.
- CLI-first instead of web-first.
- Full pytest can pull in heavy ML dependencies; use the documented smoke checks unless broader validation is needed.

## License

MIT. See [LICENSE](LICENSE).
