# Architecture

## Intent

MLSearch is organized around a benchmark-first retrieval workflow for arXiv `cs.LG`
papers:

- Fixed corpus build path
- Fixed reviewed evaluation harness
- Narrow training and experiment surfaces
- Optional second-stage reranking over a fixed retriever shortlist
- Shared CLI for corpus, benchmark, evaluation, training, experiment, and search work

The important boundary is that corpus and reviewed eval logic should stay stable
while retrieval, training, and experiment surfaces evolve.

## Runtime Shape

- Language/runtime: Python 3.11+
- Package manager: `uv`
- CLI entrypoint: `mlsearch = "mlsearch.cli:main"`
- Core ML dependencies: `sentence-transformers`, `torch`, `numpy`
- Tests: `pytest`
- Style: `ruff`

The full dependency set is intentionally heavier than CI. Local development installs
the project with `uv sync --group dev`; CI uses a lean gate so it does not install
Torch or sentence-transformers.

## CLI Surface

`src/mlsearch/cli.py` exposes these top-level command groups:

```text
corpus, benchmark, index, eval, experiment, train, search
```

The CLI routes to small pipeline modules rather than embedding workflow logic in
the argument parser. Keep that shape when adding commands:

- Corpus build/validation: `src/mlsearch/pipelines/build_corpus.py`,
  `src/mlsearch/pipelines/validate_corpus.py`
- Benchmark generation/review: `src/mlsearch/pipelines/generate_queries.py`,
  `src/mlsearch/pipelines/sample_review_set.py`,
  `src/mlsearch/pipelines/finalize_review_set.py`,
  `src/mlsearch/pipelines/review_workflow.py`
- Evaluation: `src/mlsearch/eval/run_eval.py`
- Training: `src/mlsearch/training/train_retriever.py`
- Experiments: `src/mlsearch/experiments/`
- Interactive output: `src/mlsearch/present/search_output.py`

## Path Model

Canonical paths live in `src/mlsearch/paths.py`.

```text
configs/       User-editable YAML config
data/          Local corpus and benchmark artifacts
artifacts/     Indexes, checkpoints, reports, and result files
docs/system/   Durable architecture, workflow, and operations docs
docs/plans/    Brainstorms and implementation plans
```

Do not spread new hard-coded repository paths through pipeline modules. Extend
`ProjectPaths` first when a new durable location is needed.

## Configuration

Config dataclasses live in `src/mlsearch/config.py`:

- `CorpusConfig`: arXiv category, date window, target size, fetch cadence, selection
  strategy
- `BenchmarkConfig`: review counts, generated query mix, hard negatives, seed
- `TrainConfig`: model, device, seed, epochs, batch size, learning rate, sample caps,
  experiment knobs

YAML loaders reject unknown keys. Keep that strictness so old config typos fail
early instead of silently changing experiment assumptions.

## Data And Artifact Boundaries

- `data/processed/corpus.jsonl` is the normalized corpus input to retrieval and
  benchmark generation.
- `data/benchmark/generated/` holds synthetic candidates and diagnostics.
- `data/benchmark/reviewed/` holds human-reviewed `dev` and `test` eval splits.
- `artifacts/index/` holds local vector indexes and embeddings.
- `artifacts/models/` holds trained retriever checkpoints.
- `artifacts/results/` holds eval and experiment reports.

Generated data and model artifacts are local working state. Treat source, configs,
tests, and docs as the reviewable repo surface.

## Design Principles

- Keep data and evaluation logic deterministic.
- Preserve paper-disjoint reviewed splits when training.
- Prefer simple local formats over service infrastructure.
- Avoid backend-specific optimization until the benchmark is stable.
- Mutate one surface at a time: benchmark, retriever, reranker, or training recipe.
- Report absolute metrics for benchmark-affecting changes, not just pass/fail status.

