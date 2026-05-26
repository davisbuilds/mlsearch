# Operations

## Local Development

```bash
uv sync --group dev
uv run mlsearch --help
uv run ruff check .
```

The full project dependencies include Torch and sentence-transformers. Initial
installs and full test runs are heavier than the CI gate.

## Useful Commands

```bash
uv run mlsearch corpus build --limit 10
uv run mlsearch corpus validate
uv run mlsearch benchmark generate
uv run mlsearch benchmark diagnostics
uv run mlsearch benchmark sample-review --count 4
uv run mlsearch benchmark review-stats
uv run mlsearch index build
uv run mlsearch eval baseline
uv run mlsearch eval baseline-rerank
uv run mlsearch search "few-shot classification" --top-k 3
uv run mlsearch search "few-shot classification" --top-k 3 --rerank
uv run mlsearch train --config configs/train.yaml
```

Use tiny corpus/review counts for smoke checks. Increase counts only for real
benchmark or model work.

## CI

Workflow: `.github/workflows/ci.yml`

CI is intentionally lean:

- `uvx ruff@0.15.12 check .`
- `uvx ruff@0.15.12 format --check .`
- `uv run --no-project --with pytest --python 3.12 python -m pytest -q tests/test_dead_code.py`

CI does not install Torch or sentence-transformers and does not run the full pytest
suite. Run the full local suite before claiming behavioral changes are complete.

## Local Verification

Use this before pushing routine changes:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest -q
```

For benchmark-affecting changes, also rerun the relevant eval command and report
absolute metrics. A passing unit suite is not enough evidence for model or benchmark
quality changes.

## Artifact Locations

- Corpus: `data/raw/`, `data/processed/`
- Benchmark generated data: `data/benchmark/generated/`
- Reviewed eval data: `data/benchmark/reviewed/`
- Indexes: `artifacts/index/`
- Model checkpoints: `artifacts/models/`
- Eval reports: `artifacts/results/`
- Plans: `docs/plans/`

Large generated artifacts are local working state. Do not add heavyweight corpora,
indexes, or checkpoints to docs or source review unless explicitly requested.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `mlsearch --help` import error | Run `uv sync --group dev`; confirm command is run from the repo or via the installed environment. |
| Full tests are slow | Expected. The suite imports ML dependencies; use targeted tests while iterating, then full pytest before handoff. |
| CI passes but local pytest fails | Expected risk. CI deliberately skips heavy dependencies and most tests. Fix local failure before claiming completion. |
| Eval compare refuses to run | Rebuild a baseline for the same split before comparing. |
| Search or eval cannot find an index | Run `uv run mlsearch index build` after building/validating the corpus. |
| Training looks suspiciously good | Verify reviewed queries and all reviewed source papers are excluded from training. |

