# MLSearch

MLSearch is a local-first retrieval project for arXiv `cs.LG` papers.

The repo is built around one idea: treat paper search as a benchmarked ML system, not just a demo. The workflow fixes the corpus and evaluation harness, generates mixed-style human search queries, keeps a reviewed held-out eval split, and compares zero-shot retrieval against local fine-tuning.

## Current Scope

The current v1 target is:

- a reproducible `cs.LG` corpus covering April 1, 2016 through March 31, 2026
- paper-level retrieval over title plus abstract
- synthetic-first query generation with manual review
- a reviewed held-out evaluation split
- local embedding-based indexing and retrieval
- a CLI-first search and review workflow

This is intentionally narrow. There is no full-text PDF/HTML ingestion in v1, and the project is optimized for local iteration on Apple Silicon rather than large-scale training infrastructure.

## Why This Repo Exists

Most paper-search prototypes stop at “embed some abstracts and query them.” MLSearch is trying to be a little stricter:

- corpus build is deterministic
- evaluation is explicit and versioned
- reviewed queries are separated from training data
- comparisons are mechanical through a results ledger
- manual review is part of the benchmark, not an afterthought

## Install

MLSearch uses `uv` and pinned dependency versions.

```bash
uv sync --group dev
uv run mlsearch --help
```

## Quickstart

```bash
uv run mlsearch corpus build --limit 10
uv run mlsearch corpus validate
uv run mlsearch benchmark generate
uv run mlsearch benchmark sample-review --count 4
uv run mlsearch benchmark review-stats
uv run mlsearch benchmark review-loop --limit 1
uv run mlsearch benchmark finalize-review
uv run mlsearch index build
uv run mlsearch eval baseline
uv run mlsearch search "few-shot classification" --top-k 3
uv run mlsearch train --config configs/train.yaml
uv run mlsearch eval compare --model latest --record-results
```

For a real run, increase the corpus and review counts after the smoke path is working locally.

## Review Workflow

The benchmark review flow is a core part of the project:

1. Export a review sample with `benchmark sample-review`.
2. Inspect progress with `benchmark review-stats`.
3. Step through rows with `benchmark review-loop` or inspect one row with `benchmark review-next`.
4. Mark each query as `accept`, `edit`, or `reject`.
5. Finalize the reviewed split with `benchmark finalize-review`.

The finalized `held_out_eval.jsonl` is the eval source for `eval baseline` and `eval compare`, and those query ids are excluded from training.

## CLI Surface

Top-level commands:

- `corpus`
- `benchmark`
- `index`
- `eval`
- `train`
- `search`

Useful review helpers:

- `benchmark review-stats`
- `benchmark review-next`
- `benchmark review-loop`

## Constraints

- fully local and cheap
- Apple Silicon Mac mini M4 as the primary machine
- no full-text HTML/PDF ingestion in v1
- CLI-first instead of web-first

## License

MIT. See [LICENSE](LICENSE).
