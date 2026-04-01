# MLSearch

Local-first semantic paper search over arXiv `cs.LG`, with a benchmark-first workflow and a CLI-first search tool.

## Status

This repo is under active construction. The first milestone is:

- build a reproducible `cs.LG` corpus for April 1, 2016 through March 31, 2026
- generate a mixed-style semantic-search benchmark
- establish a zero-shot retrieval baseline
- beat that baseline with local fine-tuning
- expose the same retrieval pipeline through a local CLI

## Workflow

The intended happy path is:

1. Build and validate the corpus.
2. Generate benchmark queries, review a held-out slice, and finalize the reviewed eval split.
3. Build a retrieval index and measure a zero-shot baseline.
4. Fine-tune locally and compare against baseline.
5. Search the local index from the CLI.

## CLI

The top-level command surface is:

```bash
uv run mlsearch --help
```

Planned subcommands:

- `corpus`
- `benchmark`
- `index`
- `eval`
- `train`
- `search`

Review-specific helpers:

- `benchmark review-stats`
- `benchmark review-next`
- `benchmark review-loop`

Current working slice:

```bash
uv sync
uv run mlsearch corpus build --limit 10
uv run mlsearch corpus validate
uv run mlsearch benchmark generate
uv run mlsearch benchmark sample-review --count 4
uv run mlsearch benchmark review-stats
uv run mlsearch benchmark review-next
uv run mlsearch benchmark review-loop --limit 1
# edit data/benchmark/reviewed/review_sample.csv and set review_status to accept/edit/reject
uv run mlsearch benchmark finalize-review
uv run mlsearch index build
uv run mlsearch eval baseline
uv run mlsearch search "few-shot classification" --top-k 3
uv run mlsearch train --config configs/train.yaml
uv run mlsearch eval compare --model latest --record-results
```

## Constraints

- Fully local and cheap
- Apple Silicon Mac mini M4 as the primary machine
- No full-text HTML/PDF ingestion in v1
- CLI-first instead of web-first
