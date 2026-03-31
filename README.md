# arXiv cs.LG Search

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
2. Generate benchmark queries and review a held-out slice.
3. Build a retrieval index and measure a zero-shot baseline.
4. Fine-tune locally and compare against baseline.
5. Search the local index from the CLI.

## CLI

The top-level command surface is:

```bash
uv run python -m arxiv_cslg_search.cli --help
```

Planned subcommands:

- `corpus`
- `benchmark`
- `index`
- `eval`
- `train`
- `search`

Current working slice:

```bash
uv sync
uv run python -m arxiv_cslg_search.cli corpus build --limit 10
uv run python -m arxiv_cslg_search.cli corpus validate
uv run python -m arxiv_cslg_search.cli benchmark generate
uv run python -m arxiv_cslg_search.cli benchmark sample-review --count 4
uv run python -m arxiv_cslg_search.cli index build
uv run python -m arxiv_cslg_search.cli eval baseline
uv run python -m arxiv_cslg_search.cli search "few-shot classification" --top-k 3
uv run python -m arxiv_cslg_search.cli train --config configs/train.yaml
uv run python -m arxiv_cslg_search.cli eval compare --model latest --record-results
```

## Constraints

- Fully local and cheap
- Apple Silicon Mac mini M4 as the primary machine
- No full-text HTML/PDF ingestion in v1
- CLI-first instead of web-first
