# Benchmark Workflow

## Inputs

- normalized corpus at `data/processed/corpus.jsonl`
- benchmark config at `configs/benchmark.yaml`

## Generated Artifacts

- `data/benchmark/generated/query_candidates.jsonl`
- `data/benchmark/generated/benchmark_manifest.json`
- `data/benchmark/reviewed/review_sample.csv`

## Query Types

The v1 benchmark mixes:

- terse keyword-style search queries
- natural-language researcher questions

Each query candidate carries:

- a source paper id
- one positive paper id
- lexical hard negatives from overlapping title tokens

## Current Limitation

The benchmark is still synthetic-first. The manual review slice exists, but reviewed labels have not been incorporated into a separate held-out evaluation split yet.
