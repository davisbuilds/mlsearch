# Benchmark Workflow

## Inputs

- normalized corpus at `data/processed/corpus.jsonl`
- benchmark config at `configs/benchmark.yaml`

## Generated Artifacts

- `data/benchmark/generated/query_candidates.jsonl`
- `data/benchmark/generated/benchmark_manifest.json`
- `data/benchmark/reviewed/review_sample.csv`
- `data/benchmark/reviewed/held_out_eval.jsonl`
- `data/benchmark/reviewed/held_out_eval_manifest.json`

## Query Types

The v1 benchmark mixes:

- terse keyword-style search queries
- natural-language researcher questions

Each query candidate carries:

- a source paper id
- one positive paper id
- lexical hard negatives from overlapping title tokens

## Review Workflow

1. Run `benchmark sample-review` to export a stratified CSV sample.
2. Run `benchmark review-stats` to see pending and completed counts by status and style.
3. Run `benchmark review-next` to inspect the next pending row with source-paper context.
4. Edit the CSV and set each row to `accept`, `edit`, or `reject`.
5. For `edit`, fill in both `edited_query` and `relevant_paper_ids`.
6. Run `benchmark finalize-review` to materialize the canonical held-out eval split.

The finalized JSONL is the only reviewed artifact used by `eval baseline` and `eval compare`.

## Split Rules

- `data/benchmark/generated/query_candidates.jsonl` remains the synthetic pool.
- `data/benchmark/reviewed/held_out_eval.jsonl` is the reviewed eval split.
- Training excludes any `query_id` that appears in the reviewed eval file.
