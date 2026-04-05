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

The generated manifest also includes overlap diagnostics so you can quantify when the synthetic query pool is drifting too close to source-paper titles.

## Review Workflow

1. Optionally run `benchmark archive-reviewed --label <name>` before regenerating candidates or replacing the current review sample.
2. Run `benchmark generate` to rebuild the candidate pool.
3. Run `benchmark diagnostics` to inspect title-overlap statistics from the generated manifest.
4. Run `benchmark sample-review` to export a stratified CSV sample.
   By default this excludes query ids already seen in archived review batches and the current held-out eval.
   Use `--include-reviewed` only when you explicitly want to revisit previously reviewed rows.
5. Run `benchmark review-stats` to see pending and completed counts by status and style.
6. Run `benchmark review-next` to inspect the next pending row with source-paper context.
7. Optionally use `benchmark review-loop` to step through rows and write `accept`, `edit`, or `reject` decisions back to the CSV interactively.
8. For `edit`, fill in both `edited_query` and `relevant_paper_ids`.
9. Run `benchmark finalize-review` to materialize the canonical held-out eval split.
   Finalization merges the current reviewed CSV into the existing held-out eval and deduplicates by `query_id`.

The finalized JSONL is the only reviewed artifact used by `eval baseline` and `eval compare`.

## Split Rules

- `data/benchmark/generated/query_candidates.jsonl` remains the synthetic pool.
- `data/benchmark/reviewed/held_out_eval.jsonl` is the reviewed eval split.
- Training excludes any `query_id` that appears in the reviewed eval file.
