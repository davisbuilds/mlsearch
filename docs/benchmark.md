# Benchmark Workflow

## Inputs

- normalized corpus at `data/processed/corpus.jsonl`
- benchmark config at `configs/benchmark.yaml`

## Generated Artifacts

- `data/benchmark/generated/query_candidates.jsonl`
- `data/benchmark/generated/benchmark_manifest.json`
- `data/benchmark/reviewed/review_sample.csv` for `dev`
- `data/benchmark/reviewed/review_sample_test.csv` for `test`
- `data/benchmark/reviewed/held_out_eval.jsonl` for `dev`
- `data/benchmark/reviewed/held_out_eval_test.jsonl` for `test`
- matching manifest JSON files for each reviewed split

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
   By default this excludes query ids and source paper ids already seen in archived review batches and the current held-out eval.
   Use `--include-reviewed` only when you explicitly want to revisit previously reviewed rows.
   Use `--split dev` for the tuning split and `--split test` for the blind split.
5. Run `benchmark review-stats` to see pending and completed counts by status and style.
6. Run `benchmark review-next` to inspect the next pending row with source-paper context.
7. Optionally use `benchmark review-loop` to step through rows and write `accept`, `edit`, or `reject` decisions back to the CSV interactively.
8. For `edit`, fill in both `edited_query` and `relevant_paper_ids`.
9. Run `benchmark finalize-review` to materialize the canonical held-out eval split.
   Finalization merges the current reviewed CSV into the existing reviewed split and deduplicates by `query_id`.

The finalized JSONLs are the reviewed artifacts used by evaluation:

- `dev` is the default split for `eval baseline`, `eval baseline-rerank`, `eval compare`, and local experiment loops
- `test` should be reserved for blind checks such as `eval baseline --split test`

## Split Rules

- `data/benchmark/generated/query_candidates.jsonl` remains the synthetic pool.
- `data/benchmark/reviewed/held_out_eval.jsonl` is the reviewed `dev` split.
- `data/benchmark/reviewed/held_out_eval_test.jsonl` is the reviewed `test` split.
- Training excludes any reviewed `query_id` and any generated query whose `source_paper_id` appears in any reviewed split.
