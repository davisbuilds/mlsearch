# Benchmark Workflow

## Inputs

- Normalized corpus: `data/processed/corpus.jsonl`
- Benchmark config: `configs/benchmark.yaml`
- CLI group: `uv run mlsearch benchmark ...`

## Generated Artifacts

- `data/benchmark/generated/query_candidates.jsonl`
- `data/benchmark/generated/benchmark_manifest.json`
- `data/benchmark/reviewed/review_sample.csv` for `dev`
- `data/benchmark/reviewed/review_sample_test.csv` for `test`
- `data/benchmark/reviewed/held_out_eval.jsonl` for `dev`
- `data/benchmark/reviewed/held_out_eval_test.jsonl` for `test`
- Matching manifest JSON files for each reviewed split

## Query Types

The benchmark mixes:

- Terse keyword-style search queries
- Natural-language researcher questions

Each generated candidate carries a source paper id, one positive paper id, and
lexical hard negatives from overlapping title tokens. The generated manifest also
records overlap diagnostics so title-shaped candidate drift can be inspected before
review.

## Review Workflow

1. Optionally archive current reviewed artifacts:
   `uv run mlsearch benchmark archive-reviewed --label <name>`.
2. Generate candidates: `uv run mlsearch benchmark generate`.
3. Inspect title-overlap diagnostics: `uv run mlsearch benchmark diagnostics`.
4. Export a stratified sample: `uv run mlsearch benchmark sample-review`.
   By default, this excludes query ids and source paper ids already seen in archived
   review batches and the current held-out eval.
5. Use `--split dev` for the tuning split and `--split test` for blind expansion.
6. Check progress with `uv run mlsearch benchmark review-stats`.
7. Inspect pending rows with `uv run mlsearch benchmark review-next`.
8. Optionally edit rows interactively with `uv run mlsearch benchmark review-loop`.
9. For `edit`, fill in both `edited_query` and `relevant_paper_ids`.
10. Finalize the reviewed split with `uv run mlsearch benchmark finalize-review`.

Finalization merges the current reviewed CSV into the existing split and deduplicates
by `query_id`. It should not replace a reviewed split wholesale.

## Split Rules

- `dev` is the default split for iterative tuning.
- `test` is the blind reviewed split and should stay out of day-to-day tuning.
- Reviewed queries must never leak into training.
- Training excludes any generated query whose `source_paper_id` appears in any
  reviewed split.
- Use `benchmark sample-review --include-reviewed` only when intentionally revisiting
  old rows.

## Review Heuristics

Prefer human-plausible search intent over title restatement:

- Shorten title-shaped phrases.
- Convert clipped title shards into application or task phrasing.
- Keep one strong domain anchor.
- Remove generic hype words unless a researcher would likely type them.
- Reject only when the query is genuinely misleading or implausible as a search.

## Evaluation Consumers

The finalized JSONL files are the reviewed artifacts used by:

- `uv run mlsearch eval baseline`
- `uv run mlsearch eval baseline-rerank`
- `uv run mlsearch eval compare --model <checkpoint>`
- Constrained experiment loops under `uv run mlsearch experiment ...`

