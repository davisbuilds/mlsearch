# AGENTS.md

Guidance for agents working in `mlsearch`.

## Project Intent

MLSearch is a benchmark-first local retrieval project for arXiv `cs.LG` papers.

The important invariant is:
- keep corpus and reviewed eval logic fixed
- mutate only narrow retrieval, training, and experiment surfaces
- judge changes mechanically on the reviewed held-out eval

## Core Workflow

1. Build or validate the local corpus.
2. Generate synthetic query candidates.
3. Review a small batch into the held-out eval.
4. Run baseline retrieval.
5. Run retriever training and compare against the reviewed eval.
6. Optionally run second-stage reranking experiments.

Prefer preserving this order. Do not change benchmark generation and model recipes in the same step unless the user explicitly asks for that coupling.

## Benchmark Rules

- `data/benchmark/reviewed/held_out_eval.jsonl` is the canonical reviewed eval split.
- Reviewed queries must never leak into training.
- `benchmark sample-review` now excludes previously reviewed query ids and source paper ids by default.
- `benchmark finalize-review` merges the current reviewed batch into the existing held-out eval instead of replacing it.
- If you need to revisit old rows intentionally, use `benchmark sample-review --include-reviewed`.
- Treat the held-out eval as paper-disjoint from training: if a paper appears in held-out eval, generated queries for that paper should not be used for training.

## Review Heuristics

The generator is no longer producing nonsense, but it still tends to emit queries that are too title-shaped or clipped.

When reviewing:
- prefer human-plausible intent over title restatement
- broaden slightly rather than copying the paper title
- keep one strong domain anchor
- remove catchy or branded phrasing like `one day`, `enhancing`, `framework`, `insight-driven` unless a user would likely search for it
- reject only when the query is genuinely misleading or not something a researcher would type

Good edits usually look like:
- title-shaped phrase -> shorter intent phrase
- clipped title shards -> application or task phrasing
- awkward question wrapper -> clean keyword search

Examples of good directions:
- `spatio temporal spot forecasting framework` -> `traffic prediction frameworks`
- `papers on subgroup performance analysis asr models` -> `ASR models performance analysis`
- `work on contextual preference collaborative measure framework` -> `preference modeling with belief systems`

## Training Notes

- The current best first-stage retriever checkpoint is `retriever-20260404T225128Z` unless newer evaluated results explicitly beat it on the current held-out eval.
- `question_prefix_augmentation` is useful as an experiment surface, but it is not automatically a new champion just because it beats the zero-shot baseline.
- `hard_query_pattern_weighting` is available as an optional experiment knob. Treat it as experimental unless it clearly beats the current incumbent.
- When comparing candidate runs, use the reviewed held-out eval and keep champion semantics explicit. Be careful not to confuse “beats baseline” with “beats incumbent.”
- The current benchmark is large enough that simple hyperparameter sweeps are informative, but benchmark changes still have more leverage than broad recipe churn.

## Reranking Notes

- Use reranking only after verifying the retriever already has strong recall.
- The current reranker path is a second-stage experiment over the retriever shortlist, not a retriever replacement.
- If `Recall@10` is already saturated and `MRR`/`nDCG` still have headroom, reranking is usually the highest-leverage next step.
- As of the expanded `45`-query reviewed eval, the strongest end-to-end system is:
  - zero-shot baseline retriever over `BAAI/bge-small-en-v1.5`
  - reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`
- After enforcing paper-disjoint held-out training, treat `eval baseline-rerank` and `search --rerank` as the most trustworthy default path unless a newer paper-disjoint fine-tuned retriever clearly beats them.

## Verification

Use:

```bash
uv run python -m pytest -q
```

Prefer targeted test slices while iterating, then run the full suite before claiming completion.

For benchmark-affecting changes, rerun the relevant eval commands and report absolute metrics, not just status labels.
