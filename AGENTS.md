# AGENTS.md

Guidance for agents working in `mlsearch`.

## Project Intent

MLSearch is a benchmark-first local retrieval project for arXiv `cs.LG` papers.

The important invariant is:
- keep corpus and reviewed eval logic fixed
- mutate only narrow retrieval, training, and experiment surfaces
- judge changes mechanically on the reviewed held-out eval

## Documentation Map

- `docs/architecture.md` — project intent, initial layout, design principles.
- `docs/benchmark.md` — benchmark inputs, generated artifacts, query types, review workflow, split rules.
- `docs/training.md` — training path, device choice, commands, split discipline, sweep loop, rerank experiment.

Run `uv run mlsearch --help` to list all CLI commands (`corpus`, `benchmark`, `index`, `eval`, `train`, `experiment`, `search`).

## Core Workflow

1. Build or validate the local corpus.
2. Generate synthetic query candidates.
3. Review a small batch into the held-out eval.
4. Run baseline retrieval.
5. Run retriever training and compare against the reviewed eval.
6. Optionally run second-stage reranking experiments.

Prefer preserving this order. Do not change benchmark generation and model recipes in the same step unless the user explicitly asks for that coupling.

## Benchmark Rules

- `data/benchmark/reviewed/held_out_eval.jsonl` is the `dev` reviewed eval split.
- `data/benchmark/reviewed/held_out_eval_test.jsonl` is the `test` reviewed eval split.
- Reviewed queries must never leak into training.
- `benchmark sample-review` now excludes previously reviewed query ids and source paper ids by default.
- Use `--split dev` for the working benchmark and `--split test` for blind expansion.
- `benchmark finalize-review` merges the current reviewed batch into the existing split instead of replacing it.
- If you need to revisit old rows intentionally, use `benchmark sample-review --include-reviewed`.
- Treat all reviewed splits as paper-disjoint from training: if a paper appears in any held-out split, generated queries for that paper should not be used for training.

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
- The strongest current end-to-end system is validated on the paper-disjoint blind `test` split, not just `dev`.
- As of the expanded `50`-query blind `test` split, the strongest known setup is:
  - zero-shot baseline retriever over `BAAI/bge-small-en-v1.5`
  - reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `rerank_depth=10`
- `rerank_depth=5` is not enough on the current blind split, and `cross-encoder/ms-marco-TinyBERT-L-2-v2` is close but still measurably worse than the default reranker.
- After enforcing paper-disjoint held-out training, treat `eval baseline-rerank` and `search --rerank` as the most trustworthy default path unless a newer paper-disjoint fine-tuned retriever clearly beats them.

## Verification

Use:

```bash
uv run ruff check .
uv run python -m pytest -q
```

Prefer targeted test slices while iterating, then run the full suite before claiming completion.

For benchmark-affecting changes, rerun the relevant eval commands and report absolute metrics, not just status labels.

- **Ruff** enforces style + small fixups (`E, F, I, B, UP, W, C4, SIM, ERA, RUF, PIE`); `ruff format` keeps formatting consistent. Fix violations the linter flags rather than restating rules here.
- **Dead-code gate** (`tests/test_dead_code.py`): static checks for unused public symbols, orphaned modules, and unreachable code. It owns cross-file dead code; ruff `F`/`ERA` own within-file unused imports/locals and commented-out code. When a symbol/module is intentionally unreferenced (external API, framework-invoked), add it to `SYMBOL_EXCEPTIONS`/`MODULE_EXCEPTIONS` with a reason rather than silencing the test.

## Working Agreement

- **Push back before building.** If a request is incoherent or self-contradictory, or a spec/plan is vague or skips key decisions, stop and interview me — ask clarifying questions and confirm intent before writing code or changing files. Don't guess at scope or comply silently. (Clear, well-scoped requests don't need this.)
- **Keep docs current.** After a significant change, PR, or completed spec/plan, update any now-stale reference docs under `docs/` (`architecture.md`, `benchmark.md`, `training.md`) so they match shipped behavior. Skip this for trivial changes.
