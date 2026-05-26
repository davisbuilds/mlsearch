# Training And Experiment Workflow

## Current Path

The v1 fine-tuning loop uses:

- Base model: `BAAI/bge-small-en-v1.5`
- Training pairs from generated query-to-positive-document matches
- In-batch negatives through `MultipleNegativesRankingLoss`
- Checkpoint output under `artifacts/models/`

Second-stage reranking is an experiment over the retriever shortlist, not a retriever
replacement.

## Device Choice

Inference uses the runtime device detection path. Training uses the configured
device in `configs/train.yaml`.

Supported values:

- `mps`
- `cpu`
- `cuda`
- `auto`

`auto` prefers `mps` on Apple Silicon, then `cuda`, then `cpu`.

## Training Config

`TrainConfig` lives in `src/mlsearch/config.py` and rejects unknown YAML keys.
Important fields include:

- `base_model_name`
- `device`
- `seed`
- `num_epochs`
- `batch_size`
- `learning_rate`
- `max_examples`
- `question_prefix_augmentation`
- `hard_query_pattern_weighting`

The seed is applied before dataloader shuffling and optimization so experiment runs
are easier to compare.

## Commands

```bash
uv run mlsearch train --config configs/train.yaml
uv run mlsearch eval compare --model latest --record-results
uv run mlsearch eval baseline-rerank
uv run mlsearch experiment sweep --reference-model latest --learning-rate 1e-5 2e-5 --num-epochs 1 2 --record-results
uv run mlsearch experiment rerank --retriever-model latest --reference-model latest --record-results
```

Baseline and compare reports in `artifacts/results/` include per-query breakdowns.
Compare reports include `query_deltas` so improved, unchanged, and regressed reviewed
queries can be inspected directly.

## Split Discipline

- Training examples come from generated query candidates.
- Any query promoted into `data/benchmark/reviewed/held_out_eval.jsonl` or
  `data/benchmark/reviewed/held_out_eval_test.jsonl` is excluded from training.
- Any generated query for a held-out eval `source_paper_id` is excluded, so training
  is paper-disjoint from all reviewed splits.
- Use `dev` for iterative tuning and `test` for blind checks.
- `eval compare` requires a baseline report built against the same benchmark split
  and refuses stale baseline comparisons.

## Sweep Loop

`experiment sweep` is the constrained autoresearch loop in this repo.

- It reads a base config from `configs/train.yaml`.
- It can start from the zero-shot baseline or an existing checkpoint.
- It expands a small Cartesian grid over safe training knobs: `learning_rate`,
  `num_epochs`, `batch_size`, `max_examples`, and `seed`.
- It trains each variant locally, evaluates against the reviewed split, and compares
  against the current champion metrics.
- It can append every run to `results.tsv` with `--record-results`.

Keep sweeps small and interpretable. Prefer a handful of runs over a reviewed
benchmark with real headroom, not wide searches against a trivial split.

## Reranking

Use reranking after verifying first-stage retriever recall is strong enough.

Useful commands:

```bash
uv run mlsearch eval baseline-rerank
uv run mlsearch eval baseline-rerank --split test
uv run mlsearch search "your query" --rerank
```

When model, depth, or champion claims change, update docs with the eval command and
absolute metrics that justify the change. Avoid leaving stale "best known" claims
without evidence.

