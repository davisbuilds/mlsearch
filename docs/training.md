# Training Workflow

## Current Path

The v1 fine-tuning loop uses:

- base model: `BAAI/bge-small-en-v1.5`
- training pairs from generated query-to-positive-document matches
- in-batch negatives through `MultipleNegativesRankingLoss`
- checkpoint output under `artifacts/models/`

## Device Choice

Inference currently uses the default device detection path, which prefers MPS on Apple Silicon when available.

Training now uses the configured device in `configs/train.yaml`.

Supported values:

- `mps`
- `cpu`
- `cuda`
- `auto`

`auto` reuses the runtime device detection path and prefers `mps` on Apple Silicon, then `cuda`, then `cpu`.

The tracked training config now also carries a `seed`, and training applies that seed before dataloader shuffling and optimization so experiment runs are easier to compare fairly.

The tracked training recipe can also augment question-style queries across the supported review prefixes (`papers on`, `research on`, `work on`). This broadens the local training pool without mutating the reviewed eval set.

There is also an optional `hard_query_pattern_weighting` knob that biases the capped training sample toward broader, lower-overlap queries. It is useful as an experiment surface, but it is not the default recipe unless it produces a new champion.

## Commands

```bash
uv run mlsearch train --config configs/train.yaml
uv run mlsearch eval compare --model latest --record-results
uv run mlsearch experiment sweep --reference-model latest --learning-rate 1e-5 2e-5 --num-epochs 1 2 --record-results
```

Baseline and compare reports in `artifacts/results/` now include per-query breakdowns.
Compare reports also include `query_deltas`, which make it easy to see which reviewed queries improved, stayed flat, or regressed versus baseline.

## Split Discipline

- Training examples come from generated query candidates.
- Any query promoted into `data/benchmark/reviewed/held_out_eval.jsonl` is excluded from training.
- `eval compare` requires a baseline report built against the same benchmark split and will refuse to compare against a stale baseline.

## Sweep Loop

`experiment sweep` is the first constrained autoresearch loop in the repo.

- It reads a base config from `configs/train.yaml`.
- It can start from the zero-shot baseline or from an existing checkpoint with `--reference-model`.
- It expands a small Cartesian grid over safe training knobs: `learning_rate`, `num_epochs`, `batch_size`, `max_examples`, and `seed`.
- It trains each variant locally, evaluates it on the reviewed held-out split, and compares it against the current champion metrics.
- It can append every run to `results.tsv` with `--record-results`.

The intended use is small and disciplined. Prefer 3-8 runs over a reviewed benchmark that already has real headroom, rather than wide sweeps against a trivial split.
