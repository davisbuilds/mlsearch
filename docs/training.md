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

## Commands

```bash
uv run mlsearch train --config configs/train.yaml
uv run mlsearch eval compare --model latest --record-results
```

## Split Discipline

- Training examples come from generated query candidates.
- Any query promoted into `data/benchmark/reviewed/held_out_eval.jsonl` is excluded from training.
- `eval compare` requires a baseline report built against the same benchmark split and will refuse to compare against a stale baseline.
