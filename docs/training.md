# Training Workflow

## Current Path

The v1 fine-tuning loop uses:

- base model: `BAAI/bge-small-en-v1.5`
- training pairs from generated query-to-positive-document matches
- in-batch negatives through `MultipleNegativesRankingLoss`
- checkpoint output under `artifacts/models/`

## Device Choice

Inference currently uses the default device detection path, which prefers MPS on Apple Silicon when available.

Training currently forces CPU because the sentence-transformers training path hit an MPS allocation error in live verification. This keeps the workflow stable on the Mac mini at the cost of slower training.

## Commands

```bash
uv run python -m arxiv_cslg_search.cli train --config configs/train.yaml
uv run python -m arxiv_cslg_search.cli eval compare --model latest --record-results
```

## Current Limitation

The compare step currently evaluates against the generated benchmark rather than a reviewed held-out slice, so it is useful for plumbing verification but not yet a trustworthy research metric.
