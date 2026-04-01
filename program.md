# mlsearch autoresearch policy

This repo is structured for future autoresearch-style local iteration.

## Fixed surfaces

- `configs/corpus.yaml`
- normalized corpus artifacts under `data/processed/`
- reviewed benchmark artifacts under `data/benchmark/reviewed/`
- evaluation metrics and report logic under `src/mlsearch/eval/`

## Editable optimization surfaces

- training configuration under `configs/train.yaml`
- retriever training code under `src/mlsearch/training/`
- retrieval formatting and scoring code under `src/mlsearch/retrieval/`

## Goal

Improve reviewed retrieval quality for human semantic-search queries over `cs.LG` papers while keeping the corpus and evaluation harness fixed.
