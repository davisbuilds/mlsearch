---
date: 2026-03-31
topic: cslg-semantic-paper-search
stage: brainstorm
---

# cs.LG Semantic Paper Search

## What We Are Building
A fresh greenfield ML research prototype for local semantic search over arXiv `cs.LG` papers. The first version will index about 5,000 papers from April 1, 2016 through March 31, 2026 and retrieve papers at the title-plus-abstract level in response to human-style search queries. The project will be benchmark-first but will also ship a CLI-first local search tool over the same index and retrieval pipeline.

## Why This Direction
This keeps the project aligned with the strongest transferable idea from `karpathy/autoresearch`: one narrow editable optimization surface, a fixed evaluation harness, a mechanical metric, and iterative keep/discard improvement. Compared with full-text section retrieval, paper-level title-plus-abstract retrieval is much more tractable on an Apple Silicon Mac and avoids unnecessary ingestion complexity in v1. Compared with citation-signal training, human semantic search is more interesting, more product-relevant, and better matches the intended demo experience.

## Key Decisions
- Corpus: arXiv `cs.LG` only.
- Time window: April 1, 2016 through March 31, 2026.
- Corpus size: about 5,000 papers for v1.
- Retrieval unit: paper-level records using title + abstract only in v1.
- User task: semantic search for humans, not citation-graph retrieval.
- Benchmark generation: synthetic-first query generation.
- Query styles: mixed terse keyword queries and natural-language researcher questions.
- Review budget: light manual review of about 25-40 queries for the held-out eval slice.
- Project goal: both benchmark and usable tool, but benchmark-first.
- Interface: CLI-first local search tool.
- Modeling strategy: start with a strong zero-shot baseline, then fine-tune a small-but-stronger sentence-transformer style retriever.
- Scope control: no full HTML/PDF fallback logic in v1.

## Constraints
- Must run fully locally and cheaply.
- Primary machine is an Apple Silicon Mac mini M4.
- v1 should remain weekend-scale and avoid infrastructure sprawl.
- arXiv full-text ingestion is deferred; metadata plus abstracts are enough for v1.
- Success must be measured on a reviewed eval slice, not just synthetic labels.

## Success Criteria
- A reproducible corpus build for the selected `cs.LG` slice exists.
- A reviewed held-out eval slice of roughly 25-40 mixed-style human queries exists.
- A zero-shot baseline is established with clear retrieval metrics such as `Recall@k`, `MRR`, and `nDCG`.
- A fine-tuned retriever beats the zero-shot baseline on the reviewed eval slice.
- A CLI search command returns useful top papers for natural researcher queries.
- The repo structure cleanly supports an autoresearch-style iteration loop for future model improvements.

## Open Questions
- Exact base embedding model to use for the zero-shot baseline and fine-tuning path.
- Exact source path for collecting the `cs.LG` corpus and query-generation metadata.
- Whether v1 fine-tuning uses full-model updates, LoRA/PEFT, or another lightweight adaptation path.
- Whether the CLI should expose only search, or also dataset build and evaluation commands from day one.
- Repo name for the fresh project.

## Next Step
Proceed to planning.
