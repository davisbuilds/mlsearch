# MLSearch Roadmap

## Current Direction

MLSearch is a local-first, benchmark-first retrieval project for arXiv `cs.LG` papers.
The corpus and reviewed evaluation logic stay fixed while retrieval, training, and
experiment surfaces are changed one at a time and judged mechanically on held-out data.

## Shipped Foundation

- **Reproducible local retrieval** — the CLI builds and validates an arXiv `cs.LG`
  corpus, creates embedding indexes, and supports semantic search without hosted services.
- **Reviewed evaluation discipline** — generated query candidates are reviewed into
  `dev` and blind `test` splits; reviewed source papers remain excluded from training.
- **Experiment loop** — baseline retrieval, training, sweeps, second-stage reranking,
  and a results ledger support mechanical comparison rather than anecdotal model choice.
- **Useful local default** — the documented baseline-rerank path is the current
  trustworthy operating path unless a newer paper-disjoint evaluation beats it.

## Current Operating Sequence

1. Maintain the reviewed benchmark before broadening model or recipe work.
2. Use `dev` for tuning and preserve `test` for blind validation.
3. Change one of benchmark, retriever, reranker, or training recipe at a time; report
   absolute evaluation metrics before replacing the current default.

## Product Boundaries

- Local and Apple-Silicon-first; the project uses no secrets or hosted inference.
- CLI-first rather than web-first.
- v1 indexes titles and abstracts, not full-text HTML or PDFs.
- Full behavioral verification remains a local responsibility because CI intentionally
  avoids the heavy ML dependency stack.

Completed work is recorded here; future friction and deferred follow-ups belong in
[BACKLOG.md](BACKLOG.md).
