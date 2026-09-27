# MLSearch Roadmap

## Current Direction

MLSearch is a local-first retrieval project for arXiv `cs.LG` papers. Keep the
corpus and reviewed evaluation logic fixed while changing retrieval, reranking,
training, or experiment surfaces one at a time. Judge proposed defaults on
paper-disjoint held-out data, not anecdotal search examples.

## Operating Sequence

1. Maintain the reviewed benchmark before broadening model or recipe work.
2. Use `dev` for tuning and preserve `test` for blind validation.
3. Report absolute evaluation metrics before replacing the current default.

The local baseline-rerank path remains the documented operating default unless
a newer paper-disjoint evaluation supports a change.
[Benchmark](../system/BENCHMARK.md) and [Training](../system/TRAINING.md) own the
protocol and evidence; [Operations](../system/OPERATIONS.md) owns local checks.

## Boundaries

The CLI is local and Apple-Silicon-first. The v1 index covers titles and
abstracts rather than full papers. GitHub source releases version the package,
not a model promotion or benchmark achievement; [Releases](../system/RELEASES.md)
own that distinction. Full behavioral verification remains local because CI
intentionally omits the heavy ML stack.

Unselected future work belongs in [Backlog](BACKLOG.md). Git and experiment
records preserve detailed completed history.
