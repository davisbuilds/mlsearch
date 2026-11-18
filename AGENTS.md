# AGENTS.md

MLSearch is a benchmark-first local retrieval project for arXiv `cs.LG` papers.
Keep the corpus and reviewed eval logic fixed, mutate only narrow retrieval,
training, and experiment surfaces, and judge changes mechanically on the
reviewed held-out eval.

## Documentation Map

- `CONTRIBUTING.md` — contribution scope, review expectations, delivery policy.
- `docs/system/ARCHITECTURE.md` — runtime shape, CLI routing, path model,
  config, data and artifact boundaries.
- `docs/system/BENCHMARK.md` — benchmark artifacts, review workflow, split
  rules, review heuristics.
- `docs/system/TRAINING.md` — training path, sweep loop, reranking, recorded
  champion findings.
- `docs/system/OPERATIONS.md` — setup, lean CI, local verification, troubleshooting.
- `docs/system/RELEASES.md` — version ownership and CI-gated source releases.
- `docs/project/ROADMAP.md`, `BACKLOG.md` — direction and future-only follow-ups.
- `docs/project/GIT_HISTORY_POLICY.md` — merge strategy and commit subjects.

## Benchmark and Experiment Invariants

- **Workflow order:** corpus, synthetic query candidates, review a small batch
  into the held-out eval, baseline retrieval, retriever training, then optional
  reranking. Do not change benchmark generation and model recipes in the same
  step unless the user explicitly asks for that coupling.
- **Reviewed queries never leak into training.** The `dev`
  (`data/benchmark/reviewed/held_out_eval.jsonl`) and `test`
  (`held_out_eval_test.jsonl`) splits are paper-disjoint from training: no
  generated query for a held-out paper is used to train. Use `dev` for tuning
  and `test` for blind checks.
- **Reviewed splits are gitignored local data.** A clone without them cannot
  run evals; say so instead of claiming an eval result.
- **Keep champion semantics explicit.** "Beats baseline" is not "beats
  incumbent". Rerank only after first-stage recall is strong; reranking is a
  second-stage experiment, not a retriever replacement.
- **The eval is the real judge** for retrieval, ranking, and training changes.
  Rerun the relevant eval and report absolute metrics; a passing unit test
  shows only that the code runs.

## Verification

Local gate (pre-push):

```bash
node --test scripts/release-commits.test.cjs
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest -q
python3 scripts/check_publication_hygiene.py
```

**CI is lean and never runs the full pytest suite**, so it cannot catch a
failing test: it runs the commit-category check, the locked dev-group ruff, the
dead-code test, and the publication-hygiene scan.
Green CI does not mean tests passed; run the full suite locally and record the
result in the PR. Use targeted test slices while iterating. The dead-code gate
(`tests/test_dead_code.py`) takes intentional entries in `SYMBOL_EXCEPTIONS` /
`MODULE_EXCEPTIONS` with a reason.

## Git

Merge commit or rebase; squash is disabled. Every retained non-merge commit
needs a Conventional Commit subject, which CI checks
(`docs/project/GIT_HISTORY_POLICY.md`). Release Please owns versions and
changelogs: merging its release PR creates a tag and GitHub Release, so that
merge waits for the maintainer. Merging other PRs publishes source only. This
repository is public; keep commits, PR text, and docs self-contained.

## Working Agreement

- **Clarify before building** when a request is vague, contradictory, or skips a
  key decision; clear, well-scoped requests need no interview.
- **Fix small issues now; backlog the rest.** Fix simple, quick, or blocking
  issues in this repo as part of the current change and call them out. Record in
  `docs/project/BACKLOG.md` only what can't be finished now or needs the
  maintainer's decision; don't open a branch or PR just for a backlog entry.
- **Keep docs current** when a change makes a documented contract, boundary,
  procedure, or direction stale.
- **Use `tmux` for long-running work.** `mlsearch train`,
  `mlsearch experiment sweep`, and full-corpus index builds outlast a session
  and must survive disconnect and sleep. Run them detached, keep the scrollback,
  and clean up the session afterward.
- **Plan and hand off work.** Reuse clear task context. Save planning docs when
  decisions, acceptance, or coordination need persistence. Leave a brief
  handoff only when needed to resume unfinished work. Commit shared docs and
  keep status current. Before archiving reviewed, completed docs in ignored
  `docs/archive/`, preserve lasting decisions and open work in maintained docs
  and repair links.
- **TDD:** red/green for features, major refactors and large changes; the red
  test must fail on the behavior, not on a missing symbol. For smaller edits, run
  the relevant existing checks.
- **Commit and push.** Commit coherent chunks, staging by explicit path. Push
  your task branch and open a PR once the work is complete and the local gate
  passes; don't push directly to `main`.
- **Merge** only under the standing merge authority in the maintainer's
  workspace instructions, or when the maintainer asks. Without those
  instructions in context, leave the PR for the maintainer.
- **Re-ground after compaction** in source, this file, the owning docs, and
  recent commits. A summary and your own earlier proposals are not user approval.
