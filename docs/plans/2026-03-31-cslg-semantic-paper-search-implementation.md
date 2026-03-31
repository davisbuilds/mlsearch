---
date: 2026-03-31
topic: cslg-semantic-paper-search
stage: implementation-plan
status: draft
source: conversation
---

# cs.LG Semantic Paper Search Implementation Plan

## Goal

Build a fresh local-first ML research repo that indexes about 5,000 arXiv `cs.LG` papers from April 1, 2016 through March 31, 2026, establishes a reviewed semantic-search benchmark over title-plus-abstract retrieval, beats a zero-shot baseline with local fine-tuning, and exposes the same retrieval pipeline through a CLI-first search tool.

## Scope

### In Scope

- Create a new standalone repo rooted at `/Users/dg-mac-mini/Dev/arxiv-cslg-search`.
- Build a reproducible `cs.LG` corpus pipeline using arXiv metadata and abstracts only.
- Generate a synthetic-first mixed-query benchmark and a lightweight manual review flow.
- Implement zero-shot embedding retrieval, indexing, and benchmark evaluation.
- Implement a local fine-tuning path for a small-but-stronger sentence-transformer style retriever.
- Ship a CLI with build, eval, train, and search commands.
- Add an autoresearch-style experiment ledger and task surface for future keep/discard iterations.

### Out of Scope

- Full HTML/PDF/source ingestion or section-level retrieval.
- Public hosting, remote APIs, paid evaluation, or SaaS deployment.
- Multi-node training, distributed indexing, or large-scale arXiv mirroring.
- GUI or web application work in v1.
- Production-grade ranking infrastructure beyond a local prototype.

## Assumptions And Constraints

- Primary execution environment is an Apple Silicon Mac mini M4 with local storage and no required cloud budget.
- v1 corpus size is capped near 5,000 papers to preserve fast local iteration and manageable index rebuild times.
- arXiv metadata and abstracts are sufficient for v1; full e-print ingestion is deferred by design.
- The benchmark must include a manually reviewed held-out slice of about 25-40 queries to avoid optimizing purely against synthetic noise.
- The first model iteration should prioritize established embedding tooling over Apple-native specialization; MPS acceleration is desirable but not allowed to dominate architecture decisions.
- The repo should be shaped so future autoresearch-style loops can mutate a narrow training/config surface while keeping data and evaluation harnesses fixed.

## Task Breakdown

### Task 1: Bootstrap Repo Structure And CLI Skeleton

**Objective**

Create the new project scaffold, package metadata, reproducible environment, and a CLI surface that can grow without reshaping the repo later.

**Files**

- Create: `README.md`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/arxiv_cslg_search/__init__.py`
- Create: `src/arxiv_cslg_search/cli.py`
- Create: `src/arxiv_cslg_search/config.py`
- Create: `src/arxiv_cslg_search/paths.py`
- Create: `tests/test_cli_smoke.py`
- Create: `docs/architecture.md`

**Dependencies**

None

**Implementation Steps**

1. Initialize the repo structure with `src/`, `tests/`, `docs/`, `data/`, and `artifacts/` directories plus a `uv`-managed Python project.
2. Define a single CLI entrypoint with subcommands for `corpus`, `benchmark`, `index`, `eval`, `train`, and `search`, even if most subcommands initially stub out.
3. Centralize filesystem paths and runtime config so later tasks do not spread hard-coded paths across the codebase.
4. Document the project goal, constraints, and intended workflow in `README.md` and `docs/architecture.md`.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli --help`
- Expect: top-level help shows the planned subcommands without import errors
- Run: `uv run pytest tests/test_cli_smoke.py`
- Expect: CLI smoke test passes and package imports resolve cleanly

**Done When**

- The repo installs and exposes a working CLI entrypoint.
- Project paths and config are centralized.
- The basic repository layout supports all later tasks without reorganization.

### Task 2: Build The cs.LG Corpus Pipeline

**Objective**

Implement a reproducible data pipeline that fetches, filters, normalizes, and stores about 5,000 `cs.LG` records for the agreed 10-year window.

**Files**

- Create: `src/arxiv_cslg_search/data/arxiv_client.py`
- Create: `src/arxiv_cslg_search/data/models.py`
- Create: `src/arxiv_cslg_search/pipelines/build_corpus.py`
- Create: `src/arxiv_cslg_search/pipelines/validate_corpus.py`
- Create: `configs/corpus.yaml`
- Create: `tests/test_arxiv_client.py`
- Create: `tests/test_corpus_validation.py`
- Create: `data/README.md`

**Dependencies**

- Task 1

**Implementation Steps**

1. Implement a rate-limit-aware arXiv metadata client for `cs.LG` queries covering April 1, 2016 through March 31, 2026.
2. Normalize raw responses into a stable paper schema containing at least arXiv id, title, abstract, authors, published date, updated date, categories, and URL fields.
3. Add deterministic corpus selection logic that trims the window down to about 5,000 papers while recording selection criteria for reproducibility.
4. Persist raw pulls and normalized corpus artifacts separately so rebuilds and audits are possible.
5. Add validation checks for duplicate ids, missing abstracts, category drift, and date-window violations.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli corpus build --limit 5000`
- Expect: corpus artifact is created with about 5,000 valid `cs.LG` records
- Run: `uv run python -m arxiv_cslg_search.cli corpus validate`
- Expect: validation exits successfully and reports zero schema/date/duplicate failures
- Run: `uv run pytest tests/test_arxiv_client.py tests/test_corpus_validation.py`
- Expect: client parsing and validation tests pass

**Done When**

- The corpus can be rebuilt from source deterministically.
- A validated normalized dataset artifact exists for downstream tasks.
- Corpus size and date/category constraints are enforced mechanically.

### Task 3: Generate Synthetic Queries And Reviewable Eval Slice

**Objective**

Create the benchmark generation flow for mixed-style semantic-search queries and a manual review workflow that yields a trustworthy held-out eval subset.

**Files**

- Create: `src/arxiv_cslg_search/pipelines/generate_queries.py`
- Create: `src/arxiv_cslg_search/pipelines/sample_review_set.py`
- Create: `src/arxiv_cslg_search/benchmark/schema.py`
- Create: `src/arxiv_cslg_search/benchmark/review.py`
- Create: `configs/benchmark.yaml`
- Create: `data/benchmark/review_template.csv`
- Create: `tests/test_query_generation.py`
- Create: `tests/test_review_schema.py`
- Create: `docs/benchmark.md`

**Dependencies**

- Task 2

**Implementation Steps**

1. Define a benchmark schema that separates generated query candidates, source positives, reviewed labels, and held-out eval splits.
2. Implement synthetic query generation from titles, abstracts, keywords, and phrase templates to produce both terse search strings and natural-language researcher questions.
3. Attach intended positives and candidate hard negatives so later training and evaluation can reuse the same benchmark data model.
4. Implement a review sampler that selects about 25-40 representative queries for manual inspection and editing.
5. Add import/export tooling for a lightweight review artifact such as CSV or JSONL so manual corrections can be merged back deterministically.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli benchmark generate`
- Expect: generated benchmark artifact contains mixed query styles and linked source positives
- Run: `uv run python -m arxiv_cslg_search.cli benchmark sample-review --count 30`
- Expect: review file is created with exactly 30 candidate queries
- Run: `uv run pytest tests/test_query_generation.py tests/test_review_schema.py`
- Expect: query generation and review schema tests pass

**Done When**

- The repo can generate a synthetic-first benchmark without manual intervention.
- A lightweight reviewed eval slice workflow exists and preserves edits cleanly.
- Query artifacts are structured for both evaluation and fine-tuning.

### Task 4: Establish Zero-Shot Baseline Retrieval And Metrics

**Objective**

Implement the first real retrieval baseline, index build, and evaluation harness so future improvement claims have a stable point of comparison.

**Files**

- Create: `src/arxiv_cslg_search/retrieval/embedder.py`
- Create: `src/arxiv_cslg_search/retrieval/index.py`
- Create: `src/arxiv_cslg_search/retrieval/search.py`
- Create: `src/arxiv_cslg_search/eval/metrics.py`
- Create: `src/arxiv_cslg_search/eval/run_eval.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_retrieval_smoke.py`
- Create: `artifacts/results/.gitkeep`

**Dependencies**

- Task 2
- Task 3

**Implementation Steps**

1. Implement document text packing for paper-level retrieval using title plus abstract in a single canonical representation.
2. Add embedding model loading and batching for one strong zero-shot baseline model plus a secondary candidate if needed for comparison.
3. Build a local vector index and retrieval path for top-k paper search.
4. Implement benchmark metrics including at least `Recall@k`, `MRR`, and `nDCG`.
5. Persist baseline evaluation outputs under `artifacts/results/` with timestamps and model identifiers so future runs remain comparable.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli index build`
- Expect: local embedding/index artifacts are created successfully
- Run: `uv run python -m arxiv_cslg_search.cli eval baseline`
- Expect: evaluation completes and writes a metrics report under `artifacts/results/`
- Run: `uv run pytest tests/test_metrics.py tests/test_retrieval_smoke.py`
- Expect: metrics and retrieval smoke tests pass

**Done When**

- A reproducible zero-shot baseline exists with persisted metrics.
- Index build and search are local and deterministic from the corpus artifact.
- Future gains can be measured against a stable baseline report.

### Task 5: Ship The CLI-First Search Experience

**Objective**

Turn the retrieval pipeline into a useful local search tool rather than leaving it as an internal benchmark only.

**Files**

- Modify: `src/arxiv_cslg_search/cli.py`
- Create: `src/arxiv_cslg_search/present/search_output.py`
- Create: `tests/test_search_cli.py`
- Modify: `README.md`

**Dependencies**

- Task 4

**Implementation Steps**

1. Implement `search` subcommands that accept a free-form query and return top-k papers with scores and concise metadata.
2. Add output formats that work well in the terminal, with a readable default table and at least one machine-readable format.
3. Make the CLI read from the local built index by default and fail clearly when prerequisites are missing.
4. Document the common end-user flow from corpus build through search in `README.md`.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli search \"test-time adaptation under distribution shift\" --top-k 5`
- Expect: five ranked papers are returned with ids, titles, and scores
- Run: `uv run python -m arxiv_cslg_search.cli search \"graph neural networks for molecules\" --format json`
- Expect: valid JSON output with ranked hits
- Run: `uv run pytest tests/test_search_cli.py`
- Expect: CLI search behavior is covered by tests

**Done When**

- The CLI is genuinely usable for local paper search.
- Search output is readable in a terminal and scriptable when needed.
- Missing-index and bad-input paths are handled explicitly.

### Task 6: Add Local Fine-Tuning Pipeline

**Objective**

Implement the first local training loop that uses the benchmark artifacts to improve retrieval over the reviewed eval slice.

**Files**

- Create: `src/arxiv_cslg_search/training/dataset.py`
- Create: `src/arxiv_cslg_search/training/train_retriever.py`
- Create: `src/arxiv_cslg_search/training/checkpoints.py`
- Create: `configs/train.yaml`
- Create: `tests/test_training_dataset.py`
- Create: `docs/training.md`

**Dependencies**

- Task 3
- Task 4

**Implementation Steps**

1. Define train/dev/eval splits that protect the reviewed slice from leakage.
2. Convert benchmark artifacts into a retrieval training dataset with positives and useful hard negatives.
3. Implement a local training path for a small-but-stronger sentence-transformer style encoder with a lightweight adaptation option if full-model training is too slow on M4.
4. Save checkpoints, configs, and metrics in a structure compatible with later autoresearch-style comparisons.
5. Add a compare step that evaluates trained checkpoints against the persisted zero-shot baseline.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli train --config configs/train.yaml`
- Expect: at least one checkpoint and training summary artifact are produced successfully
- Run: `uv run python -m arxiv_cslg_search.cli eval compare --model latest`
- Expect: comparison report shows trained-model metrics versus zero-shot baseline
- Run: `uv run pytest tests/test_training_dataset.py`
- Expect: training dataset construction tests pass

**Done When**

- The repo can train a local retriever from benchmark artifacts without touching the eval harness.
- Model comparisons against the zero-shot baseline are automated.
- The project has a credible path to meeting the “beats baseline” success criterion.

### Task 7: Add Autoresearch-Style Experiment Ledger And Fixed Surfaces

**Objective**

Shape the repo so future autonomous iteration can mutate a narrow surface while preserving fixed data and evaluation boundaries.

**Files**

- Create: `program.md`
- Create: `results.tsv`
- Create: `src/arxiv_cslg_search/experiments/logging.py`
- Create: `src/arxiv_cslg_search/experiments/compare.py`
- Create: `tests/test_experiment_logging.py`
- Modify: `README.md`

**Dependencies**

- Task 4
- Task 6

**Implementation Steps**

1. Define the human-owned research policy in `program.md`, including fixed corpus/eval rules and allowed optimization surfaces.
2. Add a `results.tsv` ledger format for baseline, fine-tuned runs, and future keep/discard experiments.
3. Implement helpers that append structured experiment rows and compare new runs against the current best metrics mechanically.
4. Document which files are mutable in future autonomous loops and which are frozen by policy.

**Verification**

- Run: `uv run python -m arxiv_cslg_search.cli eval compare --model latest --record-results`
- Expect: a new row is appended to `results.tsv` with metrics and status
- Run: `uv run pytest tests/test_experiment_logging.py`
- Expect: ledger logging and comparison logic tests pass

**Done When**

- The repo has an explicit fixed-versus-editable boundary for future autonomous research.
- Experiment history is recorded in a stable, append-only artifact.
- Model improvement decisions can be made mechanically from stored metrics.

### Task 8: Reproducibility Pass And End-To-End Verification

**Objective**

Close the loop with documentation, a clean end-to-end workflow, and enough verification to execute confidently in the next session.

**Files**

- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/benchmark.md`
- Modify: `docs/training.md`
- Create: `tests/test_e2e_smoke.py`

**Dependencies**

- Task 1
- Task 2
- Task 3
- Task 4
- Task 5
- Task 6
- Task 7

**Implementation Steps**

1. Document the exact happy-path workflow from corpus build to benchmark generation, baseline eval, fine-tuning, comparison, and search.
2. Add one end-to-end smoke test that exercises the smallest practical local path with a tiny fixture corpus.
3. Ensure the repo can be bootstrapped by a zero-context engineer from the docs alone.
4. Record known limitations and explicitly defer full-text and section-level retrieval to a later version.

**Verification**

- Run: `uv run pytest tests/test_e2e_smoke.py`
- Expect: fixture-based end-to-end smoke test passes
- Run: `uv run python -m arxiv_cslg_search.cli --help`
- Expect: all planned subcommands are discoverable and documented
- Run: `uv run python -m arxiv_cslg_search.cli search \"meta-learning for few-shot classification\" --top-k 3`
- Expect: search works after following the documented setup path

**Done When**

- The repo has a documented end-to-end workflow.
- A minimal smoke path proves the major subsystems compose correctly.
- The next execution session can start building without re-deciding scope.

## Risks And Mitigations

- Risk: Synthetic queries are too templated and lead to reward hacking.
  Mitigation: keep the reviewed eval slice separate, preserve mixed query styles, and compare qualitative search outputs before trusting metric gains.
- Risk: M4 training throughput is slower than expected.
  Mitigation: start from a strong zero-shot baseline, prefer lightweight adaptation if needed, and keep corpus size capped at about 5,000 papers.
- Risk: arXiv metadata retrieval or date/category filtering produces a skewed corpus.
  Mitigation: persist raw pulls, validate normalized outputs mechanically, and record corpus selection rules in config and docs.
- Risk: Fine-tuning appears to improve synthetic metrics but not real search quality.
  Mitigation: require comparison on the reviewed eval slice and spot-check representative CLI queries before accepting model changes.
- Risk: CLI design sprawls into too many commands before core behavior is stable.
  Mitigation: keep v1 subcommands narrow and focused on build, eval, train, and search only.

## Verification Matrix

| Requirement | Proof command | Expected signal |
| --- | --- | --- |
| Repo bootstrap is usable | `uv run python -m arxiv_cslg_search.cli --help` | CLI help renders without import or config errors |
| Corpus build is reproducible and valid | `uv run python -m arxiv_cslg_search.cli corpus validate` | Validation passes with zero duplicate/schema/date errors |
| Benchmark generation works | `uv run python -m arxiv_cslg_search.cli benchmark generate` | Query artifact is created with linked positives |
| Reviewed eval slice workflow exists | `uv run python -m arxiv_cslg_search.cli benchmark sample-review --count 30` | Review file with 30 candidate queries is produced |
| Zero-shot baseline is established | `uv run python -m arxiv_cslg_search.cli eval baseline` | Baseline metrics report is written to `artifacts/results/` |
| CLI search is usable | `uv run python -m arxiv_cslg_search.cli search "test-time adaptation under distribution shift" --top-k 5` | Ranked paper results are printed with ids, titles, and scores |
| Fine-tuning path runs locally | `uv run python -m arxiv_cslg_search.cli train --config configs/train.yaml` | Training completes enough to emit a checkpoint and summary artifact |
| Trained model comparison is mechanical | `uv run python -m arxiv_cslg_search.cli eval compare --model latest` | Output reports trained-model metrics alongside the zero-shot baseline |
| Autoresearch-style ledger is wired | `uv run python -m arxiv_cslg_search.cli eval compare --model latest --record-results` | `results.tsv` receives a new structured row |
| End-to-end workflow composes | `uv run pytest tests/test_e2e_smoke.py` | Fixture-based smoke path passes |

## Handoff

1. Execute in this session, task by task.
2. Open a separate execution session.
3. Refine this plan before implementation.
