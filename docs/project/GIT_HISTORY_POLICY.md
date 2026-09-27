# Git History and Branch Hygiene

Last updated: September 27, 2026

## Repository Merge Settings

Configured on GitHub repository `davisbuilds/mlsearch`:

- `allow_squash_merge`: `false`
- `allow_merge_commit`: `true`
- `allow_rebase_merge`: `true`
- `delete_branch_on_merge`: `true`
- `merge_commit_title`: `PR_TITLE`
- `merge_commit_message`: `PR_BODY`

Result:

- PR branches retain their full commit history when merged.
- `main` receives either a merge commit (preserving the PR boundary) or rebased commits (linear history), depending on which strategy the merger picks for that PR.
- Squash merging is disabled — full per-commit history is preserved.
- Merged remote branches are auto-deleted.

## Merge Strategy

Merge commits and rebase merges are both allowed; squash merges are disabled. This
is this repository's standing merge policy.

- **Default — merge commit.** Preserves the PR as a discoverable boundary in `main`'s history. Best when the PR contains multiple meaningful commits worth keeping addressable individually.
- **Rebase merge.** Use when the PR's commits are clean and the linear history reads better without an extra merge node. Avoid if the PR's commits are noisy (WIP, fixups) — clean them up locally first.
- **Authoring expectation.** Because squash is gone, individual PR commits land in `main`. Keep PR commit messages tidy: meaningful subjects, no WIP markers, no fixup chains. Squash or reword locally before opening the PR if needed. For agent-assisted work, name the actual assisting agent in the commit/PR co-author trailer. Use `Co-Authored-By: NAME <EMAIL>` with that agent's own attribution address.

## Commit Categories and Releases

Every retained non-merge commit must use a Conventional Commit subject:
`feat(search): add filter`, `fix: correct result ordering`, or a maintenance type
(`docs`, `test`, `chore`, `build`, `ci`, `style`, `refactor`, `revert`, `perf`).
PR and main-push CI validate commit subjects through `scripts/release-commits.cjs`; the PR title
alone cannot repair retained commit history. Actual merge commits are exempt.
The checker handles up to 250 PR or pushed commits; split larger changes. Main
pushes must preserve history and an existing base; incomplete comparisons fail
closed, so direct pushes cannot bypass release category validation.

Use `!` or a `BREAKING CHANGE:` footer when consumers must migrate. Reviewers own
that classification. [Release operations](../system/RELEASES.md) describe pre-1.0
bumps and the generated release PR review gate. Generated release commits use
`chore(main): release ...`; merging a release PR is a release decision.

## CI Gates

GitHub Actions workflow: `.github/workflows/ci.yml` — a **lean gate** that deliberately skips the heavy test suite so CI never installs `torch`/`sentence-transformers`.

Quality gates before merge (also the pre-push expectation locally):

- Conventional Commit subjects on PRs and main pushes (`Release commit categories`)
- `node --test scripts/release-commits.test.cjs` locally
- `uv run ruff check .`
- `uv run ruff format --check .`
- the dead-code test (ephemeral `uv run --no-project` env)

CI does **not** run the full `pytest` suite — always run `uv run pytest -q` locally before claiming completion.

## Branch Protection Status

This is a public repository, so the branch-protection APIs are available. No
required reviews or status checks are enforced as branch rules yet — CI gates
below are enforced by convention. Enable `required_conversation_resolution` when
the review flow warrants it.

## Recommended Ongoing Hygiene

1. Create short-lived feature branches from `main` (`feat/*`, `fix/*`, `docs/*`, `chore/*`).
2. Open PRs early; keep them focused on one intent.
3. Tidy your PR commit history *before* merging — reword/squash locally so what lands on `main` reads cleanly.
4. Pick **Create a merge commit** by default; pick **Rebase and merge** when linear history is materially better.
5. Periodically prune local branches:

```bash
git fetch --prune
git branch --merged main | grep -v ' main$' | xargs -n 1 git branch -d
```
