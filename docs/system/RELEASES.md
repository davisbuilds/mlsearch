# Releases

MLSearch releases identify the Python package and local CLI source. Users clone
or check out a version and run `uv sync --locked --group dev`; this automation
creates GitHub tags/releases, with no PyPI publication or model/artifact upload.
A package release does not certify a model improvement or change the reviewed
benchmark, champion, or training gates.

## Version and compatibility intent

`pyproject.toml` owns the package version. Runtime `mlsearch.__version__` reads
installed distribution metadata, falling back to the source project's TOML when
running an uninstalled checkout with `PYTHONPATH=src`. Release Please updates
`pyproject.toml`, the `mlsearch` entry in `uv.lock`, its manifest, and `CHANGELOG.md`.

PR CI validates its own commits. Main-push CI fetches complete Git history, checks
that the push preserves ancestry, and validates the full unreleased window from
the real `v<manifest version>` tag (annotated or lightweight) to the tested head.
If that tag is absent, it conservatively uses the configured bootstrap commit;
this includes the interval after a release PR bumps the manifest but before its
tag exists. Earlier failed pushes remain covered. Accumulated history has no
250-commit limit, while the PR API check retains its 250-commit bound.

Each retained non-merge commit needs a Conventional Commit subject:
`type(scope): description` (scope optional). Use `feat` for a feature and `fix` or
`perf` for a patch. A `!` or `BREAKING CHANGE:` footer declares incompatibility.
Before 1.0, features and breaking changes bump the minor version; fixes and
performance changes bump the patch. Other categories do not independently start
a release. Syntax checking cannot decide whether a change is breaking. The 1.0
transition and any manual release override require an explicit maintainer decision.

## Release boundary and review

`.github/workflows/release-please.yml` runs only after successful same-repository
main-push `CI`. It checks that main still matches the tested SHA before obtaining
a repository-scoped GitHub App token. The job has no checkout, artifact download,
or cache loading; release writes are serialized without cancelling an active
writer. The SHA check is a preflight, not a lock against subsequent pushes.

Repository setup requires Actions variable `RELEASE_APP_CLIENT_ID`, secret
`RELEASE_APP_PRIVATE_KEY`, and an installed App with Contents, Issues, and Pull
requests write permissions. Provision those through private credential storage;
never commit the key. App tokens allow generated release PRs to start normal CI.

Review and merge release PRs under the normal Git history and CI policy. Verify
version/lockfile agreement, generated changelog and PR body, and compatibility
intent. Full local pytest remains required because hosted CI skips the ML stack.
After a release PR merge and successful main CI, the workflow creates its tag
and GitHub Release. No workflow automatically merges PRs.

## Bootstrap and recovery

As of September 27, 2026, there were no tags or GitHub Releases. `0.1.0` is the
existing package version, not a historical published release. The manifest starts
there, and bootstrap commit `103e7c34aeea52a51905f3e2209b6d76d88362ae` bounds
future release notes. Do not create a fictitious `v0.1.0` tag or replay earlier
unclassified history. Release Please currently generates a
comparison from the manifest's synthetic `v0.1.0` baseline even though that tag
does not exist. Before merging the first release PR, correct both its changelog
and PR body comparison to `103e7c34aeea52a51905f3e2209b6d76d88362ae...v0.2.0`
(or the actual proposed version). Recheck after any automation refresh. Do not
create a baseline tag to repair the link.

If CI failed or its revision was superseded, fix/run CI for the current main
revision; do not bypass the release gate. An unclassified commit already on main
keeps later pushes blocked until an explicit maintainer recovery decision. Stop
the release writer and review that commit's compatibility intent, version impact,
and release notes. Do not silently reword published history or create a fake
baseline tag to clear the gate. Any necessary baseline adjustment must be an
explicitly reviewed decision documenting how the omitted changes are accounted
for, followed by validation of the remaining history before resuming releases. For credential failures, repair App
setup and rerun the failed release job only if its tested revision is still
current. Inspect remote PR/tag/release state before retrying an ambiguous write.
Disable the release workflow before manual recovery so there remains one writer;
never move a published tag silently. Registry publishing remains separate work.

Local checks:

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run python -m pytest -q
node --test scripts/release-commits.test.cjs
uvx zizmor@1.30.0 --offline .github/workflows/
```
