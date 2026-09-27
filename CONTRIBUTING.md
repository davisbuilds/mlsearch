# Contributing

Bug reports, focused fixes, documentation improvements, and supported proposals
are welcome. Discuss benchmark/split changes, model promotion, new dependencies,
or broad search interfaces before major implementation. This is a solo-maintained
research tool; contributions do not imply a support or response-time promise.

Agent-assisted work is welcome. Submitters should understand the change's intent,
important behavior, tradeoffs, and verification. For retrieval or ranking changes,
report absolute results on the relevant reviewed split and protect blind-test
discipline; a passing unit test alone does not establish an improvement. Explain
limitations in the PR. No prompt transcript or manual rewrite is required.

A clear [Backlog](docs/project/BACKLOG.md) entry can go directly to a PR; use an
issue when discussion or coordination helps. Work on a focused branch from `main`.
[Operations](docs/system/OPERATIONS.md) owns setup and local full-suite checks
that CI omits; [Git policy](docs/project/GIT_HISTORY_POLICY.md) owns retained
Conventional Commits and merge strategy. [Release operations](docs/system/RELEASES.md)
separate package versions from model promotion. Review generated consumer notes
and migration steps in the release PR rather than editing a parallel log.
