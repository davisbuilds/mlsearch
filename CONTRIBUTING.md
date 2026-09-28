# Contributing

## Welcome and scope

Bug reports, focused fixes, documentation improvements, and supported proposals
are welcome. Discuss benchmark/split changes, model promotion, new dependencies,
or broad search interfaces before major implementation.

This is a solo-maintained project; contributions do not imply a support or
response-time commitment.

## Understanding and agent use

Agent-assisted work is welcome. Submitters should understand the change's purpose,
important behavior, tradeoffs, and verification limits. Explain what you checked
and what remains uncertain; no prompt transcript or manual rewrite is required.

For retrieval or ranking changes, report absolute results on the relevant reviewed
split and protect blind-test discipline. Passing unit tests alone do not establish
an improvement.

## Choosing work

[Roadmap](docs/project/ROADMAP.md) records selected direction;
[Backlog](docs/project/BACKLOG.md) records unresolved work. Backlog entries can be
delegated directly to agents or become focused PRs. Use an issue when persistent
discussion, investigation, or coordination helps; there is no mandatory graduation
step. An entry or issue alone is not a feature commitment. When an issue owns the
details, keep only a useful linked summary in the backlog.

## Delivering a change

Work on a focused branch from `main` (or an appropriate parent for stacked work).
Keep commits coherent. Describe the problem and resulting behavior in the PR,
with relevant verification and limitations. Merge after applicable checks pass
and review conversations are resolved.

[Operations](docs/system/OPERATIONS.md) owns setup and local full-suite checks
that CI omits. [Git policy](docs/project/GIT_HISTORY_POLICY.md) owns retained
Conventional Commits and merge strategy. [Release operations](docs/system/RELEASES.md)
separates package versions from model promotion. Review generated consumer notes
and migration steps in the release PR instead of maintaining a parallel log.

Update the owning reference when its claims change and reconcile affected backlog
entries. Roadmap tracks direction; Git and PRs hold routine delivery history.
