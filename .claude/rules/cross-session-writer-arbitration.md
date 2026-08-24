# Cross-session writer arbitration — shared worktrees, churning files

**Fires when:** you are about to dispatch an implementer/writer into a
worktree an INDEPENDENT live session may be writing, or two sessions must
share one worktree/branch. Pre-split multi-unit builds are the EXPECTED
case, not an edge case: the units build on one branch, and a branch checks
out in exactly one worktree, so a concurrent session working in the same
tree is a normal shape. Durable markers + git probes are the ONLY steering
channels — an independent Happy session is not SendMessage-addressable
(#1586).

## Probe (before dispatch)

1. `uv run python scripts/spawn_session.py list` — live sessions mapped to
   the same issue, or whose cwd sits inside the target worktree.
2. Marker scan — the task's latest `file-set claim:`-leading `epm:progress`
   notes (§ Claim below).
3. Git recency on the intended paths:
   `git -C <WT> log --since='90 minutes ago' --oneline -- <paths>` plus
   `git -C <WT> status --porcelain -- <paths>` — uncommitted churn on the
   intended paths is a live writer.

## Claim (before dispatching a writer)

Post a durable `epm:progress` note leading `file-set claim:` naming
`paths=<comma-list>`, the round, and the owning session — the #1336 v147
shape, now normative. Claims are ADVISORY + time-bounded, and the claim is
the only steering channel that reaches an independent session (#1586).

## Arbitrate

Overlapping live claim or live-writer probe hit ⇒ never dispatch a concurrent writer.
Either sequence-after-commit (wait for the sibling's commit to land / its
claim release) or split to a DISJOINT file set (re-scope the brief). A
stale claim (no commit on the claimed paths AND no heartbeat for >~90 min)
may be overridden with a note naming the staleness evidence.

## Release

Post `file-set release:` when the claimed round lands (or the round's
implementation marker supersedes it).

## Read-pinning under external churn (referenced by both implementer specs)

Record `BASE_SHA=$(git rev-parse HEAD)` at round start. When a target file
changes underneath you, do NOT re-read the live file repeatedly — pin
reads to `git show <BASE_SHA>:<path>`, finish the edit against the
snapshot, take ONE bounded provenance probe
(`git log --oneline <BASE_SHA>..HEAD -- <path>`), and reconcile at commit
time — the commit is the designed conflict-resolution point. The unbounded
re-read loop on a churning 1,132-line file is #1336 death #9.

## Files of record

Task bodies #1336 (v144/v145/v147 + the Unit B death note), #1586, #2158.
No YAML frontmatter by design — behavioral rule, not path-triggered
(`check_rule_frontmatter_parses` treats frontmatter as optional).
