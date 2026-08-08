---
name: adoptable-tests-no-worktree-paths-no-network
description: Evidence/regression tests offered for tests/ adoption must resolve paths repo-root-relative and gate live HF downloads — Step 9c runs them fleet-wide (#1491)
metadata:
  type: feedback
---

Any evidence/regression test suite offered for adoption into `tests/` (even
when held in /tmp under a file-set ownership constraint) must ALREADY be
adoption-shaped: (1) NO hardcoded worktree paths — `WT =
Path("/home/.../.claude/worktrees/issue-<N>")` breaks from the repo root and
the worktree is reaped by the stale-worktree cron; resolve via
`git rev-parse --path-format=absolute --git-common-dir` (parent = main repo
root) or a fixture; (2) NO ungated live network fetches —
`AutoTokenizer/AutoConfig.from_pretrained` are live HF downloads on a cache
miss, and anything in `tests/` runs in EVERY issue's Step 9c gate, so an HF
outage / 429 storm turns the fleet-wide gate red; network-dependent tests get
a skip/gate.

**Why:** #1491 (2026-08-04) — my 13-test blocker-evidence suite was rejected
for verbatim adoption on exactly these two blockers ("please don't ship this
shape again"), despite the pinning value being wanted.

**How to apply:** when writing ANY test I expect (or offer) to be committed,
write it repo-root-relative + network-gated from the first draft — the /tmp
location does not exempt it from adoption shape.
