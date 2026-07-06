---
name: GCE workload scripts must source ./.env conditionally
description: Pod/GCE-side shell scripts never unconditionally `. ./.env` — GCE startup scripts export tokens with no .env file; an &&-chained sourcing failure silently short-circuits the real command
type: feedback
---

`set -a && . ./.env && set +a && <cmd>` inside a subshell: on GCE the .env does not exist (tokens are
exported by the startup script), the sourcing fails, `<cmd>` never runs, and the subshell's rc=1 gets
mislabeled by fail-fast rc classifiers as a non-transient app failure.
**Why:** #923 att-20260703-163121 — the join poll died this way immediately after a fully successful
Phase-2 reduce; the sentinel it was polling for already existed.
**How to apply:** in any script that runs on pod/GCE lanes, use
`if [ -f ./.env ]; then set -a; . ./.env; set +a; fi` (VM/worktree venues keep working); never put the
sourcing inside the && chain of the command whose rc is being classified.
