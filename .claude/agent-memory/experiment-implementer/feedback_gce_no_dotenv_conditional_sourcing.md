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

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [GCE workload scripts must source ./.env conditionally](feedback_gce_no_dotenv_conditional_sourcing.md) — GCE exports tokens with no .env; &&-chained sourcing silently kills the real command (#923)
