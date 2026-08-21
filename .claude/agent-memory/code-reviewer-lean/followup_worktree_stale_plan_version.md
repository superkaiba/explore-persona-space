---
name: followup-worktree-stale-plan-version
description: "Split-review briefs pointing at a WORKTREE-relative plan path can resolve to a PRE-CRITIQUE draft — resolve plans/vN.md from the MAIN checkout and state the version used per plan-grounded finding (#2329 q35_ladder_decay r1 g8)"
metadata:
  type: feedback
---

Rule: in a follow-up-round split review, NEVER trust the brief's worktree-relative
plan path. The round worktree is cut from `origin/main@<base>` BEFORE the round's
critique cycles land, so its `tasks/.../plans/` holds only the plan versions that
existed at cut time and `plan.md` symlinks a pre-critique draft (observed #2329
q35_ladder_decay: worktree had v1–v4, `plan.md → v4.md`; the approved plan was v8
on main, carrying the R2-M1/R2-S1–S4/F1–F4 fold-ins — including the exact
parenthetical my commit's gate change was supposed to be verified against, which
did not exist in v4 at all). The manifest (`artifacts/planned_manifest.json`) goes
stale the same way.

**Why:** a reviewer validating plan adherence against the stale draft either
false-FAILs items the critique added, or PASSes a substituted gate quantity the
approved plan forbids — and on a CONTRACT-BEARING group the whole round verdict
inherits the error. The orchestrator confirmed the hazard mid-round and asked for
per-finding plan-version citations.

**How to apply:** (1) resolve the plan from the MAIN checkout's canonical tasks
dir (`task.py find <N>`/`plans/`), pick the HIGHEST vN (cross-check the round's
`epm:plan-revision-log` / `epm:plan-approved` markers name it); (2) state in the
verdict WHICH plan version each plan-grounded finding was checked against;
(3) add a round-level observation naming which sibling sub-scopes the stale copy
could have affected (items that exist only in the approved version). Pairs with
[[registered-gate-quantity-substituted]] — the literal parenthetical you diff a
gate against must come from the APPROVED plan version.
