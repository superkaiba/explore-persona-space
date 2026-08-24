---
name: spec-freshness-sync-commit-review
description: Mechanical 4-check recipe for reviewing a Step 5a spec-freshness sync commit in a split review — blob-OID identity vs sync-time main, SPECS containment, subject anchor, no-lost-row shortcut
metadata:
  type: feedback
---

Review a Step 5a sync commit (`sync workflow-surface specs from`) with 4 mechanical probes, never by prose inspection — the content is main's, not the branch's, so prose findings are noise:

1. **Base identity:** per path, `git rev-parse <sync-sha>:<path>` vs `origin/main:<path>` AND vs the SYNC-TIME main sha (take it from the round's divergence-probe `main=<sha>` note — the tip moves; matching only the current tip can false-FAIL a correct sync or false-PASS a mangled one that coincidentally matches later main).
2. **Scope containment:** every path must resolve to a SPECS entry (list in the worktree's `09-step-5.md`; `.claude/agent-memory` is a singleton), and NONE of the round's own deliverables may appear — a deliverable in the sync commit = the family-atomic dirty-skip failed (clobbered branch work).
3. **Subject anchor:** the literal phrase `sync workflow-surface specs from` must be in the subject (dirty-scan exclusion + Step 10d Guard 3 + verdict re-bind all key on it).
4. **No-lost-row shortcut:** `git log <merge-base>..<sync-sha>^ -- .claude/agent-memory` EMPTY ⇒ the parent state is itself main-historical, so nothing local-only could have been lost — settles the MEMORY.md no-lost-row check without row-by-row diffing. Non-empty ⇒ do the row diff (and question why the singleton wasn't marked dirty).

**Why:** #2327 R1 g2 — the brief itself warned that misreading the sync commit as authored payload makes every finding noise; all four checks resolved mechanically in ~6 git commands.
**How to apply:** any split-review sub-scope whose commit subject carries the anchor phrase, or any round where Step 5a ran. Related: [[two-sided-size-corridor-review]] (round-scoped grandfather probes).
