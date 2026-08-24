---
name: lens13-plan-fetch-patch
description: When inlining the fifteen-lens reference, also replace Lens 13's plan-fetch bash block (task.py find) with a pointer to the prompt's absolute PLAN path — the spec mandates only the Lens 14 ledger-block replacement.
metadata:
  type: feedback
---

When copying `clean-result-critic-lens-reference.md` Lens 1-15 into the
Codex prompt, Lens 13 ("Planned-vs-actual coverage") opens with a bash
block that tells the reader to derive the plan path via
`plan_path=$(uv run python scripts/task.py find <N>)/plans/plan.md` and
`cat` it. Replace that block with a one-line pointer to the absolute
PLAN path already in the prompt header ("existence-checked at compose
time; do not run any repo script to re-derive it").

**Why:** the agent spec's Step 2 mandates only the Lens 14 ledger-fetch
replacement, and the Step 4 no-residue greps do not match `task.py find`
— so an unpatched Lens 13 block ships a run-a-repo-script instruction
into a read-only sandbox that contradicts the prompt's own "Do not
execute any repo script" rule, inviting a spurious `BLOCKED — could not
run task.py` on a load-bearing lens (Lens 13 BLOCKED forces
needs_targeted_fix + data-access-blocked). Applied at #2476 r2
(2026-08-23); surfaced as a workflow-fix prose follow-up the same round.

**How to apply:** every markdown-branch compose, right after the Lens 14
patch — assert exactly one replacement (`raw.count(old) == 1`) so lens
reference drift fails loud instead of silently shipping the stale block.
Related: [[delta-rounds-beyond-r3]] for round-scope composition.
