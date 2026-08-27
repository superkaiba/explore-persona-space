---
name: brief-artifact-traps-verify-at-compose
description: When the orchestrator brief names artifact-level traps for the composed prompt, verify each stated fact against the artifacts at compose time and inline the corrected precise statement; reviewing-status artifacts live worktree-only — pass absolute worktree paths.
metadata:
  type: feedback
---

When a brief directs "surface these artifact-level traps in the composed
prompt", do NOT copy the brief's wording verbatim — verify each stated
fact against the artifacts first and inline the corrected, precise form:

1. Recount/reprobe every stated count, field location, and file set
   (#2546 r1: brief said "every cell JSON carries ceiling_status:
   missing_reliability_capture" — reality was 161/170, top-level field,
   9 cells lack the key; brief said "ladder 10 units" — the ladder/ dir
   holds 12 files, 2 of them operator_comparison__*.json non-ladder
   units worth an explicit exclusion line so Codex's own recount does
   not fold them in).
2. Verify the named source lines exist (sed the script lines; quote
   what they assign) before asserting "hardcoded literal at file:line"
   to Codex.
3. A task at status `reviewing` has its eval_results/figures ONLY in
   the issue worktree (merge to main happens at awaiting_promotion) —
   the spec's "nothing this prompt references lives in a worktree"
   claim does not cover brief-directed artifact checks. Pass ABSOLUTE
   worktree paths, existence-checked at compose time, with a header
   note explaining why; body/plan/lens-reference stay canonical-main;
   figures stay body-pinned-blob per #922.
4. Ledger independence (#2326): snapshot list-concerns ONCE, record the
   UTC timestamp in the prompt, and state that the parallel Claude
   critic's post-snapshot rows are excluded by design (absence is not a
   finding). Also state the binding-vs-NIT split (e.g. 25 rows = 14
   binding CONCERNs + 11 NITs) so Codex does not flag the
   envelope-count vs verifier-line "N binding" difference as a
   mismatch.

**Why:** an imprecise trap statement generates the exact false blockers
the trap was meant to prevent (a "every cell" claim Codex refutes on
the 9 keyless cells; a 12-file ladder recount read as an off-by-two).

**How to apply:** every compose whose brief carries artifact facts;
pairs with [[compose-recipe-lens-ref-replacements]] (splice mechanics)
and [[adjudicated-concern-count-override]] (envelope-authoritative
Lens 14 block).
