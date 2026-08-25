---
name: compose-recipe-lens-ref-replacements
description: Inlining the lens reference needs THREE run-it-yourself replacements — Lens 14 Step-0 block, Lens 14 composition-note literal, AND Lens 13's plan-fetch bash block (spec mandates only the first two).
metadata:
  type: feedback
---

When inlining `.claude/rules/clean-result-critic-lens-reference.md`
(lines `### Lens 1` .. end) into the Codex prompt, three spans instruct a
repo-script run and must be replaced with by-name references:

1. Lens 14 **Step 0 prerequisite** bash block (`task.py list-concerns <N>
   --open-only --json`) -> "read the OPEN-CONCERNS JSON envelope" (spec-
   mandated; the Step 4 no-residue grep catches a miss).
2. Lens 14 **Composition note** literal ("`task.py list-concerns --open-only
   --json` returns non-empty binding concerns") -> "the OPEN-CONCERNS JSON
   envelope contains non-empty binding concerns" (ALSO matches the no-residue
   grep — easy to miss because it is prose, not a bash block).
3. Lens 13 plan-fetch bash block (`plan_path=$(uv run python scripts/task.py
   find <N>)/plans/plan.md` + `cat`) -> "the absolute PLAN path at the top of
   this prompt". NOT caught by the no-residue grep (pattern only bans
   verifier/audit/list-concerns), but it contradicts the prompt's "Do not
   execute any repo script" rule and task.py cannot run in the Codex sandbox.

**Why:** verified 2026-08-25 composing #2378 r1 — the grep scan found exactly
hits 1+2; hit 3 was found by reading Lens 13. SPEC.md (inlined in full,
~107 KB) had ZERO banned-pattern hits, so no replacements needed there.

**How to apply:** assemble via a python splice script with `assert old in
text` before each `replace` (a silent no-op replace ships the stale
instruction); then run the spec's Step 4 semantic envelope + no-residue
guards. Practical sizes (2026-08-25): lens span ~72 KB, SPEC.md ~107 KB,
total prompt ~215 KB — normal, no elision needed.
