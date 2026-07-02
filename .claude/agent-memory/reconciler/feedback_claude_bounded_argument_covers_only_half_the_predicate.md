---
name: Claude "bounded/non-silent" APPROVE covers only half a bundled predicate
description: When Claude APPROVEs an optional-verification finding as "bounded, non-silent", check whether its bound covers EVERY invariant the optional test bundles — a merge-conflict backstop bounds dedup but NOT recursion (unbounded fan-out has no backstop).
type: feedback
---

When the disputed finding is "the plan makes a load-bearing predicate's
EXECUTABLE test OPTIONAL" and Claude APPROVEs it as Real-but-non-blocking
("bounded, non-silent — caught at runtime"), do NOT accept the bound at face
value. Verify the bound covers EVERY invariant the optional test bundles
together. A single §6 acceptance line often bundles two predicates whose
failure modes differ wildly.

**Why:** #678 r1 (infra workflow-surface migration). §6 made ONE optional test
cover BOTH the dedup predicate AND the recursion-guard predicate. Claude's
"bounded, non-silent" bound was the §7 merge-conflict backstop — which genuinely
bounds the DEDUP failure (a missed dedup files a duplicate task; the second
/issue Step 10d merge conflicts + requeues). But that backstop does NOT exist for
the RECURSION predicate: a mis-implemented guard that fails to recognize a
workflow-fix session spawns sessions recursively with NO natural stop and NO cost
cap — the plan itself named this as "unbounded fan-out" (Q4). Shipping the one
invariant whose failure is genuinely unbounded with zero executable verification,
on a prose-only branch the plan explicitly permitted, is a verification-
sufficiency gap. Codex's REVISE (make the predicate executable + non-optional)
was right; Claude's APPROVE under-weighted it by averaging the bound across both
predicates. REVISE.

**How to apply:**
1. When Claude's APPROVE rationale is "bounded / non-silent / caught at runtime,"
   identify the runtime backstop it names (merge conflict, loud failure, gate).
2. Enumerate EVERY invariant the disputed optional test/gate covers. §6 lines
   that say "this test pins (1) ... (2) ..." are the tell — two predicates, one
   optional test.
3. For EACH invariant ask: does the named backstop actually catch THIS one's
   failure? A merge-conflict backstop catches duplicate FILES, not recursive
   SPAWNS. A "fails loudly" frozen-file `name:` catches a stale spawn in THAT
   file, not in a sibling file.
4. If the bound covers some invariants but not the unbounded/silent one → the
   finding is Real-blocking for the uncovered invariant → REVISE.

**Sibling pattern (same incident, MF#2):** Claude bounded a manual-grep
acceptance gate ("frozen agent file's name: fails loudly") — but that protects
only the ONE known file; a surviving executable `Agent(subagent_type="X", ...)`
instruction in any OTHER file passes a manual-inspection gate. When a reviewer
offers a concrete `rg ... returns zero` assertion to replace manual inspection
of grep hits, and the migration's stated conclusion is "old path fully retired,"
the mechanical assertion IS conclusion-changing (proves the conclusion vs asserts
it). Prefer the mechanical gate.
