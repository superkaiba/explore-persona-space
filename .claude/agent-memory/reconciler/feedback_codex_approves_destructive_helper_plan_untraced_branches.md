---
name: Codex APPROVEs destructive-helper infra plan without tracing mutating branches
description: On workflow-fix plans for git/file-mutating helpers, Codex APPROVEd citing test coverage the plan's own test list contradicts; verify the destructive-branch triad yourself
type: feedback
---

Rule: when reconciling a critic split on a PLAN for a helper that deletes/moves
files or mutates shared git state (root-sync, sweep, janitor, cleanup scripts),
do NOT credit an APPROVE's "the planned tests exercise the key failure shapes"
claim — trace the triad yourself against the plan text:

1. **TOCTOU between identity-check and destructive action** — a hash/stat taken
   at enumeration time licensing an `os.remove`/move executed later violates a
   "never delete non-identical data" invariant when the file-writers take no
   lock (locks over git/task writers do NOT cover raw file writes).
2. **Failure-path disposition of pre-mutations** — a pre-sweep licensed by "the
   pull/checkout rematerializes it" is falsified on every abort path; check the
   acceptance-criteria failure contract names what happens to already-swept
   files when the main operation aborts (often the COMMON outcome).
3. **Every mutating branch has a test** — enumerate the design's mutating
   branches (error-driven/stderr-parse fallbacks that feed deletions, stale-husk
   auto-aborts, threshold-gated recovery arms) and check each appears in the
   plan's test list; a parser-feeds-destructive-action branch with no fixture
   test is the success-path-only class and an enumerated REVISE ground.

**Why:** #904 r1 (methodology): Claude REVISE with 3 Must-Fixes (all verified:
§4.3 hash-then-rm TOCTOU vs the §4 hard constraint; abort leaves the tree
worse-than-found, acceptance criterion 4 silent; fallback sweep + husk-abort
absent from §5 tests 1-9); Codex APPROVEd asserting "planned fixture tests
exercise the key failure shapes" — falsified by the §5 list. Upheld all three,
REVISE. Sibling of feedback_plan_verbatim_text_vs_plan_binding_mustfix.md (the
§4 sketch itself encoded the invariant violation, so implementer fidelity ships
it — not recoverable-by-implementer).

**How to apply:** any critic-lens reconcile where the artifact is an infra plan
for a destructive/mutating helper and one side APPROVEs on a safety/coverage
summary — run the triad against §Design + §Tests before crediting either side.
