---
name: Reuse-fitness (a)-(g) mirror-set completeness check
description: When reviewing a workflow-fix plan that adds/edits an artifact-reuse fitness check letter, independently grep the full (a)-(g) call-site set — the documented mirror memory can be STALE (missed verify_plan.py c6 + its test)
type: feedback
---

When a `kind: infra` workflow-fix plan adds or edits a letter in the
trained-artifact reuse fitness checklist `(a)-(g)`, the Methodology lens
completeness check is: independently
`rg "\(a\)[-–]\(g\)" .claude/ CLAUDE.md scripts/ tests/` and confirm the
plan touches EVERY in-scope workflow-surface hit, not just the set the
`implementer/reference_reuse_fitness_mirror_set.md` memory records.

**Why:** that memory documented only a 4-FILE mirror set (planner step 5,
CLAUDE.md bullet, critic item 9, consistency-checker ×2) — but the live
call-site set is 6: it ALSO includes `scripts/verify_plan.py` `c6_reuse_fitness`
(regex `[a-g]`, `/7` denominator, `(a)–(g)` WARN strings) and
`tests/test_verify_plan.py` (asserts `"4/7"`, `"(a)–(g)"`/`"seven"`). A plan
that trusts the stale memory would leave an enforcement surface checking the
old contract. #737's plan caught this by grepping independently and grew the
memory to 6 files — the correct move.

**How to apply:** the hit set is dominated by OUT-OF-SCOPE frozen snapshots
(`tasks/**` historical plans, `.claude/plans/**`, `docs/methodology/**`) and
unrelated experiment smoke-gates (`leave_one_out_505/` `(a)-(f)`) — all
correctly excluded. The in-scope set is the live agents/rules/CLAUDE.md/
verify_plan.py/test surfaces. Don't REVISE for the frozen snapshots; DO
require the verify_plan.py heuristic + its test (they enumerate the letter
range and go stale). For the c6 WARN-only heuristic, the right scope is
letter-range BOOKKEEPING (`[a-h]`/`/8`/strings) with the `>=4` PASS threshold
UNCHANGED — a new letter is frequently N/A (most reuse is adapters), so raising
the threshold would false-WARN every valid adapter-only reuse plan. A NEW
mechanical fetchability verifier (parsing reuse claims out of free-form plan
prose) is a separable larger change, legitimately out of scope; prose-only
consistency-checker re-check is consistent with how `(e)` HF-resolution is
already enforced (manual list_repo_files in the BLOCK row, no verifier).
