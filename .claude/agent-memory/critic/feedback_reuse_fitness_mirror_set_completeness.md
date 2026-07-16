---
name: Reuse-fitness mirror-set completeness check
description: When reviewing a workflow-fix plan that adds/edits an artifact-reuse fitness check letter, independently grep the full lettered call-site set (currently (a)-(k)) — the documented mirror memory can be STALE (it once missed verify_plan.py c6 + its test)
type: feedback
---

When a `kind: infra` workflow-fix plan adds or edits a letter in the
trained-artifact reuse fitness checklist (the lettered set, currently
(a)-(k)), the Methodology lens completeness check is: independently
`rg "\(a\)[-–—]\([a-z]\)" .claude/ CLAUDE.md scripts/ tests/` and confirm the
plan touches EVERY in-scope workflow-surface hit, not just the set the
`implementer/reference_reuse_fitness_mirror_set.md` memory records.

**Why:** that memory once documented only a 4-FILE mirror set (planner step 5,
CLAUDE.md bullet, critic item 9, consistency-checker ×2) — but the live
call-site set also included `scripts/verify_plan.py` `c6_reuse_fitness` (the
letter-range regex class, the `/N` denominator, the en-dash range WARN
strings) and `tests/test_verify_plan.py` (the coupled `"4/N"` + range /
count-word assertions). A plan that trusts a stale memory leaves an
enforcement surface checking the old contract. #737's plan caught this by
grepping independently and growing the memory — the correct move (as of #871
the memory lists 15 sites, re-synced to 20 at #941; grep anyway).

**How to apply:** the hit set is dominated by OUT-OF-SCOPE frozen snapshots
(`tasks/**` historical plans, `.claude/plans/**`, `docs/methodology/**`) and
unrelated experiment smoke-gates (`leave_one_out_505/` uses its own lettered
gates) — all correctly excluded. The in-scope set is the live agents / rules /
CLAUDE.md / verify_plan.py / test surfaces plus the agent memories that name
the range. Don't REVISE for the frozen snapshots; DO require the
verify_plan.py heuristic + its test (they enumerate the letter range and go
stale). For the c6 WARN-only heuristic, the right scope is letter-range
BOOKKEEPING (regex class / denominator / count words) with the `>=4` PASS
threshold UNCHANGED — a new letter is frequently N/A (most reuse is
adapters), so raising the threshold would false-WARN every valid adapter-only
reuse plan. A NEW mechanical fetchability verifier (parsing reuse claims out
of free-form plan prose) is a separable larger change, legitimately out of
scope; prose-only consistency-checker re-check is consistent with how (e)
HF-resolution is already enforced (manual list_repo_files in the BLOCK row,
no verifier).

**#941 ((j) pairwise provenance coherence) nuances:** (1) the #871 range-free
remedy-line design held — a data-side letter routes through the "other than
(i) → regenerate" branch with NO remedy edit; check any new letter against
that branch before demanding remedy rewording. (2) Provenance-DATE predicates
(`get_paths_info(expand=True)` `last_commit.date`, input ≤ capture) compare
UPLOAD ordering, not GENERATION ordering — a capture made from a stale local
copy but uploaded after the input's regeneration false-negatives; that is
acceptable-with-note (plan-time floor + runtime parity-assert backstop), not
a REVISE. False positives (unrelated commit bumps) err conservative and are
absorbed by a "confirm why regenerated" remedy step. (3) Cross-repo pairs
(mix↔adapter spans dataset+model repos) need one call per repo — a wording
concern, not a design flaw.
