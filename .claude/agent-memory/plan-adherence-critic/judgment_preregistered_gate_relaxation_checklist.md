---
name: judgment-preregistered-gate-relaxation-checklist
description: How to adjudicate a diff that relaxes a pre-registered gate parameter (e.g. a feasibility-aware floor on an item-limited pilot arm) — the four checks and the severity key
metadata:
  type: feedback
---

When a diff RELAXES a pre-registered gate parameter (canonical shape: #2329 r9 —
rule-26 pilot `min_effective_draws_per_arm` dropped from the registered 51 to an
item-limited family's realized capacity 30, `allow_subresolution_pilot=True`),
adjudicate with four checks, not a reflex FAIL:

1. **Confinement grep incl. SIBLING gate call sites.** Grep the relaxing kwarg +
   helper across the whole script — the leak risk is the OTHER pilot phases
   (gate-6 `phase_pilot` in #2329) silently inheriting the relaxation. Also
   check whether the plan's registered floor is SCOPED to one gate (§7 gate 3-pre
   carried ≥51; the gate-6 registration did not) before flagging siblings.
2. **Shared-module unconditionality.** Verify the never-waived clause (rule-26(a)
   truncation) lives in the SHARED module and that module is byte-identical to
   origin/main across the branch (`git diff origin/main...HEAD -- <module>` empty),
   not just untouched by the one commit.
3. **Disclosure trio + direction.** Code docstring at the helper, per-family
   artifact fields (`floor_applied`/`floor_ceiling`/`sub_resolution`/
   `parse_fail_resolution_pct`), and the implementer marker §(d) with an explicit
   "belongs in the run's methodology" routing note STATING THE DIRECTION (which
   true-rate band now passes undetected). A realized gate report's `instrument`
   block doubles as grep-the-literal evidence for inherited constants (judge
   model, max_tokens, target) — quote it.
4. **Severity keyed on pipeline phase.** If the report phase has not run yet, a
   not-yet-in-methodology residual is a named FOLLOW-THROUGH obligation
   (mechanizable report-verifier check), not a revise-the-diff CONCERNS — there
   is nothing in the diff to fix. Untracked-but-not-ignored eval_results gate
   JSONs mid-run are the normal state (commit lands at P6/Step 8); flag the
   commit obligation, don't block on it.

**Why:** #2329 r9 (2026-08-17) — authorized single-call-site relaxation with
exemplary disclosure; the only real residuals were future-phase obligations.
**How to apply:** any diff touching `min_effective_draws`/floor/threshold kwargs
of a plan-registered gate, or passing an `allow_*` escape flag to shared gate
machinery. See also [[judgment-unimplementable-literal-substitute]].
