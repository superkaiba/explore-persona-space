---
title: 'daily-held: #825 Takeaway refuted by guarded rerun'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-26T07:07:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 3): #825 sits at awaiting_promotion
  carrying the bolded Takeaway that the user''s next turn is linearly unpredictable,
  which its own settle-battery round refuted: with the fit module''s registered dof-cap
  mitigation engaged all four M user cells flip from negative to positive and the
  committed default is the unguarded arm.'
workflow: v1
---
## Why this needs you

Filed by the `/daily` 2026-07-25 problem sweep as a **route-3 judgment call** under the
"Scientific-meaning changes" carve-out: correcting a promoted clean-result's Takeaway
changes how a result is interpreted, and the promotion classification is user-only.

**#825 is at `awaiting_promotion` with `has_clean_result: true` and carries a Takeaway
its own follow-up round refuted yesterday.**

## What was found

Session `63122023` (2026-07-25, Phase 0 of the settle battery) established that the
λ-selection guard the fit module itself documents for the `n_train < D` regime
(`GCV_DOF_CAP`) **defaulted OFF on every committed #825 fit**. With the guard engaged:

| cells | unguarded (committed) | guarded |
|---|---|---|
| M user (×4) | −1.43 … −1.65 | +0.19 … +0.25 |
| M assistant | +0.076 / −0.461 | 0.588 / 0.493 |

Every user cell flips sign. The result reproduced under two independent selectors.

**#825's live body still reads (verbatim, verified in the task body at compose time):**

> the user's next turn is linearly unpredictable (ridge R² negative for both real
> human turns and model-generated user turns) and only weakly nonlinearly predictable
> (MLP 0.19–0.23)

That claim is an artifact of the unguarded selector.

**The blast radius is wider than one bullet.** The session also flagged that the 12
banked `role-map-comparison` ROLE-GAP deltas are all computed unguarded, and that a
comparison table given to you in chat two days earlier (−0.77 / −1.84 etc.) was all
unguarded numbers. Assistant verbatim: *"The table I gave you two days ago … was all
unguarded numbers — superseded"* and *"the #825 takeaway … is wrong"*. The session
deliberately left the body unrevised — *"those are your text"* — and filed nothing.

## Verified at filing (2026-07-25)

- `task.py view 825 --json` → `status: awaiting_promotion`, `has_clean_result: True`,
  `classification: None` (unpromoted). Takeaway line quoted verbatim from that body.
- `grep -rn 'GCV_DOF_CAP' src scripts` → `scripts/issue825_selector_audit.py:103-104`
  labels the two arms in the audit's own words:
  `"gcv_unguarded": "GCV_DOF_CAP=None, lambda_selection=gcv (committed #825 default)"`
  and `"gcv_guarded": "GCV_DOF_CAP=<cap>, … (registered mitigation)"`. So
  "the committed default is unguarded, and the guard is the module's own registered
  mitigation" is the code's description, not an inference.
- The settle-battery outputs landed: `eval_results/issue_825/trackm_settle_battery/`
  (committed 2026-07-25 11:12).

## The decisions that are yours

1. **#825's body** — rewrite the Takeaway (and the two-turn / MLP-recovery sections
   that rest on it) against the guarded numbers, or promote it `not-useful`, or
   something else. An automated rewrite of a headline scientific claim is exactly what
   the route-3 carve-out reserves for you.
2. **The 12 banked `role-map-comparison` ROLE-GAP deltas** — re-run guarded, or mark
   them superseded. They are inputs other tasks may already have consumed.
3. **Whether `GCV_DOF_CAP` should default ON** project-wide (or fail loud when
   `n_train < D` and the guard is off) rather than being an opt-in the next reuse can
   silently miss. This is an analysis-semantics default, not a workflow tweak.

## Related work already routed tonight

- The *workflow* half — "an inline round that refutes a claim in a promoted body must
  apply the correction or file a task in the same turn", plus an n-versus-d guard-rail
  on ridge fits — is filed separately as a route-2 workflow-fix task from this same
  sweep. That task deliberately does NOT touch #825's body.
- `.claude/rules/artifact-reuse.md` clause (l) (validity-domain transfer) is the
  existing rule closest to this incident: a reused instrument's registered mitigation
  must be engaged when the new data regime crosses its validity boundary. Worth
  deciding whether clause (l) should require the mitigation be shown ENGAGED in the run
  config rather than merely available.
