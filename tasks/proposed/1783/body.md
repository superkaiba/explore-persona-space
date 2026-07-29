---
title: 'workflow-fix: pre_reg audit misses #1092 bare registered-noun escape set'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1670f3363159
created_at: '2026-07-29T02:08:48Z'
has_clean_result: false
origin_prompt: 'clean-result-critic Lens 7 minor m2 (mechanizable: yes) on #1092:
  extend the audit''s pre-reg regex with a scoped bare registered-<noun> form — 7
  phrasings PASSed the gate across rounds 1-3 (verbatim prose quoted in the body Provenance
  section)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a clean-result-critic prose
follow-up (Lens 7 minor m2, `mechanizable: yes`) raised on task #1092
(emitting agent: clean-result-critic, `crossed-core-sae` round-1 review, 2026-07-29).

## Goal

Extend the pre_reg audit's bare-'registered <noun>' coverage (generalize toward the critic's `\bregistered\s+\w+` scoped-to-reader-facing-prose form, or enumerate the newly escaped head nouns), so the #1092 escape set is caught mechanically.

## Workflow gap

- **Bug observed:** Seven bare "registered <noun>" phrasings in #1092's promoted v4 body PASSed the audit's pre_reg check across rounds 1–3 — "the registered downgrade precondition", "the registered confidence intervals", "the registered trait-per-factor leg", "a registered subsample", "two registered operator-identity residual tests", "the registered monitoring-gap group-size curve", "the registered design" — the existing bare-'registered <noun>' branch (added #1419; head nouns extended twice since — layers?/rungs?/windows? for the #1586 escape at `1d7b685e18`, and the #1090 verdict-lattice set cuts?/paths?/clauses?/controls?/levers?/bars?/smokes?) does not cover these head nouns.
- **Why it is a workflow gap:** The check is head-noun-enumerated, so each new body's vocabulary escapes until another enumeration round is filed (#1419 → the #1090 filing → the #1586/#1638 filing → this is at least the THIRD escape-driven extension); the clean-result-critic flagged the class as mechanizable, and the audit ran PASS on the live #1092 body carrying all 7 phrasings this round (2026-07-29 mechanical pre-pass) — the recurring churn itself is evidence for the generalization arm over another enumeration.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'registered' scripts/audit_clean_results_body_discipline.py` → pre_reg branch present at lines 74–101 (presence hit READ in context: the branch is head-noun-enumerated with the #1419 lineage documented in its comments; it does NOT already cover the 7 phrasings — the audit ran PASS on the live #1092 body carrying them this round, so the gap is live, not landed); `git log --oneline --since='7 days ago' -- scripts/audit_clean_results_body_discipline.py` → 2 commits: `1d7b685e18` (#1638 head-noun extension layers?/rungs?/windows? — same mechanism, different nouns; does not cover this escape set) + `c9b2bbab82` (unrelated ban-family allowlist) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

Per the critic's mechanizable note: extend the audit's pre-reg regex toward
`\bregistered\s+\w+` scoped to reader-facing prose (Takeaways/Goal/Methodology/Results),
keeping the existing verb-register allowlist ("registered on HF", "registered in WandB",
"the model registered a clear ...") and adding needed allowlist entries
(e.g. "registered design marks"). Narrower alternative: add the escaped head nouns
(precondition(s)?, interval(s)?, leg(s)?, subsample(s)?, test(s)?, curve(s)?, design(s)?)
to the enumerated set. The planner picks generalization-vs-enumeration with the file
open; the #1419 comment lineage documents why the branch was enumerated (false-positive
control on the verb register) — re-measure any generalization against the promoted-body
corpus per the #1419 method before shipping.

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep before editing (`grep -rln 'pre_reg' scripts/ tests/`) and update the audit's pinned pattern tests alongside; list hits in the plan.
- The 7 live phrasings in #1092's body are the regression corpus — the fix should flag them on a corpus copy. Do NOT edit #1092's body from this task (the analyzer batches the wording pass into the #1773-instrumented follow-up round per the critic's disposition).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Grandfathered v3/v2 bodies must not be newly hard-FAILed (keep the audit's existing severity conventions).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 1670f3363159

Verbatim surfaced prose (clean-result-critic verdict, Lens 7, /tmp/issue-1092-cr-critique-r1.md):

> MINOR (m2, pre-existing rounds 1–3, previously PASSed by this gate): 7 bare "registered <noun>" phrasings ("the registered downgrade precondition", "the registered confidence intervals", "the registered trait-per-factor leg", "a registered subsample", "two registered operator-identity residual tests", "the registered monitoring-gap group-size curve", "the registered design") are the spec-text pre-registration-vocabulary class — a wording pass replacing "registered" with "the plan's" (the form the body already uses elsewhere) would clear it; plus 3 ± band notations ("17.9 ± 1.6, 10 draws" + "null ±0.03" in captions, "±0.15 transfer band" in prose) that describe null-band widths / plan thresholds, not credence intervals on estimates. Nothing new this round. mechanizable: yes — extend the audit's pre-reg regex with `\bregistered\s+\w+` scoped to Takeaways/Goal/Methodology/Results (allowlist "registered design marks" if desired).
