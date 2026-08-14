---
title: 'audit_clean_results_body_discipline: extend pre_reg noun grammar (recompute/sensitivity/comparison)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T13:15:05Z'
has_clean_result: false
origin_prompt: 'clean-result-critic Lens 7 on #2254: four ''registered <noun>'' pre-registration
  phrasings (recompute/sensitivity/comparison) passed the mechanical audit pre-pass'
workflow: v1
---
## Goal
Extend the `pre_reg` banned-prose noun grammar in `scripts/audit_clean_results_body_discipline.py` (Lens 7 pre-registration-claim check) so it catches "registered <noun>" phrasings where `<noun>` is `recompute` / `sensitivity` / `comparison` (and near-synonyms). On #2254 four bare "registered recompute / registered sensitivity / registered comparison" mentions passed the audit's mechanical pre-pass and were only caught by the clean-result-critic Lens 7 human read.

## Scope
- Audit `audit_clean_results_body_discipline.py` for the current `pre_reg` noun list; add the missing analysis-noun variants.
- Add a regression fixture body reproducing the #2254 phrasings; confirm it FAILs the audit before the fix and after adding the variants.
- Do NOT broaden into false positives (a legitimate "registered the pod" / "registered session" is not a pre-registration claim) — anchor on the analysis-artifact noun set.

Provenance: surfaced by clean-result-critic Lens 7 on #2254 (workflow-fix-candidate, prose follow-up).
