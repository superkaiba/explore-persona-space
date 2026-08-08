---
title: 'update #1345 Takeaway bullet 5: slot-position reads re-fit by lambda audit
  (deficit survives); every-layer statements stay ambient-only'
kind: infra
tags: []
created_at: '2026-07-31T07:22:44Z'
has_clean_result: false
origin_prompt: '#1887 P3 fold agent: #1345 Takeaway bullet 5 ''were not re-tested''
  stale — the 67-cell audit re-fit all six slot cells + free-form layer-19 cells;
  conclusions survive.'
workflow: v1
---
## Overview / Motivation

Auto-filed by #1887 (lambda-selection hardening + audit) per the refuted-body duty: the #1887 67-cell lambda audit makes one clause of a bolded #1345 Takeaway stale. Takeaway-touching corrections are filed, never edited directly.

## Goal

Update #1345's Takeaway bullet 5 second sentence to reflect that the slot-position story reads HAVE now been re-fit under the corrected estimator (conclusion survives), while every-layer statements remain ambient-basis-only.

## Refuting evidence

- verified-at-filing: `grep -c 'were not re-tested' tasks/awaiting_promotion/1345/body.md` → 1 (2026-07-31, this session). Corrections table: `eval_results/issue_1345/lambda_audit_1887/corrections_table.{json,md}` (67 rows, replay gate PASS on all; branch issue-1887, merging via #1887).
- Takeaway bullet 5, second sentence (verbatim): "Every per-slot and per-layer story read below was fit in the ambient basis, so the slot-position and every-layer statements are estimator-limited and were not re-tested." — "were not re-tested" is now STALE: the audit re-fit all six slot cells (inner-group-CV +0.141…+0.508; pre-answer best +0.508, anchor +0.435; matched chat +0.645 — the no-slot-reaches-chat deficit and the pre-answer-best ranking SURVIVE) and the free-form layer-19 cells (instruct −0.754 → +0.295 inner-CV; CJK-excluded −0.652 → +0.294).

## Proposed change (refine in planning)

Replace the stale clause with: "…the slot-position reads were since re-fit by the lambda audit (inner-group-CV +0.37 to +0.51 vs matched chat +0.64: deficit and pre-answer-best ranking survive); the every-layer statements remain ambient-basis-only." Then verify_task_body.py --issue 1345 must PASS. Body prose corrections (non-Takeaway) were already applied by #1887 P3 (commits 5027e28f86 + 1ad0664a99); do not duplicate them.

## Scope / surfaces

- tasks/awaiting_promotion/1345/body.md (Takeaways bullet 5 only) via task.py set-body. No eval JSONs. Task stays at awaiting_promotion; classification untouched (user-only).

## Provenance

Surfaced by #1887 P3 fold agent report (2026-07-31); refuting artifacts as above.
