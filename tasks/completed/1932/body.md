---
title: 'correct #1310 Takeaway bullet 3 + title qualifier per nd-estimator/lambda
  audits (per-turn deflation was a lambda-selection artifact)'
kind: infra
tags: []
created_at: '2026-07-31T07:22:37Z'
has_clean_result: false
origin_prompt: '#1887 P3 fold agent: #1310 Takeaway bullet 3 ''real at the per-turn
  grain'' refuted (all four onpolicy_instruct cells sign-flip under inner-group-CV);
  H1 qualifier ''once each scene is aggregated to one point'' materially weakened.'
workflow: v1
---
## Overview / Motivation

Auto-filed by #1887 (lambda-selection hardening + audit) per the refuted-body duty: the #1887/nd-estimator audits refute one bolded Takeaway and materially weaken the H1 title qualifier of #1310 (awaiting_promotion). Takeaway-touching corrections are filed, never edited directly.

## Goal

Correct #1310's Takeaway bullet 3 and H1 title qualifier to match the audited estimator-corrected reads, preserving everything the audit confirmed.

## Refuting evidence

- verified-at-filing: `grep -c 'real at the per-turn grain' tasks/awaiting_promotion/1310/body.md` → 1; `grep -c 'once each scene is aggregated to one point' tasks/awaiting_promotion/1310/body.md` → 2 (2026-07-31, this session). Corrections table: `eval_results/issue_1310/nd_estimator_audit/corrections_table.md` (on main, commits fc8acf205d + 6eb700738d) + `eval_results/issue_1345/lambda_audit_1887/` (branch issue-1887, merging via #1887).
- Takeaway bullet 3 (verbatim): "Round 1's per-turn instruct anti-prediction (−0.10 to −0.19, swap inverted) was a within-scene near-duplicate-context fold artifact — real at the per-turn grain, not a property of the character map." — "real at the per-turn grain" is REFUTED: cells onpolicy_instruct_{Wren,HELIOS,Dana,Vex} published −0.1783/−0.0996/−0.1888/−0.1762 → inner-group-CV +0.2760/+0.3134/+0.2512/+0.2366 (published-deflated; all four sign-flip). Caveat: the swap-contrast cells ("swap inverted") were NOT re-audited.
- H1 title qualifier "once each scene is aggregated to one point" is materially weakened: aggregation is not required for per-turn positivity — the published capped-GCV λ selection was the deflator (inner-CV selects λ 3,162–10,000 vs published λ=100).

## Proposed change (refine in planning)

1. Takeaway bullet 3 → "Round 1's per-turn instruct anti-prediction (−0.10 to −0.19) was a λ-selection artifact conditioned by within-scene near-duplicate contexts — inner-group-CV reads all four cells positive (+0.24 to +0.31); the swap inversion is not re-audited."
2. H1 title (+ frontmatter title via task.py set-title): drop or soften the "once each scene is aggregated to one point" qualifier consistent with the corrected per-turn reads. Optionally note in bullet 6's method caveat that the capped selector also deflates per-turn cells.
3. Re-run verify_task_body.py --issue 1310 (must PASS). Body prose corrections (non-Takeaway) were already applied by #1887 P3 (commit 9932469695); do not duplicate them.

## Scope / surfaces

- tasks/awaiting_promotion/1310/body.md (Takeaways + H1/title only) via task.py set-body + set-title. No eval JSONs. Task stays at awaiting_promotion; classification untouched (user-only).

## Provenance

Surfaced by #1887 P3 fold agent report (2026-07-31); refuting artifacts as above.
