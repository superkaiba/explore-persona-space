---
title: 'daily-held: matched-n control for cross-cell R2 comparisons'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-25T06:51:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 3): The 825 Track-M two-turn-collapse
  number was presented as a data property when the user-requested audit showed about
  70 percent of the collapse is fitting machinery and n-sensitivity - 0.482 at n=700,
  0.258 at n=2000, 0.673 at n=5000 on Track S matched-n curves'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 1a81f094, task #825). HELD as needs-human — carve-out: **scientific-meaning change** (alters how results are evaluated/interpreted; a standing measurement rule of this kind has always been a user-decided rule).

## The held item

The #825 Track-M "two-turn collapse" R² number was presented in an earlier round as a property of the data; the user flagged confusion ("can you run an audit of why it failed here? it's a bit confusing") and the 2026-07-24 inline audit (committed 010d4b7bf1; epm:progress v304 + epm:free-analysis-followup-run v9 on #825) found ~70% of the collapse is fitting machinery / n-sensitivity — matched-n Track S curve: 0.482 @ n=700 → 0.258 ± 0.039 @ n=2,000 → 0.673 @ n=5,000.

## Proposed rule (needs your ok)

Add to the measurement-validity surface (CLAUDE.md Measurement bullet / `.claude/rules/experiment-guidelines.md` + statistics-critic lens): any cross-corpus / cross-cell held-out-R² (or ρ) comparison where cells differ in n MUST include a matched-n (subsample) control before a lower value is narrated as a property of the corpus/condition; without it the comparison is confounded by estimator n-sensitivity.

## Suggested action

Approve → file the wording as a standing rule (one /issue session wires planner/critic surfaces); reject → note why so the pattern is at least documented in gotchas.
