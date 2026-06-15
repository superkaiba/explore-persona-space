---
title: 'Sidequest: does #611''s bare-role-header leakage split generalize beyond pirate↔villain
  (6-persona style+intent panel + on-policy emission)?'
kind: experiment
tags: []
created_at: '2026-06-15T21:27:47Z'
has_clean_result: false
parent_id: 611
---
# Sidequest: does #611's bare-role-header leakage split generalize beyond pirate↔villain?

**Tracking card (sidequest).** Parked for later, out of the active queue. This is the standalone card for a **same-issue follow-up on [#611](https://eps.superkaiba.com/tasks/611)** — the scope is already filed as `epm:followup-scope v1` (`followup_label: more-personas-panel`) on #611. When picked up, execute via `/issue 611` (the same-issue follow-up loop), NOT as a standalone child run.

## Question
#611 showed (teacher-forced, bare wording, pirate↔villain only): a bare role header leaks **less** marker mass to the other persona (install-onward) but **more** to the default assistant (every checkpoint). Does that split hold across more personas, and does it survive on-policy emission?

## What changes vs #611 (locked with Thomas)
1. **Persona panel → ~6-persona bystander panel spanning style + intent** (style near pirate: chef/coach/poet; intent near villain: tyrant/con-artist). Each trained as a marker source; leakage probed across held-out panel personas + default. Reported per-persona and per-type, never pooled. Also re-tests #464's style-installs / intent-doesn't asymmetry at panel scale.
2. **Add an on-policy emission count** alongside the teacher-forced log-prob + EOS-margin reads (free-decode, count actual ` ※` at wrong-persona + default slots). Converts #611's binding teacher-forced caveat into a behavioral claim — mainly bites on the default-assistant slot.

## Inherited unchanged (single-variable discipline)
Bare-word recipe (`system_minimal` vs `role_bare`), lr=5e-6, LoRA r=32/α=64, marker ` ※` id 83399, marker-only loss, step grid {18,30,60,120}, 5 seeds, four-float logit capture.

## Planner must handle
Contrastive-negative disjointness (panel ∩ realized sources = ∅, the #527/#538 trap); on-policy positives; recipe comparability to #611.

## Hypotheses
- H1: the split generalizes across the ≥6-persona panel.
- H2: the role advantage is carried by style personas; intent personas behave like #464's villain (weak/absent install → weaker/reversed split).
- H3: on-policy default-slot emission is nonzero for style role-header cells, ~0 for intent cells.

**Cost:** needs-gpu. **Routing:** `/issue 611` same-issue follow-up; artifacts under `eval_results/issue_611/more-personas-panel/`.
