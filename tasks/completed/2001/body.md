---
title: 'daily-fix: planner names measurement grain (unit of analysis'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9bb103ec6716
- daily-auto-filed
created_at: '2026-08-02T07:11:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1900''s leakage-predictor
  race ran per (prefix, query) row; Thomas wanted per-PREFIX (''RESULTS THAT RAN WERE
  PER QUERY WHICH IS TERRIBLE'') — full redo filed as #1979 (~15-25 GPU-h). No plan
  field names the DV''s unit of analysis or its aggregation; planner section 6 and
  the statistics lens are silent on measurement grain (only reuse-row COUNT grain
  and fold-disjointness group grain exist).'
workflow: v1
---
# daily-fix: planner names measurement grain (unit of analysis) per DV

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C5 (miner 6; session 75f66748, #1900 → redo filed and run as #1979).

## Goal
Make the measurement GRAIN — the unit of analysis (per-query vs per-prefix vs per-arm) plus the aggregation from raw rows to that unit — a NAMED per-DV plan field in planner §6 Evaluation, with a matching statistics-critic lens item that REVISEs a plan whose DV grain is unstated or mismatched to the Goal's construct.

## Workflow gap
- **Bug observed:** #1900's leakage-predictor race was computed per (prefix, query) row; Thomas wanted per-PREFIX: "basically i want to measure leakage on a per prefix level, averaged across queries" / "RESULTS THAT RAN WERE PER QUERY WHICH IS TERRIBLE". The assistant conceded the grain mismatch (the answer-similarity predictor was "asymmetric on two axes at once"); the full redo was filed as #1979 (~15–25 GPU-h, ~a day wall) — an entire run's spend lost to an unstated unit-of-analysis choice.
- **Why it is a workflow gap:** the plan pipeline names both mapping ARMS (prefix vs context), constructs, and on-distribution status per DV, but nowhere requires stating the unit each DV is computed AT and how rows aggregate to it — so a grain the user never intended survives planner + critics untouched.
- **Confidence:** high (gap re-verified by grep; incident transcript-grain, miner-inferred).
- verified-at-filing: `grep -n 'grain\|unit of analysis\|per-query\|per-prefix\|aggregat' .claude/agents/planner.md` → 3 hits, ALL about reuse-row row-COUNT grain (§10 Reproducibility Card :471-474 "counted realized grain for any reuse row whose row/line count feeds a plan floor ... (#1900)" — the #1910/7fed02515d fact-checker fix, a sizing check, NOT a DV unit-of-analysis field) and a smoke-slice corpus-grain line (:559); §6 Evaluation (:330-352) requires Measurement validity / Dual-DV / nulls / folds / mapping-baselines blocks — no unit-of-analysis or aggregation field. `grep -n 'grain\|unit of analysis\|aggregat' .claude/agents/statistics-critic.md` → 1 hit (:170, "DISJOINT ... at the group grain" — fold disjointness only). `grep -n 'grain\|unit of analysis' .claude/rules/critic-lens-reference.md` → no measurement-grain item (hits are compute-routing text). `git log --oneline --since='7 days ago' -- .claude/agents/planner.md .claude/agents/statistics-critic.md .claude/rules/critic-lens-reference.md` → 7fed02515d (#1910) covers only the reuse-row-count sense of "grain"; no landed DV-grain field (2026-08-02).

## Proposed change (refine in planning)
1. `.claude/agents/planner.md` §6 Evaluation: extend the Measurement-validity per-DV table with a required **Unit of analysis** column/field — the grain the DV is computed at (e.g. per-prefix), the raw-row grain it is derived from (e.g. per-(prefix, query)), and the aggregation between them (e.g. mean over queries within prefix) — plus one sentence tying the chosen grain to the Goal's construct. Named N/A escape for DVs with a single natural grain.
2. `.claude/agents/statistics-critic.md` (+ the v1 Statistics lens in `.claude/rules/critic-lens-reference.md`): new item — REVISE when a DV's unit of analysis is unstated, when the aggregation from raw rows to the stated unit is unstated, or when the stated grain does not match the Goal's construct (the #1900→#1979 shape: a per-prefix leakage question scored per (prefix, query) row).
3. Cross-reference the existing "both mapping arms" rule so grain and arm are stated together for mapping-line DVs.

## Scope / surfaces
- Primary target: `.claude/agents/planner.md, .claude/agents/statistics-critic.md, .claude/rules/critic-lens-reference.md`
- Also touch `.claude/rules/planner-section-reference.md` § 6 (the full §6 template lives there) so template and summary stay in sync; keep the reuse-row "grain count" wording distinct from the new DV field to avoid conflating the two senses.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- v2 lens→owner ledger: if the statistics-critic gains the item, update `.claude/rules/lens-coverage-map.md` (`--check-lens-coverage`).
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 9bb103ec6716
- workflow_fix_target: .claude/agents/planner.md, .claude/agents/statistics-critic.md, .claude/rules/critic-lens-reference.md
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C5.
