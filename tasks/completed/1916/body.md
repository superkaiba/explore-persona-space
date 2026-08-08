---
title: 'daily-fix: judge max_tokens floor 600 for JSON rubrics'
kind: infra
tags:
- wf-fix
- wf-fix-fp:85e9a904e86c
- daily-auto-filed
created_at: '2026-07-31T06:54:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): the rule-23 >=300 response-token
  floor measurably failed twice this week (#1739 at 400, #1769 at 300 with 12.5% truncation-censored
  draws, arm-asymmetric to 42.5%), each costing a full re-judge batch.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-3 P1, session d0fe5a10 / issue #1769).

## Goal

Raise the llm-judging rule-23 response-budget default for reason-then-score JSON rubrics from ≥ ~300 to ≥ ~600 tokens, and thread the higher default through the judge call sites that use such rubrics.

## Workflow gap

- **Bug observed:** the #1769 fu1 judge run (21,000 calls, reason-then-score JSON rubric) ran at the rule-23 floor `max_tokens=300` and truncation-censored 12.5% of hallucination draws (874/7000 parse errors, arm-asymmetric up to 42.5% on `hallucination/decode_only/a3`), forcing a full mt600 re-judge (~3h wall + a second batch spend). Second measured floor failure: #1739's wave at 400 tokens censored 5.4% of draws the same week.
- **Why it is a workflow gap:** rule 23's "≥ ~300" floor is now measurably insufficient for the multi-field JSON reason-then-score rubrics the project actually runs; each failure costs a full re-judge batch and risks the selection-artifact-mimicking censoring rule 23 exists to prevent.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n '300 response tokens' .claude/rules/llm-judging.md` → line 197 (`**≥ ~300 response tokens** (the #1090 recovery`) + the enforcement echo at line 457 — the 300 floor is still the stated default (2026-07-31 filing time). The #1769 drop numbers are quoted from that session's own drop-stat tool results (unverified hypothesis — verify at plan time: exact per-arm rates from `eval_results/issue_1769/` fu1 judge drop stats).

## Proposed change (candidate diff sketch — refine in planning)

Rule 23: state ≥ ~600 as the default for reason-then-score JSON (multi-field) rubrics, keeping ≥ ~300 for short single-rationale rubrics and keeping the post-run per-arm drop re-measure as the binding check; cite the two measured failures (#1739 at 400, #1769 at 300). Optionally raise the `graded_judge.judge_graded` recommended caller value in the same pass.

## Scope / surfaces

- Primary target: `.claude/rules/llm-judging.md` (rule 23 + its enforcement bullet)
- Secondary: the rule-23 cross-references in CLAUDE.md § Measurement validity and `experiment-guidelines.md` item 9 if they quote the floor (grep `~300` across the workflow surface and update every hit).

## Constraints / invariants

- The floor stays a floor; the post-resize per-arm drop re-measure stays the binding check (rule 23's #1739 clause).
- Cache caveat unchanged (rubric-level cache does not key on max_tokens — re-judge guidance stays).

## Provenance

- fingerprint: 85e9a904e86c

- workflow_fix_target: .claude/rules/llm-judging.md
- origin: /daily 2026-07-30 miner-3 P1 (transcript d0fe5a10, issue #1769 fu1 judge round)
