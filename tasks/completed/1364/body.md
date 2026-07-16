---
title: 'workflow-fix: analyzer CJK audit must cover judged install-instrument pools'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6ac5eaf8f15e
created_at: '2026-07-15T21:47:59Z'
has_clean_result: false
origin_prompt: 'interpretation-critic r1 prose follow-up on #1315: extend the analyzer
  language-intrusion audit to judged install-instrument pools (Qwen under non-CJK
  evals)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised by the interpretation-critic on task #1315 (round 1).

## Goal

Add to analyzer.md a language-intrusion audit duty: whenever the evaluated model is Qwen-family under a non-CJK eval, run the per-arm CJK scan over the JUDGED install-instrument pools (Tier-1/Tier-2/parity temp-1.0 completions joined with all_scores), reporting fired-overlap and zeroed/excluded-intrusion bounds next to any PASS/WARN adjudication — not only over greedy capture rollouts.

## Workflow gap

- **Bug observed:** #1315 r1: the analyzer's CJK audit covered only greedy capture rollouts (clean) while the judged install-instrument pools carried 16-23% CJK intrusion (persona parity 18/100 rows intruded / 13 fired, WildChat 16/100 / 11 fired, FT_pos Tier-2 23/200 / 20 fired) that flipped two fu4 parity PASSes under zeroed-intrusion bounds — invisible in the draft body; the interpretation-critic Lens 7 scan caught it post-hoc (the #1090 fu4 pattern recurring one experiment later).
- **Why it is a workflow gap:** NO workflow-surface file mandates a language-intrusion audit at all — the practice exists only as analyzer habit, applied this run to the geometry substrate (capture rollouts) but not to the judged pools the install adjudications rest on, so the recurrence class (#1090 fu4 → #1315) has no spec-level defense.
- **Confidence (emitter):** high (the critic's full-population scan is the evidence; both flipped adjudications reproduced with zeroed bounds)
- verified-at-filing: `grep -rlniE "CJK|language.intrusion|intrusion" .claude/ CLAUDE.md scripts/verify_task_body.py` → 0 workflow-surface hits (all matches are .claude/cache/ judge-state junk; per-target `grep -ciE "CJK|intrusion|language.*audit" .claude/agents/analyzer.md` → 0) — absence-of-duty claim, the 0-hit results ARE the evidence (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In .claude/agents/analyzer.md (near the raw-text / sample-plausibility duties):
+ Language-intrusion audit (Qwen-family model under a non-CJK eval): scan
+ BOTH (a) the capture/geometry substrate rollouts AND (b) every JUDGED
+ install-instrument pool (Tier-1/Tier-2/parity temp-1.0 completions joined
+ with all_scores) for CJK intrusion per arm; report intruded-row counts,
+ fired-overlap, and zeroed/excluded-intrusion bounds NEXT TO any PASS/WARN
+ install adjudication that rests on the pool (#1090 fu4, #1315: 16-23%
+ intrusion flipped two parity PASSes under zeroed bounds while the greedy
+ rollouts were clean).

## Scope / surfaces

- Primary target: `.claude/agents/analyzer.md`
- The planner may also consider `.claude/agents/interpretation-critic.md` Lens 7 (which caught it post-hoc) for a cross-reference; grep `grep -rln 'Lens 7\|raw-text sample' .claude/agents/` and list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md
- fingerprint: 6ac5eaf8f15e

(Origin: interpretation-critic round-1 prose follow-up on #1315, verbatim in the epm:interp-critique v1 marker + /tmp/interp-critique-1315-r1.md.)
