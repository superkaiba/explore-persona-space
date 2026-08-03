---
title: 'daily-fix: down-width probe for stalled multi-arm dispatch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:90229b0efdf1
- daily-auto-filed
created_at: '2026-07-23T07:03:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): #1112''s 1x-runnable arm
  sat ~14 h behind a coupled 4x/8x provision during a GCP drought; the 1x shape had
  stock all along and finished in ~55 min once split out'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). #1112's dispatch coupled a 1×-runnable arm (Arm A LoRA ladders) and a 4×-needing arm (Arm B2 ZeRO-3 full-FT) behind one 4×/8× provision. During a ~14 h GCP A100 drought (26 create attempts, both zones), the narrower 1× shape was never probed — when the session finally split out an Arm-A-only 1× dispatch at 21:09Z it provisioned IMMEDIATELY ("The 1× A100-80 shape had stock") and finished in ~55 min. ~14 h of wall-clock lost on cells that could have run all day.

## Goal

Plan/dispatch guidance (the down-going sibling of the #1121 wide-rung walk): when a stalled multi-arm dispatch contains arms with strictly NARROWER GPU requirements, the narrower shape is probed early (down-width split) instead of holding all arms behind the widest arm's provision.

## Workflow gap

- **Bug observed:** 24ae2158 (#1112), 06:52Z → 21:09Z: 14 h behind a 4×/8× queue while 1× had stock; two user interventions ("what's the bottleneck here").
- **Why it is a workflow gap:** `plan-compute-sizing.md` and planner §9 cover width RIGHT-SIZING per phase and the wide-rung capacity walk, but nothing states the decomposition duty when a multi-arm dispatch STALLS and a subset of arms is narrower-runnable.
- **Confidence:** medium-high.
- verified-at-filing: `grep -c 'down-width\|narrower shape' .claude/rules/plan-compute-sizing.md .claude/agents/planner.md` → 0 hits in both (absence claim), 2026-07-23 UTC.

## Proposed change (refine in planning)

One clause in `.claude/rules/plan-compute-sizing.md` (+ a pointer in `planner.md` §9): a multi-arm plan names each arm's MINIMUM runnable width; on a sustained capacity stall (≥ ~1 h queued/stocked-out), the dispatcher splits out and probes the narrowest-runnable arms before continuing to hold the coupled wide provision.

## Scope / surfaces

- Primary targets: `.claude/rules/plan-compute-sizing.md`, `.claude/agents/planner.md` (§9).

## Constraints / invariants

- The #1121 wide-first walk for shardable work is unchanged (this is the STALL-time down-going complement); the saturate-or-downsize idle-width protections unchanged. Recursion guard applies.

## Provenance

- fingerprint: 90229b0efdf1

- workflow_fix_target: .claude/rules/plan-compute-sizing.md
