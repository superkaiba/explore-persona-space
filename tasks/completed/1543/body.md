---
title: 'workflow-fix: staging probe per source-family+consumer'
kind: infra
tags:
- wf-fix
- wf-fix-fp:75257e1efcfb
- daily-auto-filed
created_at: '2026-07-19T07:07:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): Two #1481 staged-layout
  crashes landed the same day (datagen_topup vs datagen 404; stage_hub_prefix reread
  crash) because the single global (h)(iv) staging probe missed a second source-family
  and a second consumer (c3-P5 + c3-P12).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P5 + c3-P12). Route-2 filing.

## Goal

Make the artifact-reuse (h)(iv) 1-file staging probe run PER reused
source-family AND per staged consumer, not once globally per plan — so a
second reused source-family or a second consumer re-reading the staged layout
is probed before production.

## Workflow gap

- **Bug observed:** TWO #1481 staged-layout crashes landed the same day —
  `datagen_topup/` vs `datagen/` 404 at 06:14, and a `hub.stage_hub_prefix`
  reread-layout crash at 20:45.
- **Why it is a workflow gap:** artifact-reuse.md (h)(iv) requirement (3)
  ("BEFORE any production run, a 1-file staging probe + consumer-open") is
  phrased around "the artifact's ENTRY file" — a single probe. When a plan
  reuses MULTIPLE source-families staged into MULTIPLE consumers, one global
  probe leaves the others unprobed, so a mismatch in a second family/consumer
  crashes at run time.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -cn 'per reused source-family\|per staged consumer\|per source-family' .claude/rules/artifact-reuse.md` → 0 hits; the (h)(iv) requirement (3) span (line ~223) prescribes ONE "1-file staging probe + consumer-open" against "the artifact's ENTRY file", with no per-source-family / per-consumer multiplicity requirement (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# artifact-reuse.md (h)(iv) requirement (3): make the probe multiplicity explicit —
+ Run the 1-file staging probe + consumer-open ONCE PER reused source-family
+ AND per staged consumer (each family's entry file staged through its real
+ staging path, each consumer's entry-point open/init run against its staged
+ root); a single global probe on one family/consumer does NOT satisfy this
+ leg when the plan stages ≥2 families or ≥2 consumers (#1481: two same-day
+ staged-layout crashes — datagen_topup vs datagen 404, and a stage_hub_prefix
+ reread crash).
```

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Amend (h)(iv) requirement (3); keep the existing pure-mapping-fn +
  fail-loud-entry-check + real-staging-path requirements intact.

## Constraints / invariants

- Workflow-surface only. Do not weaken the "must exercise the real staging
  path" requirement — this only multiplies the probe across families/consumers.
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 6e2e44e3a0c0

Surfaced problem (c3-P5 + c3-P12): two #1481 staged-layout crashes in one day
that a single global (h)(iv) staging probe did not prevent.
