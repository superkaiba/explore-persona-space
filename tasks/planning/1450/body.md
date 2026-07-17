---
title: 'daily-fix: size-cap trigger for home HF cache reap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7ee891aa6903
- daily-auto-filed
created_at: '2026-07-17T06:56:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the #1376/#1377 home-HF-cache
  reap is age-gated only (EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS, default 7d), so ACTIVE
  downloads grew ~/.cache/huggingface/hub to 101 GB (295 cached revisions of the project
  data repo, ~37.6 GB) and drove / to 98% twice on 2026-07-16 with the guard never
  firing on them'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining: two /-at-98% incidents in one day (04:30Z chat recovery pruned 16 stale revisions -> 81%; 04:37-05:26Z session found 295 cached revisions ~37.6 GB the guard never swept).

## Goal

Stop fresh high-churn HF revision accumulation from exhausting / before the age gate can ever fire.

## Workflow gap

- **Bug observed:** the #1376/#1377 home-HF-cache reap is age-gated only (EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS, default 7d), so ACTIVE downloads grew ~/.cache/huggingface/hub to 101 GB (295 cached revisions of the project data repo, ~37.6 GB) and drove / to 98% twice on 2026-07-16 with the guard never firing on them
- **Why it is a workflow gap:** An age-only reap is structurally blind to same-day churn; the guard exists precisely to keep / off the silent-Bash-failure regime.
- **Confidence (emitter):** high (two incidents in one day)
- verified-at-filing: `grep -n 'EPS_VM_HOME_HF_REVISION_MAX_AGE_DAYS' scripts/vm_disk_guard.py` -> L47/L809/L825-826 (age gate present); `grep -n 'size.cap\|SIZE_CAP' scripts/vm_disk_guard.py` -> 0 hits (absence claim)

## Proposed change (candidate diff sketch — refine in planning)

add a size-cap trigger (total home-hub-cache bytes, or per-repo revision-count/bytes cap) to the home-HF-cache pass so fresh-but-huge revision accumulations reap oldest-first under disk pressure, keeping the newest + ref'd revisions per the existing contract

## Scope / surfaces

- Primary target: `scripts/vm_disk_guard.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 7ee891aa6903

