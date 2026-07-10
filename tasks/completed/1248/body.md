---
title: sweep suppression misses prior-night disposition records
kind: infra
tags:
- wf-fix
- wf-fix-fp:187a7efa9336
- daily-auto-filed
created_at: '2026-07-10T06:55:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): candidates on 815/880/917
  re-enumerated tonight despite epm:workflow-fix-task-filed disposition records posted
  last night (fingerprint-form mismatch)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from an orchestrator observation during the Step-C parked-candidate routing pass.

## Goal

Make `scripts/sweep_parked_wf_candidates.py` suppression recognize already-routed candidates whose routed-record marker carries a different fingerprint form than the candidate (e.g. `fingerprint: n/a (prose park)` records vs a formal-block candidate fingerprint), so dispositioned candidates stop re-enumerating night after night.

## Workflow gap

- **Bug observed:** Tonight's sweep (2026-07-10T06:29Z) re-enumerated at least 3 candidates (#815 fp 123060ae62e0, #880 fp fe046b20d35c, #917 fp ab749bae51d7) that last night's /daily (2026-07-09T07:02Z) had already dispositioned with `epm:workflow-fix-task-filed` routed-records posted on the same source tasks ("already-fixed on main..."). The triage pass had to re-verify all three from scratch (subagent time + tokens), and they will re-enumerate again tomorrow.
- **Why it is a workflow gap:** The sweep's routed-suppression predicate is what bounds re-scans (daily SKILL.md Step C: "suppression is what bounds re-scans"); a record that fails to match its candidate — plausibly because the record's `fingerprint:` field reads `n/a (prose park)` or a different fp than the enumerated candidate's, or because the record's `origin_candidate_ts` is absent/mismatched — leaves the candidate permanently un-suppressed and the nightly sweep re-triages it forever.

## Proposed change (refine in planning)

Diagnose why the three named records did not suppress (fingerprint-form mismatch vs origin_candidate_ts keying vs target_file string mismatch), then widen the suppression predicate accordingly — e.g. match on (source task, origin_candidate_ts) when the fingerprint is `n/a`, and/or normalize target_file comparison — with a regression test using the #815/#880/#917 record shapes verbatim.

## Scope / surfaces

- Primary target: `scripts/sweep_parked_wf_candidates.py`
- Secondary: `tests/` (regression fixtures from the #815/#880/#917 records)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: scripts/sweep_parked_wf_candidates.py

Evidence: #815 events 2026-07-09T07:02:42Z (`filed_task: n/a (already-fixed on main: guard_repo_root_branch.sh line 523 ...)`), #880 07:02:37Z, #917 07:02:37Z — all re-enumerated by the 2026-07-10T06:29Z sweep with `suppressed: false`.
