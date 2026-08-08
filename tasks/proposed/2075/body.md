---
title: 'workflow-fix: pod audit terminates teammate pods (substring ownership + wrong
  stale clock)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c55989f33b27
created_at: '2026-08-04T21:21:45Z'
has_clean_result: false
origin_prompt: 'User chat 2026-08-04: teammate reported ''styfeng-8xH200 was cancelled'';
  investigation found pod_audit --terminate-stale killed 77 teammate-owned pods.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised
during an interactive-chat incident investigation (2026-08-04): a teammate reported
their pod `styfeng-8xH200` was cancelled. Our daily `cron_pod_audit.sh`
(`pod.py audit-stale --terminate-stale --yes`) terminated it, along with 76 other
teammate-owned pods on the shared RunPod team account.

## Goal

Stop `pod_audit.py --terminate-stale` from destroying teammate-owned pods: replace the
substring-match ownership signal with a provenance-backed one, and measure staleness
from the EXIT time rather than pod creation time.

## Workflow gap

- **Bug observed:** `pod_audit.py --terminate-stale` (daily cron, `37 9 * * *`)
  terminated 77 unique pods between 2026-07-22 and 2026-08-04, **none** of them
  EPS-owned (0 carried a `pod-` / `epm-issue-` name). Confirmed kill of
  `y3b0x9o15yn7ak` / `styfeng-8xH200` (8×H200) at 2026-07-31T09:39-07:00
  (`logs/pod_audit/2026-07-31.log:203`). Affected owners include styfeng (9 pods),
  asherps (10), dipika (7), mattschwartzscience (6), plus ~15 others.
- **Why it is a workflow gap:** two independent defects in the workflow-surface
  script `scripts/pod_audit.py`:
  1. **Ownership signal 3 is a bare substring match.** `_scan_task_references`
     (line 154) reads each task's whole `events.jsonl` blob and matches
     `if any(n in blob for n in needles)` on the pod NAME / id (line 166). EPS
     sessions routinely post fleet-audit dumps into `epm:progress` markers — dumps
     that enumerate every team pod. The poisoning text on #1738/#1739 is literally
     the audit's own report line `unmanaged-name  y3b0x9o15yn7ak  RUNNING
     'styfeng-8xH200'`. So a pod the audit itself labelled NOT-ours became
     "EPS-owned" the next day, flipping it from the report-only
     `unmanaged-exited` bucket into the auto-terminate `stale` bucket
     (line 406). Self-poisoning loop: the report is the evidence.
  2. **The staleness clock measures the wrong interval.** Line 357 computes
     `age = _age_hours(p.created_at)` and line 395 gates on
     `age >= max_exited_hours`. The documented policy (module docstring line 15,
     `cron_pod_audit.sh` header) is "EXITED for longer than 24h" — but a pod
     CREATED >24h ago and stopped 10 minutes ago is immediately terminable. The
     24h grace period does not exist. `styfeng-8xH200` was RUNNING at the
     2026-07-30 audit (age 65.2h) and destroyed at the 2026-07-31 audit — its
     actual exited-duration was under 24h.
  Together: the destructive arm fires on teammate pods, with no grace window, on a
  shared account. `#1471` (commit `7d50a1d082`, 2026-07-17) extended the
  auto-terminate from managed-name-only to ALL names gated solely on
  `_is_eps_owned`, which is what put every team pod in range of defect 1.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "age = _age_hours(p.created_at)" scripts/pod_audit.py`
  → 1 hit (line 357); `grep -n "if any(n in blob for n in needles)" scripts/pod_audit.py`
  → 1 hit (line 166); `grep -n 'bucket = "stale" if _is_eps_owned' scripts/pod_audit.py`
  → 1 hit (line 406); all three in the single named target
  `scripts/pod_audit.py` (2026-08-04). Kill record:
  `grep -rn "y3b0x9o15yn7ak" logs/` → `logs/pod_audit/2026-07-31.log:203`
  `ok   y3b0x9o15yn7ak  styfeng-8xH200`.

## Proposed change (candidate diff sketch — refine in planning)

    # 1. Retire the substring signal; require provenance we WROTE.
    - if any(n in blob for n in needles):
    + # Match only pods THIS project provisioned: a structured marker field
    + # (epm:run-launched pod=<name> / epm:pod-terminated), never a bare
    + # substring of the events blob. An audit dump quoting a pod name is
    + # NOT evidence of ownership.

    # 2. Measure staleness from the EXIT, not from creation.
    - age = _age_hours(p.created_at)
    + exited_age = _age_hours(p.last_status_change)   # time since EXITED
    + # bucket 'stale' on exited_age >= max_exited_hours; keep created_at
    + # age for the display column / orphan-running only.
    + # Unknown/unparseable last_status_change => fail toward KEEP.

Consider additionally: an explicit EPS-provisioned allowlist (pods_ephemeral.json as
the authoritative record) as the ONLY terminate-eligible set, i.e. make signal 3
report-only.

## Scope / surfaces

- Primary target: `scripts/pod_audit.py`
- Also update the stale policy prose in `scripts/cron_pod_audit.sh` (header) and the
  `pod_audit.py` module docstring so the documented contract matches the code.
- Add regression tests: (a) a pod whose name appears only inside an audit-dump
  `epm:progress` note is NOT terminate-eligible; (b) a pod created 100h ago and
  EXITED 1h ago is NOT stale.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The RunPod account is TEAM-SHARED. Fail-toward-KEEP on every ambiguity: a false
  keep costs volume storage; a false terminate irreversibly destroys a teammate's
  data.
- The cron entry (`37 9 * * *`) was DISABLED in the user crontab on 2026-08-04 as
  incident containment. Re-arming it is part of this task's completion, and only
  after the tests above pass.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod_audit.py
- fingerprint: c55989f33b27

<!-- workflow-fix-candidate v1 -->
target_file: scripts/pod_audit.py
bug_observed: The daily `pod.py audit-stale --terminate-stale` cron terminated 77 unique teammate-owned RunPod pods (0 EPS-owned) between 2026-07-22 and 2026-08-04, including `styfeng-8xH200` on 2026-07-31.
why_workflow_gap: `_scan_task_references` declares a pod EPS-owned on a bare substring match of its name against any task's whole events.jsonl blob, so EPS's own fleet-audit dumps pasted into `epm:progress` markers read back as proof of ownership; and the staleness clock uses `created_at` instead of time-since-EXITED, so the documented 24h grace period does not exist.
proposed_change: Require structured provenance (a marker field naming a pod we provisioned) instead of a substring match for terminate-eligibility, and gate the stale bucket on time-since-EXIT rather than pod creation age.
diff_sketch: |
  - if any(n in blob for n in needles):
  + # structured marker-field match only; an audit dump quoting a pod name
  + # is NOT ownership evidence
  - age = _age_hours(p.created_at)
  + exited_age = _age_hours(p.last_status_change)  # fail toward KEEP if unknown
confidence: high
related_task: n/a (interactive-chat incident, 2026-08-04)
<!-- /workflow-fix-candidate -->
