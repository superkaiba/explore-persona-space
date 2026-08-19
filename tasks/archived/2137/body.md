---
title: 'daily-fix: pod_audit ownership by substring self-poisons'
kind: infra
tags:
- wf-fix
- wf-fix-fp:57a258d321c2
- daily-auto-filed
created_at: '2026-08-06T07:21:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): bare substring match over
  events blobs destroyed styfeng-8xH200 and mis-claimed cluster nodes 13 days; interlock
  landed, inference residual'
workflow: v1
---
# daily-fix: pod_audit ownership inference is a bare substring match over events blobs — destroyed a fellow's pod, mis-claimed cluster nodes for 13 days

## Workflow gap

`scripts/pod_audit.py`'s `_scan_task_references` infers EPS ownership by bare substring
match over whole `events.jsonl` blobs, which self-poisons: any pod name PASTED into a
marker note (fleet listings, incident reports) makes that pod read as EPS-owned. Two
consequences found 2026-08-04:

1. The daily pod-audit cron terminated another fellow's pod `styfeng-8xH200` on the
   shared org (2026-07-31 09:39 PT; EXITED at the time, volume/data destroyed) — the
   fellow messaged Thomas. Diagnosed in-session at `logs/pod_audit/2026-07-31.log:203`.
2. Near-miss: fellows-cluster nodes were tagged `task #1112` in EVERY audit log
   07-22→08-03; "Any one of them EXITing on an audit morning would have been destroyed"
   (a creation-age clock bug guaranteed stale-eligibility at 4,000+ h).

The kill-approval interlock (EPS_ALLOW_POD_TERMINATE / EPS_ALLOW_COMPUTE_KILL at the
terminate_pod choke point; audit path now refuses) landed in-session — this filing is the
RESIDUAL: the ownership inference itself is still substring-based, so the audit's
reports/escalations still mis-attribute, and any future re-enable inherits the defect.

verified-at-filing: incident + near-miss are the recovery miner's probed reads of session
b4963de1 (rows 127, 179, 225, log-line evidence quoted in-session).
`grep -n '_scan_task_references' scripts/pod_audit.py` run at compose time to confirm the
inference path is live.

## Proposed change

Replace substring ownership inference with structured evidence: match only pod names/ids
in STRUCTURED marker positions (the `pod=<name>` token convention, `epm:run-launched`
lead position — the same fields the watcher's per-pod shield reads), never free prose;
treat pasted-listing hits as non-evidence. Fix the creation-age clock bug the near-miss
rode on. Keep the interlock as the enforcement backstop.

## Provenance

- fingerprint: 57a258d321c2

- workflow_fix_target: scripts/pod_audit.py
- origin: /daily 2026-08-04 recovery sweep — miner 4 P5/P6.
