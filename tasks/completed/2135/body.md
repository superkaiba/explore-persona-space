---
title: 'daily-fix: pod HOLD gates + launches must arm a re-drive'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e7fcd03e307
- daily-auto-filed
created_at: '2026-08-06T07:20:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): pod-1947-r3 sat 10h at
  an unowned HOLD gate; pod-1947-loc bootstrapped and never launched (~$500 idle)'
workflow: v1
---
# daily-fix: pod HOLD gates and post-provision launches must arm a re-drive / confirm a live workload — two 10h idle pods (~$500)

## Workflow gap

Two #1947 pods each burned ~10 h idle on 2026-08-04 because a pod-side state had no
VM-side owner:

1. **pod-1947-r3 (8×H200, ~$32/h):** finished smoke+pilot and stopped at its designed
   circuit-breaker "[launcher] pilot leg rc=0 — HOLD for cost review" at 03:52:43Z — but
   no review loop was armed; the session's own admission: "I built that gate and never
   closed the loop on it." Discovered only by Thomas's "check progress" at ~13:58Z.
2. **pod-1947-loc (4×H200):** bootstrapped and never ran any workload — no logs dir, no
   processes, nothing written in 6 h; the dispatch that should have launched its workload
   never fired. Terminated on Thomas's "Kill loc immediately — it's pure waste."

verified-at-filing: both are the recovery miner's probed live-API/ssh reads at discovery
time (session 024a96de rows 1544–1630: uptime 10.48 h, 0% on 8 GPUs, empty proc list).
The pods are long gone; the gap is the pattern.

## Proposed change

Two clauses in `.claude/rules/pod-side-reporting.md` (the (re)launch contract):
1. Any pod-side HOLD/gate emission must arm a VM-side re-drive at the moment the HOLD is
   written (an `epm:progress` hold note naming the reviewer + a Monitor/cron wakeup); a
   HOLD sentinel with no armed reviewer is itself watcher-escalation material.
2. Post-provision launch confirmation: the launcher verifies (and records in the
   `epm:run-launched` note) a live workload pid + first log line within N minutes of
   bootstrap, else escalates/self-terminates — mirror into
   `.claude/agents/experimenter.md`'s launch contract.

## Provenance

- fingerprint: 1e7fcd03e307

- workflow_fix_target: .claude/rules/pod-side-reporting.md, .claude/agents/experimenter.md
- origin: /daily 2026-08-04 recovery sweep — miner 4 P1/P2 (probed at discovery).
