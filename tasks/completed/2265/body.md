---
title: poll_pipeline emits status:dead while its own gpu_util + log-mtime evidence
  shows the workload alive
kind: infra
tags: []
created_at: '2026-08-13T06:07:44Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'Surfaced during /issue 2223: poll tick returned status:dead with pid_alive:false
  while gpu_util read 97,100,100,100 and last_log_mtime_sec_ago was 294 on a fully
  healthy 4-shard leg.'
workflow: v1
---
# poll_pipeline returns `status: dead` while its own collected evidence shows the workload ALIVE

## Goal

Make `scripts/poll_pipeline.py` refuse to emit a `dead` status verdict when the evidence it has
ALREADY collected in the same tick contradicts a dead workload. Today a stale pid file alone is
sufficient for `status: dead`, even when the tick's own `gpu_util` and `last_log_mtime_sec_ago`
fields say the workload is actively computing.

## Observed (task #2223, 2026-08-13 ~06:06Z, pod-2223-q32b)

> Timestamp corrected 2026-08-13 (consistency-checker, plan round): originally recorded as "2026-08-12 23:06Z", which is local UTC-7 — the literal Z reading predates the 32B leg's 00:00:57Z launch. Durable corroboration: #2223 `epm:progress` v73 (2026-08-13T06:05:22Z); the tick JSON below is a session-side poll read consistent with that state.

One tick returned, in a single JSON object:

    {"status": "dead", "pid_alive": false, "pid_identity": "unknown",
     "sig_proc_rescue": false, "last_log_mtime_sec_ago": 294,
     "gpu_util": "97,100,100,100", "current_phase": "generate", "stall_reason": null}

`status: dead` — while the SAME payload reports all four GPUs at 97-100% utilization and a log
written 294 s ago. The leg was in fact entirely healthy: four detached generate shards were
running (verified by `pgrep` shard-id enumeration, per-shard pidfiles, and `/proc/<pid>/environ`).

Ground truth at that moment: the pid file `/workspace/logs/issue-2223-32b.pid` held the pid of a
launcher that had exited (`[launch] FATAL: a generate shard failed for arm=A0`) after all four of
its shard subprocesses OOMed. The four replacement shards had been relaunched `setsid`-detached, so
they survived the launcher's death and kept generating — but none of them owned the canonical pid
file. The tick's staleness WARNING fired correctly; the VERDICT did not follow it.

## Why this matters

`dead` is a strong, actionable claim. A dead verdict on a healthy leg can route to failure
handling / a watcher respawn / a relaunch decision, and a relaunch on top of live detached workers
is the duplicate-runner collision the ownership-check rules exist to prevent (same GPUs, same
output paths). Here the orchestrator caught it only because it independently held GPU and shard
evidence; a less-instrumented consumer would have taken `dead` at face value.

The poller already collects everything needed to veto the verdict — this is not a new probe, just
a consistency check over fields it returns in the same object.

## Proposed change (implementation is the spawned session's to design)

When `pid_alive` is false, do NOT emit `status: dead` if same-tick evidence contradicts it.
Candidate corroboration signals, all already present:

- `gpu_util` — any GPU meaningfully busy on the pod
- `last_log_mtime_sec_ago` / `phase_log_mtime_sec_ago` — recent writes
- a bracketed `pgrep` for the phase's own invocation (the `sig_proc_rescue` path already exists and
  returned false here — worth understanding WHY it did not rescue a live, differently-named worker)

Emit a distinct non-terminal verdict instead (`unknown` / `pid-stale-workload-live` /
`needs-probe`) carrying the contradiction, so the consumer probes rather than concludes. Reserve
`dead` for pid-absent AND no corroborating liveness.

Do NOT simply suppress the staleness warning — the warning was correct and useful. The defect is
the verdict, not the warning.

## Scope notes

- The #2259 sibling defect (pid-staleness compared ISSUE-keyed rather than POD/leg-scoped, so one
  leg's `epm:run-launched` makes another leg's pid file read stale) is RELATED but DISTINCT: that
  one produces a spurious staleness WARNING; this one produces a wrong terminal STATUS. Fixing
  #2259 would not fix this.
- The pid-file launch contract (#813) says a relaunch must rewrite the pid file with the live
  workload pid, and in #2223 that contract was genuinely not satisfiable by the detached catch-ups
  (they are per-shard workers, not the launcher). A verdict that degrades safely under a violated
  contract is the point — contracts get violated in recovery scenarios, which is exactly when a
  wrong `dead` is most damaging.
- Verification should include a test reproducing the shape: pid file pointing at a dead pid, GPU
  util non-zero, recent log mtime => verdict is NOT `dead`.

## Provenance

Surfaced by the #2223 orchestrator while recovering the 32B leg from four consecutive shard OOMs.
Evidence: #2223 `epm:progress` v73 (launcher death + recovery record) and the tick payload quoted
above.
