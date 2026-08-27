---
title: poll_pipeline cpu_advancing override has no rate floor, masking an HF-transfer
  deadlock as healthy
kind: infra
tags: []
created_at: '2026-08-27T08:33:06Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'orchestrator-diagnosed during #2546 arm 2 ge_gate: poller logged ''stall
  conjunction met BUT session CPU advancing (current=13.0 vs running-max=207404.0)''
  and reported status=running through a 25-min deadlock; session CPU moved 13.0->39.0
  across ~30 min (~0.9 CPU-sec/min) while all threads sat on futexes with a CLOSE-WAIT
  socket holding 25 unread bytes and rx at 504 bytes/5s'
workflow: v1
---
---
kind: infra
---

# `poll_pipeline` CPU-advancing override has no RATE floor, so it masks an HF-transfer deadlock as healthy

## Goal

Make `poll_pipeline.py` able to distinguish a genuinely-progressing CPU/IO phase from a deadlocked
one. Today its `cpu_advancing` override treats ANY positive session-CPU drift as health, so a fully
deadlocked process reports `status=running` indefinitely. Add a rate floor, and add the one probe
that positively identifies this wedge class.

## Observed (issue #2546 arm 2, `ge_gate`, 2026-08-27)

The phase deadlocked on an HF CDN transfer and the poller reported it healthy across every tick.
Verbatim from the tick log:

    INFO poll_pipeline: stall conjunction met (logs >900s + GPUs idle) BUT session CPU advancing
      (current=13.0 vs running-max=207404.0) on pod pod-2546-arm2
      (#518/#658 silent CPU-bound override); reporting status=running

The stall conjunction (logs >900 s + GPUs idle) had ALREADY fired — correctly. The CPU-advancing
override then suppressed it.

Ground truth at that moment, measured directly:

    wchan               : futex_wait_queue
    threads             : ~18, ALL futex_wait_queue except one timer thread
                          NO thread reading the socket
    socket              : CLOSE-WAIT, **25 bytes unread**, peer = HF CloudFront CDN
    net rx              : 504 bytes / 5 s
    cpu                 : utime 979 + stime 360 ticks => ~13.4 s TOTAL, then flat
    proc state          : S (not D — not blocked on disk)
    MooseFS spot read   : 0.003 s (mount healthy; not the FUSE read-wedge)

The process never recovered and was killed ~25 min in.

## The defect: "advancing" is tested for SIGN, not RATE

Across two ticks roughly 30 minutes apart, `session_cpu_secs` went **13.0 -> 39.0**. That is
positive, so `cpu_advancing` stayed true and the override kept firing. But 26 CPU-seconds over 30
wall-minutes is **~0.9 CPU-sec/min, about 1.5% of one core** — noise, not work.

For contrast, the SAME phase after the wedge was cleared (same script, same pod, now in its refit
stage) measured **801 ticks of CPU per 8 wall-seconds ≈ 60 CPU-sec/min** — a ~65x higher rate. A rate
floor separates the two cases with enormous margin; a sign test cannot separate them at all.

Note the override is not wrong to exist (#518/#658 — a silent CPU-bound analysis phase legitimately
freezes its log and idles GPUs for hours). The bug is only that its liveness test is a sign test.

## Requested changes (for the plan to choose among; not prescriptive)

1. **Rate floor on `cpu_advancing`.** Require the INTER-TICK CPU delta to exceed a threshold rather
   than merely be positive. Either an absolute floor (CPU-sec per wall-minute) or, better, a floor
   relative to the rate that phase has previously sustained — the poller already tracks a
   `running-max`, so it has the material for a self-calibrating comparison.
2. **Add the positively-identifying wedge probe.** The signal that was decisive here, and that the
   poller does not look at: **a socket in `CLOSE-WAIT` holding a NON-ZERO receive queue with no
   process draining it**, together with a near-zero interface rx delta. That combination is not
   ambiguous.
3. **Surface it as its own status**, not as `dead`. The #2265 false-dead veto exists for good reason;
   this wants a distinct verdict (e.g. `transfer-wedged`) carrying the socket evidence, so the
   orchestrator can run the documented `.claude/rules/upload-policy.md` rung ladder instead of
   guessing.

## Negative knowledge — three signals that look diagnostic and are NOT

Learned the hard way in the same session; encoding it so a future implementation does not re-derive
it. Each of these was part of my original wedge diagnosis and each was later observed on a
demonstrably HEALTHY run of the same phase:

- **Thread count on `futex_wait_queue` is useless alone.** The wedge showed ~18/18 threads on
  futexes. The healthy download showed **91 of 92**. It is the normal idle state of a large thread
  pool.
- **`CLOSE-WAIT` presence alone is useless.** The healthy refit stage holds one — an unreaped socket
  from the completed transfer. Only CLOSE-WAIT **with unread bytes and no reader** discriminates.
- **Low CPU delta alone is useless.** The healthy download measured 3 ticks over 10 s while moving
  4.3 MB.

Two live samples each satisfied exactly TWO of three wedge conditions while perfectly healthy, in
opposite directions. So any single-term test false-positives, and the fix must be a conjunction.

## Also worth fixing while here (cheap, same file)

`du`-based progress checks are unreliable on MooseFS: a `du -sb` delta measured **0 bytes** over the
same 10 s in which the interface counter moved 4.3 MB. Any progress signal must use
`/proc/net/dev` deltas and CPU deltas, never a `du` snapshot. (This also produced a 351 MB vs 2.8 GB
discrepancy in the same incident.)

## Not to be done as part of this

Do NOT alter, weaken, or remove the #518/#658 CPU-bound override itself, and do NOT alter the #2265
false-dead evidence veto. Both are correct and both have their own incident histories; this task adds
a rate floor and a new probe, it does not relitigate either mechanism.

Do NOT touch the live #2546 arm-2 pod.

## Provenance

Diagnosed by the orchestrator during #2546 arm 2 `ge_gate`, from direct `wchan` / thread / socket /
`/proc/net/dev` / `posix_fallocate` probes after the poller reported `status=running` through a
25-minute deadlock. Recorded in #2546 markers v168 (diagnosis + rung-1 declaration), v169 (rung-1
engaged, mechanism confirmed), v172 (my own bad ETA corrected), v175 (the conjunction validated in
both directions).
