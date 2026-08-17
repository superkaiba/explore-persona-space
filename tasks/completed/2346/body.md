---
title: 'gotchas.md: warn that an interrupted posix_fallocate quota probe leaks its
  full allocation (12.6 GB leaked on a live pod, #2329)'
kind: infra
tags: []
created_at: '2026-08-17T13:25:26Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'orchestrator-surfaced during /issue 2329: a SIGTERMed 15 GiB ad-hoc
  fallocate probe left 12,864 MB on pod-2329 because Python finally does not run under
  SIGTERM'
workflow: v1
---
## Goal

`.claude/rules/gotchas.md` directs agents to use `os.posix_fallocate` to detect the
RunPod MooseFS per-pod quota (correctly — `df` cannot see it and reports the cluster's
free space, so `shutil.disk_usage` and `df` both miss `OSError errno=122 EDQUOT`).
The guidance carries no warning that an INTERRUPTED probe leaks its entire
allocation. Add that warning plus the safe form.

## What happened (#2329, 2026-08-17)

While pre-checking disk headroom before a ~6.4 h / ~51 GPU-h grid phase on a live
8x H100 pod, the orchestrator ran an ad-hoc 15 GiB probe of this shape:

```python
fd = os.open(p, os.O_CREAT | os.O_WRONLY, 0o600)
os.posix_fallocate(fd, 0, 15 << 30)
os.close(fd)
...
finally:
    os.unlink(p)          # <-- never runs under SIGTERM
```

The call exceeded its timeout and the process was SIGTERMed (exit 143). Python's
`finally` does not run on SIGTERM, so the probe file survived at **12,864 MB**
(13,509,820,416 B apparent) on a pod with live compute — i.e. the probe consumed
~12.6 GB of exactly the quota headroom it was measuring, and could itself have
caused the EDQUOT it was checking for. It was found and removed only because the
next status check happened to `ls` the path; otherwise it would have sat there
through the entire grid phase.

Two independent defects in the ad-hoc form:

1. **Interrupt-leak.** A large `posix_fallocate` is slow enough on MooseFS to hit
   any surrounding timeout, and a Python `finally` cannot clean up under SIGTERM.
2. **I/O contention.** A multi-GiB allocation competes with running compute for
   pod I/O, for information a 1 GiB probe already provides.

## Proposed fix

In `.claude/rules/gotchas.md`, at the existing MooseFS EDQUOT / `posix_fallocate`
entry, add:

- Keep ad-hoc probes SMALL (~1 GiB) and extrapolate from an already-measured
  footprint. A 1 GiB probe returned in under a second on the same pod; the 15 GiB
  probe never returned at all. Probe size buys no extra confidence about a later
  hour of the run.
- If a probe must be large, wrap cleanup so it survives the signal —
  `trap 'rm -f "$F"' EXIT INT TERM` in shell, not a Python `finally`.
- Prefer CONTINUOUS observation over a one-shot upfront probe for a long phase: a
  `du -sBM` on the out-root at each heartbeat costs nothing and catches divergence
  from the projected footprint while there is still time to act. A probe that
  passes at hour 0 says nothing about hour 5.
- After ANY interrupted probe (non-zero/​signal exit), `ls` the probe path and
  remove a survivor before continuing — treat the leak as the default outcome of
  an interrupted probe, not an edge case.

`explore_persona_space.orchestrate.preflight`'s own probe is the sanctioned check
and is NOT implicated (it runs inside a managed function); this is about the
ad-hoc variants the rule's guidance invites. Worth a scan of that probe's cleanup
path for the same signal-safety property while here.

## Acceptance

- `gotchas.md` MooseFS/EDQUOT entry carries the four bullets above.
- The preflight probe's cleanup is confirmed signal-safe (or fixed).
- No behavior change to preflight's PASS/FAIL semantics.
