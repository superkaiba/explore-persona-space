---
title: 'poll_pipeline pid-file staleness check is not pod-scoped — disables the #813
  liveness check on all but the newest leg of a multi-pod issue'
kind: infra
tags: []
created_at: '2026-08-13T00:05:37Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'Surfaced by the #2223 orchestrator: poll tick 2 on the 7B leg WARNed
  pid-file-predates-newest-run-launched while every other health signal was green;
  the newest marker belonged to the sibling 32B pod.'
workflow: v1
---
# poll_pipeline's pid-file staleness check is not pod-scoped — it silently disables the #813 liveness check on all but the newest leg of a multi-pod issue

## Goal

Make `scripts/poll_pipeline.py`'s `epm:run-launched` reads POD-SCOPED when `--pod` is
given, so a multi-pod single-issue run compares each pod's pid file against ITS OWN
launch marker. Today the checks read the issue's newest marker regardless of pod,
which both emits a recurring false WARN and — the part that matters — turns the
#813 stale-pid safety check into a no-op for every leg that is not the most recent
launcher.

## The gap

Three helpers resolve the launch marker by ISSUE only, with no pod filter, despite
the poller already knowing its pod from `--pod`:

- `scripts/poll_pipeline.py:969` — `latest_event(issue, prefix="epm:run-launched")`
  (returns the marker `pid=`)
- `:1000` — same call, returns `(pid, note_text)`
- `:1020` — same call, returns the marker age

`:1047` (`pid_file_stale_vs_marker`) then compares the polled pod's pid-file mtime
against that issue-wide newest marker.

Multi-pod single-issue runs are SANCTIONED, not exotic: `.claude/rules/pods.md`
specifies `pod-<N>-<slug>` for "a round needing a SECOND pod", and #1961 added a
per-pod watcher shield for exactly this shape. So the poller is the surface that did
not get the multi-pod treatment.

## Why this is more than log noise

`pid_file_stale_vs_marker` exists to catch a stale pid file masking a DEAD RELAUNCH
(the #813 pid-file rewrite contract). On a two-pod issue:

- Leg B launches second, so B's marker is the issue's newest.
- Leg A's pid file legitimately predates B's marker, so A trips the staleness WARN
  on EVERY tick, forever.
- `marker_pid_identity` for A resolves against B's `pid=`, so it reads `unknown` —
  the cross-check that would catch A's pid file pointing at a dead process no longer
  evaluates anything for A.

Net: the check is inverted from its purpose on the older leg — permanently WARN-ing
where nothing is wrong, while structurally unable to fire where something is. And a
WARN that fires on every tick of every multi-pod run is a channel sessions learn to
ignore, which is the failure mode `.claude/rules/repo-root-uncommitted-state.md`
names for its own escalation design.

## Measured evidence — #2223 (2026-08-12/13)

#2223 runs two legs in parallel on one issue (a user-directed parallel launch):
`pod-2223` (4x H100, 7B leg, pid 3139, marker 23:44:11Z) and `pod-2223-q32b`
(4x H200, 32B leg, pid 2285, marker 00:00:57Z).

Poll tick 2 for the 7B leg, verbatim:

```
WARNING poll_pipeline: pid file /workspace/logs/issue-2223-7b-full.pid on pod
pod-2223 predates the newest epm:run-launched marker for #2223 (pid-file age 1323s
vs marker age 198s, slack 600s) — possible stale pid from a prior launch masking a
dead relaunch (#813 pid-file rewrite contract). WARN-only; status verdict unchanged.
```

Same tick's JSON: `"pid_file_stale_vs_marker": true`, `"marker_pid_identity":
"unknown"`, alongside `"pid_alive": true`, `"pid_identity": "match"`,
`"cpu_advancing": true`, `"gpu_util": "97,96,97,97"`, `"status": "running"` — i.e.
the leg was demonstrably healthy and the WARN was a cross-leg comparison artifact.
The "prior launch" it names never existed; the newer marker belongs to the OTHER pod.

## The fix is cheap because the data already exists

#1961 made the pod name MANDATORY in structured position in the `epm:run-launched`
note — LEAD with the pod name or carry a `pod=<name>` token — precisely so consumers
can attribute a marker to a pod (it arms the watcher's per-pod shield). The poller
already parses the note's free-form `key=value` tokens (`:478-483`). So:

- Add a pod filter to the three marker reads: prefer the newest `epm:run-launched`
  whose note attributes to `--pod` (leading pod name or `pod=<name>` token).
- No attributable marker for this pod ⇒ fall back to today's issue-wide behaviour
  (back-compat for single-pod runs and pre-#1961 notes), and say which path was taken.
- Keep it WARN-only; this task changes WHICH marker is compared, not the verdict
  semantics.

## Acceptance

- A fixture with two `epm:run-launched` markers on one issue naming different pods:
  polling the OLDER pod compares against ITS OWN marker — no staleness WARN — and
  `marker_pid_identity` resolves against that pod's `pid=` (`match`, not `unknown`).
- A genuinely stale pid file on the older leg STILL trips the WARN (the check must
  keep working, not just stop firing) — the load-bearing test.
- Single-pod issues and pre-#1961 notes with no pod attribution behave exactly as
  today.
- `tests/` pins both the pod-scoped resolution and the fallback.

## Related

- `scripts/poll_pipeline.py:969,1000,1020,1047,1098` (the unscoped reads + the WARN)
- #813 (the pid-file rewrite contract this check enforces)
- #1156 (the staleness check itself)
- #1961 (per-pod `epm:run-launched` attribution — the convention that makes the fix
  a filter rather than a redesign)
- `.claude/rules/pods.md` § naming (`pod-<N>-<slug>` second-pod convention)
- `.claude/rules/pod-side-reporting.md` § Pid-file launch contract

## Provenance

Observed by the #2223 orchestrator while polling two live legs of one issue
(2026-08-13T00:0xZ). Diagnosed by reading the poller's marker-resolution helpers
after the WARN contradicted every other health signal in the same tick; the
unscoped `latest_event(issue, ...)` calls were confirmed by direct inspection, not
inferred. Auto-filed per the workflow-fix-on-bug protocol. #2223 itself needs no
change — its run is healthy and the WARN is cosmetic THERE; the durable defect is
the disabled liveness check for multi-pod runs generally.
