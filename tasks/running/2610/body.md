---
title: poll_pipeline.py cannot detect terminal success of a single-phase dispatcher
  invocation — clean completion reports pid-stale-workload-live then decays to dead
kind: infra
tags: []
created_at: '2026-08-27T00:16:18Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Observed while driving #2546 arm 3 p5_fits through /issue: phase completed
  43/43 ok with a done-file written, but the 00:14:33Z tick returned status=pid-stale-workload-live
  with stall_reason=pid_dead_evidence:log_fresh, and by its own warning would decay
  to dead within ~stall_sec — a false-failure verdict on a successful phase.'
workflow: v1
---
---
kind: infra
---

# `poll_pipeline.py` cannot detect terminal success of a SINGLE-PHASE dispatcher invocation — a cleanly-completed phase reports `pid-stale-workload-live` and then decays to `dead`

## The defect

`scripts/poll_pipeline.py` reports `status=done` only when it parses a terminal
`[phase=done]` milestone from the log tail, corroborated by either a results sentinel or a
surviving dispatcher process (the #597 / #612 hardening against a false mid-run `-> done`
transition). Dispatcher `.done` sentinel FILES are consulted only as an OUTPUT-ARTIFACT
FRESHNESS signal in the anti-stall path (`poll_pipeline.py:285`, `:2172`), never as terminal
corroboration.

But a single-phase dispatcher invocation deliberately emits NO terminal done line.
`scripts/issue2546_dispatch.sh`:

    722:    echo "[phase=done]"                                                    # `all` path only
    726:    echo "[dispatch2546] single-phase invocation of $PHASE_ARG complete (no terminal done line)"

So on clean completion of a single-phase invocation the poller has no reachable success verdict.
Its pid probes all go dead (the workload finished), and the only outcomes are:

1. `status=pid-stale-workload-live` while the #2265 evidence veto still sees a fresh log — the
   veto firing correctly by its own rules, but producing a label that asserts a live workload
   when nothing is live and the phase SUCCEEDED; then
2. `status=dead` once that freshness evidence decays (~`stall_sec`, default 900 s) — a
   **false-failure verdict on a successful phase**.

## Observed instance (task #2546, arm 3, `p5_fits`)

The phase completed cleanly at 00:08:47Z — 43/43 registry jobs `ok`, `dropped_or_degraded` empty,
all four workers `rc=0`, `issue2546-p5_fits-a3-planv4.done` written, zero processes remaining
(measured 0/0 at 00:09Z). The tick at 00:14:33Z returned:

    {"status": "pid-stale-workload-live", "current_phase": "p5_fits",
     "pid_alive": false, "pid_identity": "unknown", "marker_pid_identity": "unknown",
     "stall_reason": "pid_dead_evidence:log_fresh",
     "phase_log_mtime_sec_ago": 1000000000, "shard_log_mtime_sec_ago": 1000000000,
     "post_done_phase_lines": []}

with the log tail literally containing `[p5] complete: 43 jobs, 0 non-ok (reported)` and
`[dispatch2546] single-phase invocation of p5_fits complete (no terminal done line)`. The
completion evidence was IN the tail the poller had already read; nothing in the contract lets it
act on it.

## Why it matters beyond the label

In this instance a live orchestrator knew the phase had succeeded and dispatched the next one, so
the misleading verdict cost nothing. Unattended, the decay to `dead` is a false-failure signal on
a success, and the crash-recovery / respawn machinery keys on exactly that class of verdict — the
risk is a redundant relaunch of work that already completed, against artifacts already on disk.
The ownership-check and kill-before-relaunch rules exist to catch that, but they should not have
to defend against a poller reporting `dead` for a phase that finished correctly.

## Candidate fixes (for the planner — not pre-decided here)

(a) **Preferred-looking:** teach the poller to accept a PHASE-SCOPED done-file as terminal-done
corroboration — i.e. when the current parsed phase is `<P>` and a
`<out_root>/done/*-<P>-*.done` file exists with mtime after the phase's start, report `done`.
This reuses a signal the dispatcher already writes and that the poller already stats for
freshness, and it does not touch the log-parse path that #597 / #612 hardened.

(b) Have single-phase invocations emit a phase-scoped terminal line the poller can parse. This
re-opens the exact false-`done` risk the dispatcher's `no terminal done line` design and the
poller's docstring (`:340`-`:379`) were built to avoid, so it should clear a higher bar.

(c) Treat `pid-stale-workload-live` with an accompanying phase done-file as a distinct
`done-unconfirmed` verdict rather than letting it decay to `dead`.

Any fix must preserve the #2265 evidence veto (it is doing its job here) and must not weaken the
#597 / #612 false-`done` corroboration.

## Not a duplicate of #2605

#2605 is a VISIBILITY gap: worker/per-slot logs written under the dispatcher out-root sit outside
the poller's three log-freshness globs, so a healthy phase looks stale. This is a TERMINAL-STATE
DETECTION gap on the success path, with a different mechanism and a different fix surface. Both
were observed on the same task; they are distinct bugs on overlapping files.

## Target files

- `scripts/poll_pipeline.py` (terminal-state decision; `.done` currently freshness-only)
- the dispatcher/poller contract documented in `.claude/rules/pod-side-reporting.md`
- `scripts/issue2546_dispatch.sh:722,726` as the observed instance of the single-phase shape

## Provenance

Observed while driving task #2546 arm 3 through `/issue`. Grounded by direct reads of
`poll_pipeline.py` (done-parse + `.done`-as-freshness), `issue2546_dispatch.sh:722,726`, the
00:14:33Z tick JSON quoted above, and a pod-side probe confirming 43/43 `ok` with zero live
processes.
