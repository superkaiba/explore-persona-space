---
title: poll_pipeline.py log-freshness globs miss the <out_root>/logs/ worker-log layout,
  causing false stalled verdicts
kind: infra
tags: []
created_at: '2026-08-26T14:51:15Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
---

# poll_pipeline.py log-freshness globs miss the `<out_root>/logs/` worker-log layout, producing false `stalled` on healthy multi-worker phases

## Goal

Close a monitoring blind spot in `scripts/poll_pipeline.py`: when a pod-side dispatcher writes
its per-worker logs under its OWN out-root, none of the poller's three log-freshness globs match
them, so staleness is judged solely on the dispatcher log — which is idle by design while
workers run. The result is a false `stalled` verdict on a perfectly healthy phase.

## The defect

`poll_pipeline.py` derives its staleness verdict from

```
min(last_mtime_ago, phase_log_mtime_ago, shard_log_mtime_ago) <= stall_sec     # :3560
```

The three probes cover (docstring at `:1804-1836`):

1. `last_mtime` — the main dispatcher log passed via `--log`
2. `phase_log` — `/workspace/logs/issue-<N>-*.log` (flat, excluding the top-level log)
3. `shard_log` — three repo-rooted layouts: `<repo>/logs/issue_<N>/*.log`,
   `<repo>/logs/issue_<N>_*.log`, and `<repo>/eval_results/issue_<N>{,_*}/logs/*.log`

A dispatcher that writes per-worker logs to `<out_root>/logs/<cell>.slotN.log` — where
`out_root` is an arbitrary path passed at launch (e.g. `/workspace/issue2546`) — matches NONE of
these. Both derived probes return the `10**9` not-found sentinel (`:5581-5585`), so they cannot
contribute to the `min()`, and freshness collapses onto the dispatcher log alone.

## Observed instance (issue 2546, arm 3)

`scripts/issue2546_dispatch.sh` spawns four capture workers, each logging to
`/workspace/issue2546/logs/capture-think_on__gsm8k_test.slot<N>.log`. The dispatcher log emits
its `[spawn] slot 0..3` lines and then goes quiet for the entire cell. A tick 13 minutes later
returned:

```
"phase_log_mtime_sec_ago": 1000000000, "shard_log_mtime_sec_ago": 1000000000,
"last_log_mtime_sec_ago": 793, "pid_identity": "match", "cpu_advancing": true
```

Both worker-log probes blind, while the workers were verifiably healthy (D-state page-ins with
the mount answering spot probes, CPU time accumulating, GPU memory climbing 3109 -> 14045 MiB).
At the 900 s `DEFAULT_STALL_SEC` the next tick would have reported `stalled` on a healthy phase.

Severity is bounded and worth stating precisely: `dead` is NOT affected, because the #2265
evidence veto independently reads output mtime and refuses a dead verdict against fresh
outputs. This is a `stalled`-only false positive.

## Why the current workarounds are unsatisfying

- **Widening `--stall-sec` per issue** (what 2546 did, 900 -> 3600 s) trades a false positive for
  degraded genuine detection, and has to be re-derived per phase shape.
- **Pointing `--log` at a worker log** breaks `[phase=...]` parsing, which needs the dispatcher
  log.
- **Relying on the operator's own probes** works but defeats the purpose of an automated poller.

## Suggested direction (for the implementing session to evaluate, not a pre-approved design)

Add a fourth mtime probe covering the out-root worker-log layout. The out-root is not knowable
from `--issue` alone, so it needs a source. Options worth weighing:

1. An explicit optional `--worker-log-glob` / `--out-root` flag the orchestrator passes when it
   knows the dispatcher's layout. Most precise, requires caller adoption.
2. Read the out-root from the persisted dispatch handle
   (`.claude/cache/issue-<N>-handle.json`), which several lanes already maintain. No new caller
   duty, but couples the poller to the handle schema.
3. Widen the existing glob set with a bounded search for `*/logs/*issue*<N>*` style paths under
   common pod roots. Least caller work, highest false-match risk, and a broad find on MooseFS is
   slow — the `.claude/rules/gotchas.md` SSH-MCP 30 s ceiling entry is directly relevant.

Whatever the mechanism, it should preserve the existing `max`-over-layouts reduction so any one
fresh layout keeps the phase in `running`, and keep the not-found degrade non-fatal.

## Acceptance

- A dispatcher writing worker logs under an arbitrary out-root keeps the phase in `running`
  while those logs advance, without a per-issue `--stall-sec` override.
- The `stalled` verdict still fires when ALL known log layouts genuinely go stale.
- `dead`-verdict behavior and the #2265 evidence veto are untouched.
- A regression test pins the out-root layout being seen, using the 2546 arm-3 path shape.

## Provenance

Found while diagnosing issue 2546 arm 3 on 2026-08-26: four healthy capture workers were
invisible to the poller and only a direct `wchan` + spot-probe investigation distinguished a
MooseFS grind from a FUSE read-wedge. Distinct from #2602 (gate-anchor instrument staleness),
#2603 (CVD-UUID teardown fail-open), and #2604 (shared stash stack across worktrees).
