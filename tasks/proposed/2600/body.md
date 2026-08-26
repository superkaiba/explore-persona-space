---
title: poll_pipeline.py log globs miss /workspace/issue<N>/logs/ out-root layout —
  worker progress invisible, stall detection loses its log signal
kind: infra
tags: []
created_at: '2026-08-26T05:44:26Z'
has_clean_result: false
workflow: v1
---
# poll_pipeline.py per-phase/shard log globs miss the `/workspace/issue<N>/logs/` out-root layout — worker progress logs invisible, stall detection left resting on the cpu_advancing override alone

## Goal

Teach `scripts/poll_pipeline.py`'s phase/shard log probes about the `/workspace/issue<N>/logs/*.log` out-root layout, so a multi-worker generation phase whose progress lines land in per-slot logs is VISIBLE to the poller — for staleness, for the crash-signature tail, and for the operator reading the tick.

## The defect

The two log probes glob exactly five patterns (`scripts/poll_pipeline.py`, `phase_log_probe` ~:1960, `shard_log_probe` ~:2006):

```
/workspace/logs/issue-<N>-*.log
/workspace/explore-persona-space/logs/issue_<N>/*.log
/workspace/explore-persona-space/logs/issue_<N>_*.log
/workspace/explore-persona-space/eval_results/issue_<N>/logs/*.log
/workspace/explore-persona-space/eval_results/issue_<N>_*/logs/*.log
```

A dispatcher whose out-root is `/workspace/issue<N>/` writes its per-worker logs to `/workspace/issue<N>/logs/*.log`. That path matches NONE of the five: it sits directly under `/workspace` (not under the repo checkout) AND uses the no-underscore `issue<N>` form while every log glob uses `issue_<N>`.

The no-underscore form is already a recognized convention ELSEWHERE IN THE SAME FILE — the disk-usage probe globs `/workspace/explore-persona-space/data/issue{issue}` (~:2197) alongside the underscore form. So the inconsistency is internal to the poller, not a dispatcher inventing a novel layout.

## Observed live (#2546, 2026-08-26)

Arm-1 at phase `p2_gen_post_rig`, four vLLM slots fanned across 4 GPUs. Poller tick:

```
"last_log_mtime_sec_ago": 973, "phase_log_mtime_sec_ago": 1000000000,
"shard_log_mtime_sec_ago": 1000000000, "gpu_util": "87,84,89,82",
"cpu_advancing": true, "stall_reason": null
```

Both log probes returned the 1e9 not-found sentinel while the top-level log had been quiet ~16 min. A direct SSH check found all four slot logs fresh within ~2 minutes and visibly progressing:

```
/workspace/issue2546/logs/gen-post_greedy_a1.slot0.log  mtime=05:40:29
  [vllm-chunk] primary-post chunk 4/17 done key=4bc1149a... elapsed=199
  [vllm-chunk] primary-post chunk 5/17 (500 prompts)
... slot1 05:40:54 (chunk 5/17), slot2 05:42:41 (chunk 6/17), slot3 05:40:51 (chunk 5/17)
```

## Why this matters more than it looks

1. **The invisible lines are the anti-false-stall mechanism.** Those `[vllm-chunk]` lines exist BECAUSE of the #664 large-batch deadlock fix, whose gotchas entry states the per-chunk INFO log is "LOAD-BEARING: the poller's stall conjunction (logs >900 s + GPUs idle) trips only with no log activity, so per-chunk logs keep a long generation phase looking healthy." The mitigation is emitting them correctly; the poller just cannot see them. The fix and the detector are talking past each other.

2. **Stall detection degrades to a single signal.** With both log probes blind, the conjunction rests on `cpu_advancing`. That override holds for a GPU generation phase (parent + workers burn CPU), so no false stall fired here — but the designed redundancy is gone, and a phase that is genuinely quiet on CPU while healthy would false-stall.

3. **The crash-signature tail is lost.** #791 added tail surfacing precisely so a run arm writing ONLY to a per-phase log still yields a diagnosable tail. For this layout there is no tail: diagnosis requires a manual SSH per pod, which is what the orchestrator had to do here on a three-pod issue.

## Fix (proposed; implementer to confirm shape)

Add the out-root layout to `shard_log_probe`'s glob list, keeping the exact-issue-number discipline the existing comments insist on:

```
/workspace/issue<N>/logs/*.log
/workspace/issue<N>_*/logs/*.log
```

Both keep the issue number exact (no bare `issue<N>*`, which would let issue 5 match issue 521 — the stated #488/#521 rationale). The existing comments say "No glob change — reuse the existing narrow pattern so a cross-pod log on shared FS can never pollute the tail"; that conservatism is satisfied here because the issue number is in the path, exactly as for the current five.

Consider also normalizing the underscore inconsistency: the log globs accept only `issue_<N>` while the disk probe accepts both. Either accept both everywhere or state in one place why logs are underscore-only.

Worth weighing at implementation time: an ever-growing glob list is the reactive pattern this file's own history shows (#468, then #488/#521, then eval_results, now this). A registry of out-root log locations — or having dispatchers declare their log root in the handle sidecar the poller already reads — would end the pattern rather than extend it. That is a larger change; the glob addition is the minimal correct fix and should not be blocked on it.

## Scope

- `scripts/poll_pipeline.py` — `shard_log_probe` glob list (+ the docstring layout inventory at ~:51-58 and ~:254-278, which enumerates the known layouts and would otherwise go stale).
- A regression test asserting the composed probe string contains the out-root patterns, and that `issue<N>` cannot match a longer issue number (the #521 collision guard).
- No behavior change for single-log dispatchers: an added glob that matches nothing expands to nothing under the existing `shopt -s nullglob`.

## Provenance

Found by the `/issue 2546` orchestrator verifying forward progress after a tick showed a 973 s quiet top-level log with both per-phase probes blind. Distinct from #2599 (state-cache keyed on issue alone, same file): that is state CORRUPTION across concurrent pollers, this is log DISCOVERY for one poller. Same file, different bug, per the workflow-fix-on-bug distinct-bug rule.
