---
title: 'preflight branch-freshness check: tolerate transient git-fetch timeouts (#664
  false-positive recurrence)'
kind: infra
tags: []
created_at: '2026-06-28T05:45:25Z'
has_clean_result: false
---
## Goal

Make `src/explore_persona_space/orchestrate/preflight.py`'s branch-freshness check tolerate transient `git fetch origin` timeouts so a slow-network pod is not falsely reported as `ok=false`. Surfaced as a recurring false-positive during the #664 r9 recovery relaunch (2026-06-28) — the experimenter had to independently re-verify branch currency to push through a transient fetch-timeout ERROR on an otherwise-healthy pod.

## Repro

On a freshly-bootstrapped RunPod pod with intermittent outbound HTTPS latency, `uv run python -m explore_persona_space.orchestrate.preflight --json` reports `ok=false` with `errors: ["git fetch origin failed (timeout) — cannot verify ..."]`. Direct `git fetch origin issue-664 --depth=1` from the SAME pod completes with rc=0 within seconds; `git rev-parse HEAD == origin/issue-664` confirms the branch is up-to-date.

## Proposed fix

In the branch-freshness check inside `preflight.py`:

1. On fetch timeout, retry the fetch ONCE with a longer timeout (e.g. 60s vs the current short timeout).
2. If the retry STILL times out but a prior remote-tracking ref (`origin/<branch>` from a previous fetch) exists AND `behind_count_against_last_known_ref == 0`, downgrade the ERROR to a WARNING: "git fetch timed out; last-known origin ref shows branch up-to-date".
3. Keep ERROR semantics only when the branch is verifiably behind OR no last-known ref exists at all.

## Acceptance criteria

- Preflight returns `ok=true` on a pod with a transient fetch timeout AND a verifiably-current checkout against the last-known origin ref.
- A genuinely-stale checkout (or no last-known ref) still reports `ok=false` with a clear ERROR.
- The retry adds at most ~60s wall-clock to preflight on the slow-network case.
- Unit test added covering both cases (mock fetch timeout + currency-check pass; mock fetch timeout + behind-count > 0).

## Out of scope

- Other preflight checks (HF reachability, disk, GPUs) — only the branch-freshness fetch is in scope.
- Removing the branch-freshness check entirely — keep it; just make it timeout-tolerant.

## Provenance

Surfaced by the `experimenter` agent on the #664 recovery relaunch (2026-06-28). Full candidate block in `epm:workflow-fix-candidate v1` on task #664. Target file is under `src/explore_persona_space/orchestrate/` (not in the named workflow-surface `src/` modules), so the proper home is an infra task that PM's auto-dispatch picks up rather than `workflow-improver`.
