---
title: 'workflow-fix: SLURM sbatch _write_status heartbeat/phase-writer tmp race (BASHPID-unique
  tmp)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b23785c9eab5
created_at: '2026-07-29T18:57:34Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1689 (2026-07-29, fellows job 15192
  FAILED at preflight): _write_status''s fixed STATUS_JSON.tmp is shared by the background
  heartbeat loop and the main phase writers; the losing mv fails and set -eu kills
  the job — make the tmp writer-unique (BASHPID/mktemp).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1689 (emitting agent: issue-orchestrator).

## Goal

Make the SLURM sbatch template's `_write_status` tmp path writer-unique so the
background heartbeat loop and the main script's phase writers never collide on
the atomic rename.

## Workflow gap

- **Bug observed:** sbatch template `_write_status` uses the fixed tmp path
  `${STATUS_JSON}.tmp` shared by the background heartbeat loop and the main
  phase writers; a concurrent printf+mv race under `set -eu` killed fellows
  job 15192 at preflight (`mv: cannot stat
  '/workspace/superkaiba/eps/issue-1689/status.json.tmp': No such file or
  directory`, 2026-07-29T18:54:52Z — one full launch cycle burned on a healthy
  workload that never started).
- **Why it is a workflow gap:** `backends/slurm.py`'s rendered sbatch starts
  `_heartbeat_loop` (a background subshell calling `_write_status` every
  `HEARTBEAT_INTERVAL`) BEFORE the venv build, and the main script calls the
  same `_write_status` at every phase transition. Both writers printf to the
  SAME `${STATUS_JSON}.tmp` and `mv` it over `${STATUS_JSON}`: when the two
  interleave (printf-A, mv-B steals the tmp, mv-A finds nothing), the loser's
  `mv` fails and `set -eu -o pipefail` kills the job. The window is tightest
  when the venv is cached (preflight follows the initial heartbeat within
  seconds — exactly the 15192 shape); the race is probabilistic and had passed
  on the three prior #1689 launches (15164/15166/15188) the same day.
- **Confidence (emitter):** high (race read directly off the set -x job.out
  trace + the template source at `backends/slurm.py` ~L1531-1557; the trap's
  `kill 857941` in the trace is the heartbeat pid).
- verified-at-filing: `grep -n "STATUS_JSON}.tmp\|_write_status\|_heartbeat_loop" src/explore_persona_space/backends/slurm.py`
  → `_write_status` definition at ~L1531 with `local tmp="${STATUS_JSON}.tmp"`
  at ~L1542, `_heartbeat_loop` calling it at ~L1554, phase-writer call sites at
  ~L1565/1695/1768/1891 — single template file, fixed-tmp confirmed present
  (2026-07-29). Note `$$` is NOT a sufficient suffix: the heartbeat runs as a
  background function in the same shell where `$$` matches the parent —
  `$BASHPID` (or `mktemp`) is required for writer-uniqueness.

## Proposed change (candidate diff sketch — refine in planning)

```
  # backends/slurm.py, _write_status template lines (~L1542)
- '  local tmp="${STATUS_JSON}.tmp"',
+ '  local tmp="${STATUS_JSON}.tmp.${BASHPID}"',
  # mv "$tmp" "$STATUS_JSON" stays — rename remains atomic; each writer now
  # renames its OWN tmp. Optionally `|| true` the mv is NOT taken (a real
  # write failure should still fail loud); consider `command mv -f`.
  # Add a rendered-template regression test asserting the tmp suffix.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rn "STATUS_JSON" src/ scripts/ tests/`) and update every hit
  (renderer tests pinning the sbatch body may need the new suffix).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: b23785c9eab5

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/slurm.py
bug_observed: sbatch template _write_status uses the fixed tmp path STATUS_JSON.tmp shared by the background heartbeat loop and the main phase writers; concurrent printf+mv race under set -eu killed fellows job 15192 at preflight (mv: cannot stat status.json.tmp)
why_workflow_gap: the rendered sbatch runs two concurrent writers of one status file through a single shared tmp path; the losing mv fails and set -eu kills an otherwise-healthy job before the workload starts — a probabilistic lane-template race every SLURM launch is exposed to
proposed_change: make _write_status tmp path writer-unique (BASHPID suffix or mktemp) so heartbeat and phase writers never collide on the atomic rename
diff_sketch: |
  - '  local tmp="${STATUS_JSON}.tmp"',
  + '  local tmp="${STATUS_JSON}.tmp.${BASHPID}"',
confidence: high
related_task: #1689
<!-- /workflow-fix-candidate -->
