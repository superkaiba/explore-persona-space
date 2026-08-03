---
title: 'daily-fix: fellows stall reads need grace window'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d1422db8efa0
- daily-auto-filed
created_at: '2026-08-01T07:06:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Healthy just-launched fellows
  jobs repeatedly classified `stalled` (#1900); every read needed manual ssh verification
  — no grace window / consecutive-tick requirement beyond the run-age floor.'
workflow: v1
---
# daily-fix: fellows stall reads need grace window

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M6; miner-4:P7). Source: session 879efc0d (#1900) — the fellows/SLURM backend poller repeatedly classified healthy just-launched fellows jobs as `stalled` ("Same early `stalled` read as last time"; "The poller's `stalled` reads have been unreliable on this lane all session — verifying live"), forcing a manual ssh live-state verification before each read could be trusted. Actual job deaths in the same session were separately confirmed from logs, so these were false/premature classifications.

## Goal

Make the SLURM monitor's `stalled` classification reliable for just-launched fellows-lane jobs (grace window or phase/queue-state-aware predicate) so the orchestrator does not have to manually ssh-verify every early stall read.

## Workflow gap

- **Bug observed:** Healthy, recently-launched fellows jobs were repeatedly classified `stalled` by the poll loop in session 879efc0d (#1900); every such read required a manual live-state ssh verification before proceeding.
- **Why it is a workflow gap:** The stall predicate is constructed in `src/explore_persona_space/backends/slurm_monitor.py` — `STALL_SEC = 300` (line ~189) and `base_status = "stalled"` when `base_status == "running" and heartbeat_sec_ago > STALL_SEC` (line ~664-666). The only protections are the C2 run-age floor (heartbeat staleness capped at RUN age) and the module docstring's DOCUMENTED weakening ("a job that writes nothing to status.json ... shows as `stalled` until SLURM itself reaps it", lines 41-48). There is no fellows-lane grace window, no phase-aware predicate (e.g. a `startup`/staging phase read from status.json), and no repeated-consecutive-tick requirement before reporting `stalled` — one stale-looking tick on a young RUNNING job reads as a stall.
- **Call-hop target correction:** the CONSOLIDATED entry named `scripts/backend_poll.py` / `backends/slurm.py`; the classification is CONSTRUCTED in `src/explore_persona_space/backends/slurm_monitor.py` (`backend_poll.py` hits are consumer/doc references; `slurm.py`'s one hit is the sbatch-prelude comment). Primary target corrected accordingly.
- **Known mitigations already in place (planner should start from these):** (1) the sbatch prelude starts the heartbeat loop BEFORE the venv build precisely to avoid the ~6-40 min startup false-stall (`slurm.py` ~1805-1818: "Start the heartbeat NOW (before the long venv build)"); (2) the C2 run-age stall-clock floor (`slurm_monitor.py` ~654-662); (3) the #1836 writer-unique `_write_status` tmp fix for the heartbeat/phase-writer mv race landed 2026-07-30 14:32 -0400 (commit `8da8a426ef`) — BEFORE the 2026-07-31 session, so it does not explain these false reads. `unverified hypothesis — verify at plan time:` the residual mechanism on the fellows lane (candidates: status.json read transport latency over `ssh charmander`, the drain-rename `.processed` interaction, heartbeat interval vs STALL_SEC margin under cluster FS lag, or a RunTime-absent UNKNOWN-tick path) — the exact predicate change needs the module open with a live-session repro.
- **Confidence (emitter):** medium (behavior session-reported; construction site probed; mechanism unresolved)
- verified-at-filing: `grep -n "stalled" scripts/backend_poll.py src/explore_persona_space/backends/slurm.py src/explore_persona_space/backends/slurm_monitor.py` → slurm_monitor.py 10 hits incl. the construction site (`STALL_SEC` ~189, predicate ~666); backend_poll.py 5 hits (consumer/doc only); slurm.py 1 hit (prelude comment). Context read of slurm_monitor.py 640-690 confirms no grace window / consecutive-tick requirement beyond the run-age floor. `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/slurm_monitor.py` → 2 commits (`1ab3630cf1` done-evidence disambiguation, `3606a80892` sentinel-drain arm) — neither adds a stall grace window; no landed fix (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

```
src/explore_persona_space/backends/slurm_monitor.py:
+ # Fellows/queue-state-aware stall grace (#1900 false-positive class):
+ STALL_MIN_RUN_AGE_SEC = 600   # never flag `stalled` in the first N s of RUN time
+ STALL_CONSECUTIVE_TICKS = 2   # require >=2 consecutive stale ticks before reporting
- if base_status == "running" and heartbeat_sec_ago > STALL_SEC and slurm_status != "PENDING":
-     base_status = "stalled"
+ if (base_status == "running" and heartbeat_sec_ago > STALL_SEC
+         and slurm_status != "PENDING"
+         and (run_age_sec is None or run_age_sec >= STALL_MIN_RUN_AGE_SEC)):
+     base_status = "stalled"   # + thread a consecutive-tick counter via poll state
```
(Exact shape — min-run-age floor vs consecutive-tick accumulation vs phase-aware read of status.json `phase` — is a planning decision; the plan should first reproduce one false read to pin the mechanism.)

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm_monitor.py`
- Secondary: `scripts/backend_poll.py` (consumer — only if a consecutive-tick counter must live in poll state), `tests/test_slurm_*.py` (pin test for the new predicate)
- Grep the workflow surface for the pattern before editing (`grep -rn 'STALL_SEC\|"stalled"' src/explore_persona_space/backends/ scripts/backend_poll.py`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; the module docstring's stall-semantics section stays consistent with the new predicate.
- Must NOT delay detection of genuinely dead jobs past the documented early-init-crash weakening (the `[phase=preflight-failed]` disambiguation stays authoritative).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: d1422db8efa0

- workflow_fix_target: src/explore_persona_space/backends/slurm_monitor.py
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED M6 (miner-4:P7), /daily 2026-07-31 — "Fellows/SLURM backend poller returned unreliable early 'stalled' reads all session — each required manual live-state verification" (session 879efc0d / #1900).
