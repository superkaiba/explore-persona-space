---
title: 'workflow-fix: gate _eps_phase done on OOM-detection (GCP)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c6c9455b338f
created_at: '2026-06-30T05:47:35Z'
has_clean_result: false
origin_prompt: 'Auto-filed from /issue 744 orchestrator-self-observation: workload
  python OOM-killed at Phase 1 broader pass1 7350/7389 (2026-06-30T04:34:12Z), yet
  GCP startup-script published phase=done 2s later, masking the failure. See epm:failure
  v1 + workflow-fix-candidate block on task #744.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #744 (emitting agent: orchestrator-self-observation). The GCP backend's startup-script success path published `_eps_phase done` even though the workload bash chain experienced an OOM-kill, masking a complete workload failure as success and stranding the orchestrator at `verifying` with no usable artifacts.

## Goal

Harden the GCP backend's startup-script success path so `_eps_phase done` is never published when the workload bash chain rc-survived an OOM-kill or other SIGKILL — gate the explicit `_eps_phase done` on a positive workload-rc check (cgroup `memory.events.oom_kill` counter == 0, AND the planner-named primary deliverable paths actually exist) so an OOM'd run routes via the EXIT trap's `phase=failed` branch.

## Workflow gap

- **Bug observed:** On eps-issue-744 (`g2-standard-4` / 1× L4, ~16 GB host RAM), Phase 1 `dump_and_stream.py` (workload python pid 1553, `total-vm:51673852kB`, `anon-rss:14983132kB`) was OOM-killed by the kernel at `broader pass1 7350/7389` at 2026-06-30T04:34:12Z. The serial console shows `oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=google-startup-scripts.service,mems_allowed=0,global_oom,task_memcg=/system.slice/google-startup-scripts.service,task=python3,pid=1553,uid=0`. systemd logged `google-startup-scripts.service: Failed with result 'oom-kill'` AND `Unit process 1155 (startup-script) remains running after unit stopped`. Despite that, at 2026-06-30T04:34:14Z (2s after the kill) the workload log shows `[phase=done] startup-script 2026-06-30T04:34:14Z` and the completion sentinel `/workspace/eps-issue-744/eval_results/issue_744/att-20260630-035706/.completion-sentinel.json` (`{"phase":"done","issue":744,"attempt_id":"att-20260630-035706"}`) was written. `gcloud compute instances get-guest-attributes eps-issue-744 --query-path=eps/phase` returned `done`. `scripts/backend_poll.py --issue 744` reported `status=done, current_phase=workload_done, pid_alive=false, new_milestone=true`. The orchestrator advanced to `status:verifying` even though zero usable artifacts exist on disk (no `broader_random_pairs.pt`, `broader_summaries.json`, `population_stats.pt`, `eval_results/issue_744/`, `figures/issue_744/`, no `instruct/` arm — only Phase 0 corpora + the partial NS dump).

- **Why it is a workflow gap:** The startup-script's success-path code at `src/explore_persona_space/backends/gcp.py` lines 1493–1515 unconditionally writes the completion sentinel and calls `_eps_phase done` once execution reaches that point. The EXIT-trap's failure branch at lines 1353–1365 fires only when `rc != 0`. The bash chain `spec.workload_cmd` (here `bash X && bash Y`) somehow returned `rc=0` to the parent startup-script even though its child python pid 1553 was SIGKILLed by the OOM-killer and `set -euo pipefail` is in effect — exact mechanism not yet pinned (the bash chain may have been in a state where the rc didn't propagate through the `&&` short-circuit + `set -e` semantics expected of it, or the OOM-killer's targeting only hit the python child while the bash chain races + ANOTHER subshell return path swallowed the failure rc). The async-poller failover predicate `GcpBackend._is_gcp_async_workload_failure` is gated on `current_phase == "terminal_workload_failed"`, so a `phase=done` posted on a failed workload bypasses every defense and the only signal that the run actually failed is the absence of artifacts at the upload-verification step — which then either prompts manual recovery (this incident) or worse, a downstream task silently inherits empty inputs. The `phase=done` posting is the bug; the rc-propagation mystery is downstream of it.

- **Confidence (emitter):** medium (direct empirical evidence the bug exists, including the kernel OOM trace + the contradictory `phase=done` posting + the empty artifact tree; the root-cause mechanism for how rc=0 emerged from the OOM'd chain is not fully diagnosed and will be the spawned `/issue --auto` session's planner's job).

## Proposed change (candidate diff sketch — refine in planning)

Inside `_render_startup_script()` in `src/explore_persona_space/backends/gcp.py`, between the existing workload_block and the existing `_eps_phase done` line:

```
+ # === Verify the workload actually succeeded BEFORE publishing done ===
+ # The bash chain's rc-propagation has known weakness modes (set -e + && +
+ # subshells/pipes) under SIGKILL. Cross-check the kernel's authoritative
+ # OOM signal AND the planner-named primary deliverable paths to refuse
+ # publishing done on a workload that the chain swallowed the rc for.
+ _eps_oom_seen() {
+   # cgroup v2: memory.events lists oom_kill counter. Any non-zero ⇒ the
+   # workload (or one of its children) was killed by the kernel oom-killer
+   # in this cgroup, regardless of what rc the bash chain claims.
+   for f in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.events.local; do
+     if [ -r "$f" ] && grep -qE '^oom_kill [1-9]' "$f"; then return 0; fi
+   done
+   # systemd journal fallback (some images expose v1)
+   journalctl -k --since "10 min ago" 2>/dev/null | grep -q "Out of memory: Killed process" && return 0
+   return 1
+ }
+ if _eps_oom_seen; then
+   echo "[startup-script] OOM-kill detected in cgroup despite rc=0 — routing to failed" >&3 2>/dev/null || true
+   exit 137   # force the EXIT trap's failed branch
+ fi
+ # OPTIONAL — additional defense: verify the planner-named primary
+ # deliverable paths (handle.extra.expected_artifacts.git_paths) contain
+ # at least one file before publishing done. A workload that emits done
+ # but produced no expected artifact has self-disagreed.

  cat > <sentinel> <<EOF
  {"phase":"done",...}
  EOF
  _eps_phase done
```

The `_eps_oom_seen` check leverages the kernel-authoritative cgroup v2 memory.events `oom_kill` counter (always set when an OOM-killer hit the cgroup, regardless of what shell rc the bash chain reported). Forcing `exit 137` re-routes through the existing EXIT trap's `rc != 0` branch, which already does the right thing (`_eps_phase failed` + `_eps_persist_diagnostics` + `shutdown`). The optional primary-deliverable-existence check is a secondary defense against non-OOM silent failures.

A test should land alongside this in `tests/test_gcp_backend.py` covering: (a) a rendered startup-script body containing the new `_eps_oom_seen` guard; (b) a simulated cgroup memory.events fixture showing `oom_kill 1` triggering exit 137 in the guard; (c) the existing happy-path rendering (no OOM, no exit, _eps_phase done lands) still passing byte-identical to the pre-change baseline.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (the only file where `_eps_phase done` is rendered into the startup-script body — confirmed via `grep -rln '_eps_phase done'` over `.claude/ CLAUDE.md scripts/ src/`; the only OTHER workflow-surface hit is `tests/test_gcp_backend.py` for the regression test).
- Grep the workflow surface for the pattern before editing (`grep -rln '_eps_phase done' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`); update every hit if any new ones appear; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with this rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (auto-set from the `workflow_fix_target:` Provenance line below) and MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).
- The new guard MUST NOT add wall-time to the happy path (one cgroup-events file read + one journalctl fallback is ~ms).
- The new guard MUST be idempotent on systems where cgroup v2 isn't exposed at the expected paths (a missing file is "no OOM seen", which preserves the pre-change happy path).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: c6c9455b338f

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py
bug_observed: On eps-issue-744 (g2-standard-4 / 1xL4), the workload python (pid 1553, 49 GB virt, 14 GB RSS) was OOM-killed at broader pass1 7350/7389 (2026-06-30T04:34:12Z); systemd logged 'Failed with result oom-kill'; yet the startup-script wrote the completion sentinel + published phase=done 2s later, the async poller reported status=done with no artifacts produced (no broader_random_pairs.pt, broader_summaries.json, population_stats.pt, eval_results/issue_744, figures/issue_744, no instruct arm).
why_workflow_gap: The startup-script's success-path block at gcp.py:1493-1515 unconditionally writes the completion sentinel + calls _eps_phase done once reached. The EXIT-trap's failure branch (gcp.py:1353-1365) fires only on rc != 0. Bash chain rc-propagation under set -e + && + SIGKILL has weakness modes that allowed the OOM'd chain to return rc=0 to the parent startup-script. The async-poller failover predicate _is_gcp_async_workload_failure gates on terminal_workload_failed, so a phase=done posted on a failed workload bypasses every existing defense.
proposed_change: Insert a kernel-authoritative OOM-detection guard (`_eps_oom_seen` reading cgroup memory.events oom_kill counter) immediately before the sentinel-write + _eps_phase done lines, forcing exit 137 on any detected OOM kill so the EXIT trap's failed branch fires correctly. Optional secondary defense: verify planner-named primary deliverable paths exist.
diff_sketch: |
  + _eps_oom_seen() {
  +   for f in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.events.local; do
  +     [ -r "$f" ] && grep -qE '^oom_kill [1-9]' "$f" && return 0
  +   done
  +   journalctl -k --since "10 min ago" 2>/dev/null | grep -q "Out of memory: Killed process" && return 0
  +   return 1
  + }
  + if _eps_oom_seen; then
  +   echo "[startup-script] OOM-kill in cgroup despite rc=0 — routing to failed" >&3 2>/dev/null || true
  +   exit 137
  + fi
    cat > <sentinel> ...
    _eps_phase done
confidence: medium
related_task: #744
<!-- /workflow-fix-candidate -->
