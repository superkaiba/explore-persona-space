---
title: 'workflow-fix: GCE crash-persist must verify uploads before persist=ok'
kind: infra
tags:
- wf-fix
- wf-fix-fp:04a11d9d274c
created_at: '2026-07-15T10:39:49Z'
has_clean_result: false
origin_prompt: 'wf-fix candidate from /issue 1315: eps/persist=ok with zero files
  landed under issue1315_partial/ — per-file || true guards mask total upload failure
  in _eps_persist_diagnostics (backends/gcp.py)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1315 (emitting agent: orchestrator, /issue 1315 session).

## Goal

Make the GCE crash-persist status honest: set `eps/persist=ok` only when at least one diagnostics upload verifiably succeeded (existence-check the transcript upload; else `persist=failed_uploads`), and surface per-upload failures to serial instead of silently swallowing them with `|| true`.

## Workflow gap

- **Bug observed:** GCE crash-persist set `eps/persist=ok` while ZERO files (including `crash_persist_transcript.log`) landed under `issue1315_partial/` on the data repo — total upload failure masked by per-file `|| true` guards.
- **Why it is a workflow gap:** the persist-status contract exists so a crashed GCP run's diagnostics are recoverable (#658/#854 crash-diagnostics guarantee); `ok`-on-total-failure defeats it — on #1315 (2026-07-15, instance eps-issue-1315, attempt att-20260715-095826) the crash evidence had to be recovered by a manual diagnostic boot of the stopped instance (startup-script neutralize + boot-disk read), ~40 min of orchestrator time and an extra GPU-instance start.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "persist_status\|_eps_persist_diagnostics" src/explore_persona_space/backends/gcp.py` → 8 hits in 1 file: the status wrapper branches on subshell rc only (`(0) _eps_persist_status "ok"` at gcp.py:1948) while the per-file uploads inside `_eps_persist_diagnostics` (defined gcp.py:1615+) are individually `|| true`-guarded, so rc 0 does not prove any file landed (2026-07-15). HF-side absence verified live: `list_repo_tree(..., path_in_repo="issue1315_partial")` → 404 while guest attr read `persist=ok`.

## Proposed change (candidate diff sketch — refine in planning)

```
  # in _eps_persist_diagnostics, after the final transcript upload:
- (0)   _eps_persist_status "ok" ;;
+ # verify at least the transcript landed before claiming ok, e.g.:
+ # _ok=$(uv run python -c "from huggingface_hub import HfApi; print(HfApi().file_exists('$EPS_HF_DATA_REPO', '$_dest/crash_persist_transcript.log', repo_type='dataset'))" 2>/dev/null || echo False)
+ (0)   if [ "$_ok" = "True" ]; then _eps_persist_status "ok"; else _eps_persist_status "failed_uploads"; fi ;;
  # and stream each upload's failure line to serial (>&3) instead of bare || true
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'persist_status' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every hit;
  list them in the plan. Also update the poller/rule prose that treats `persist=ok` as
  proof-of-upload (`.claude/rules/compute-backend-failover.md` § done-persist disambiguation)
  if the status vocabulary gains `failed_uploads`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The persist path must stay fully guarded + bounded (#854) — the verification probe
  must never delay the poweroff unboundedly (keep it inside the existing 300s bound).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 04a11d9d274c

bug_observed: GCE crash-persist set eps/persist=ok while zero files (including crash_persist_transcript.log) landed under issue1315_partial/ on the data repo - total upload failure masked by per-file || true guards
why_workflow_gap: ok-on-total-failure defeats the #658/#854 crash-diagnostics guarantee; #1315 needed a manual diagnostic boot to recover the traceback
proposed_change: set eps/persist=ok only when at least one diagnostics upload verifiably succeeded (existence-check the transcript upload; else persist=failed_uploads), and surface per-upload failures to serial instead of silent || true
confidence: high
related_task: #1315
