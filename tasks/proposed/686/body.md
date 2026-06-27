---
title: 'workflow-fix: confirm-deleted guard in GcpBackend.teardown (#683 RUNNING zombie
  post phase=done)'
kind: infra
tags: []
created_at: '2026-06-27T07:36:33Z'
has_clean_result: false
parent_id: 683
origin_prompt: Auto-filed by /issue 683 workflow-fix-on-bug protocol after upload-verifier
  emitted workflow-fix-candidate v1; fingerprint=a1eff6554206; target_file=src/explore_persona_space/backends/gcp.py
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #683 (emitting agent: upload-verifier).

## Goal

After a GCP poller resolves a clean terminal (`eps/phase=done`), actively confirm the instance is GONE via `gcloud compute instances describe`; if still `RUNNING` past a short floor, issue an explicit delete rather than trusting the EXIT-trap alone.

## Workflow gap

- **Bug observed:** GCP instance `eps-issue-683` reached `eps/phase=done` + uploaded all artifacts but stayed `RUNNING` (FLEX_START `--instance-termination-action=DELETE` teardown trap never fired), silently billing an A100-80 ~40+ min past terminal, caught only because upload-verification cross-checked the live gcloud API against the brief's "already deleted" assumption.
- **Why it is a workflow gap:** the success/`[phase=done]` path relies solely on the in-instance EXIT-trap DELETE; when that trap silently no-ops there is no orchestrator-side confirm-deleted check after a clean run, so a terminal-but-RUNNING zombie persists until the once-daily 09:37 `gcp_audit` cron reaps it (up to ~24h of A100-80 billing for a successful run).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/backend_poll.py or src/.../backends/gcp.py — on clean terminal:
+ # EXIT-trap DELETE can silently no-op (#634/#683 zombie); confirm + force-delete
+ if gcp.instance_status(name, zone) == "RUNNING" and gcp.guest_phase(name) == "done":
+     gcp.delete_instance(name, zone)   # idempotent; mirrors gcp_audit reap predicate
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Secondary surface: `scripts/backend_poll.py` (per the candidate; the planner may relocate the fix to `GcpBackend.teardown` if that is the canonical site — see § "Reference draft" below).
- Grep the workflow surface for the pattern before editing:
  `grep -rln 'instance-termination-action\|gcp_audit\|workload_done\|EXIT-trap' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`

## Reference draft (do NOT auto-merge; for the planner / implementer's reference)

A draft fix already exists on branch `worktree-agent-a5658a4f7cd8ba513` @ `151ca047a44904c974f4024a96ca7ed069e15764` (produced by the now-retired `workflow-improver` path before the orchestrator rerouted through the #678 protocol). The draft:

- Places `_confirm_deleted` INSIDE `GcpBackend.teardown` (NOT `backend_poll.py`), arguing the poller is a read-only status reporter and the delete contract lives in `teardown` (called by `dispatch_issue.py finalize`).
- Fail-toward-gone discipline: 404 describe → confirmed gone; non-404 probe failure → trust the rc==0 delete (the janitor remains the backstop); status != RUNNING → trust the delete; status == RUNNING → re-issue delete ONCE; redelete 404 → silent success.
- 4 new tests in `tests/test_gcp_backend.py` (confirmed-gone, RUNNING-zombie-redelete, non-RUNNING-no-redelete, non-404-probe-failure-no-redelete).
- 678 backend tests pass; ruff clean; `test_no_dollar_budget_caps.py` PASS.

The planner / implementer should consider whether the `GcpBackend.teardown` placement is sound vs the poller placement, and EITHER cherry-pick the draft OR re-derive their own. The autonomous session has full freedom; this is reference only.

There is also one follow-up surfaced by the draft author:
- **`dispatch_issue.py finalize` / `autonomous_session_watch.py` pod-safety pass** — the zombie can also occur when the orchestrator never reaches Step 8 (crash between `done` and finalize), in which case the `teardown` guard never runs. A watcher-side predicate that treats "GCP handle with `workload_done` poll but instance still RUNNING past a short floor" as a teardown-owed signal is the structural complement. Out of scope for this task; file separately if the planner judges it material.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, § Recursion guard in `.claude/rules/workflow-fix-on-bug.md`).
- The fix is **idempotent** (calling delete on already-deleted must not raise loudly).
- The fix does NOT introduce a new orchestrator-side cost gate (per CLAUDE.md `tests/test_no_dollar_budget_caps.py`).
- The fix is scoped to GCP — it should not affect RunPod or SLURM lanes (those have their own teardown semantics).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: a1eff6554206

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py, scripts/backend_poll.py
bug_observed: GCP instance eps-issue-683 reached eps/phase=done + uploaded all artifacts but stayed RUNNING (FLEX_START --instance-termination-action=DELETE teardown trap never fired), silently billing an A100-80 ~40+ min past terminal, caught only because upload-verification cross-checked the live gcloud API against the brief's "already deleted" assumption.
why_workflow_gap: the success/`[phase=done]` path relies solely on the in-instance EXIT-trap DELETE; when that trap silently no-ops there is no orchestrator-side confirm-deleted check after a clean run, so a terminal-but-RUNNING zombie persists until the once-daily 09:37 gcp_audit cron reaps it (up to ~24h of A100-80 billing for a successful run).
proposed_change: after the GCP poller resolves a clean terminal (eps/phase=done), actively confirm the instance is GONE via gcloud describe; if still RUNNING past a short floor, issue an explicit instances delete (mirror the #634 zombie-reap predicate already in gcp_audit.py) rather than trusting the EXIT-trap alone.
diff_sketch: |
  # backend_poll.py, on clean terminal resolution for a GCP run:
  + # EXIT-trap DELETE can silently no-op (#634/#683 zombie); confirm + force-delete
  + if gcp.instance_status(name, zone) == "RUNNING" and gcp.guest_phase(name) == "done":
  +     gcp.delete_instance(name, zone)   # idempotent; mirrors gcp_audit reap predicate
confidence: medium
related_task: #683
<!-- /workflow-fix-candidate -->
