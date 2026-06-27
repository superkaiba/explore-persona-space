---
title: 'workflow-fix: GCP janitor must sweep whole project, not just eps-issue-*'
kind: infra
tags:
- wf-fix
created_at: '2026-06-27T20:34:05Z'
has_clean_result: false
origin_prompt: 'Verify GCP route working; leftover eps-cap-probe2 ran ~20h because
  the janitor only filters eps-issue-*. User: ''Also fix whatever workflow bug allowed
  it to keep running.'''
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow bug the
orchestrator hit directly during a GCP-route verification session (no parent
task — chat-mode work). An ad-hoc GCP VM (`eps-cap-probe2-1786331`, a leftover
#680 flex-start capacity probe) ran **~20h unbounded** in the dedicated GPU
project, burning ~$14 in credits and holding an L4 quota slot, because the
stale-VM janitor cron never saw it.

## Goal

Make the GCP stale-VM janitor a true project-wide credit-leak backstop: it
must catch and reap (or at minimum escalate) stale instances in the dedicated
`eps-persona-gpu-jun2026` project regardless of name, not only those named
`eps-issue-*`.

## Workflow gap

- **Bug observed:** The janitor cron (`scripts/cron_gcp_audit.sh` →
  `scripts/gcp_audit.py` → `backends.gcp.audit_stale_gcp_vms`) lists instances
  with `_AUDIT_NAME_FILTER = "name~^eps-issue-"`. Any instance in the project
  not named `eps-issue-*` (an ad-hoc probe, a manually-created dev VM) is
  structurally invisible to both reap predicates (24h age backstop +
  10-min terminal-phase reap) and runs forever.
- **Why it is a workflow gap:** `gcp_audit.py` and
  `.claude/rules/background-automation.md` both describe this cron as *"the
  credit-leak backstop"* for the dedicated GPU project, but its name filter
  scopes it to router-MANAGED names only — exactly the instances least likely
  to leak (they already carry `--max-run-duration` DELETE + an EXIT-trap).
  The leak class it should catch (un-managed, never-auto-deleting VMs) is the
  one it filters OUT. Confirmed live: `eps-cap-probe2-1786331` ran ~20h and
  the janitor would never have reaped it.
- **Confidence (emitter):** high (root cause directly observed + filter line
  identified).

## Proposed change (candidate diff sketch — refine in planning)

Separate the two concerns that currently share one name filter:

1. **Router-managed reaper / reconnect / name-reclaim** — KEEP scoped to the
   exact `eps-issue-<N>` namespace. The router must only ever auto-delete
   instances it owns; do NOT broaden `reconnect_or_none`, the stale-name
   reclaim, or any router-internal path.
2. **Janitor cron (the fleet-wide backstop)** — broaden its inventory to ALL
   instances in the dedicated project (drop the `^eps-issue-` restriction on
   the janitor's own list), then apply the SAME bounded reap fences
   (24h age; terminal-phase). Because the project is dedicated to EPS GPU
   work, any instance older than the age floor is a leak candidate.

Safety design the planner/critic should settle:
- Auto-DELETE non-`eps-issue-*` stale VMs vs WARN-and-escalate (Telegram +
  sidecar, mirroring the disk-guard active-task escalation pattern) vs a
  hybrid (auto-reap a known ephemeral prefix allowlist e.g. `eps-cap-probe*`,
  escalate the rest). Lean toward at least escalation so nothing is ever
  silently invisible again; auto-reap for clearly-ephemeral names.
- An optional name allowlist for any long-lived instance that legitimately
  should NOT be reaped (none today, but the design should not hard-assume
  "everything is disposable" without an opt-out).
- Keep the list-preflight disarmed-janitor alarm intact (it must run against
  whatever the broadened inventory query is).

Rough shape:

```
- _AUDIT_NAME_FILTER = "name~^eps-issue-"        # janitor scope == router scope (BUG)
+ # Janitor backstop sweeps the WHOLE dedicated project, not just eps-issue-*.
+ _AUDIT_NAME_FILTER = None                       # or a broadened/allowlisted filter
  # ... audit_stale_gcp_vms gains a managed-vs-unmanaged classification so
  #     un-managed stale VMs are reaped-or-escalated, while the router's own
  #     reconnect/reclaim paths stay pinned to eps-issue-<N>.
```

## Scope / surfaces

- Primary targets: `scripts/gcp_audit.py`,
  `src/explore_persona_space/backends/gcp.py` (the `audit_stale_gcp_vms`
  reaper + `render_list_argv`).
- Also update: `.claude/rules/background-automation.md` (the documented
  janitor scope) and `scripts/cron_gcp_audit.sh` if its messaging assumes
  `eps-issue-*`.
- Grep before editing: `grep -rn 'eps-issue-' scripts/gcp_audit.py
  src/explore_persona_space/backends/gcp.py .claude/rules/background-automation.md`
  and distinguish router-managed uses (keep) from janitor-scope uses (broaden).
- Add/extend tests: `tests/test_gcp_backend.py` — a non-`eps-issue-*` stale VM
  is reaped-or-escalated by the janitor path; `eps-issue-<N>` router paths
  unchanged.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The router-managed reconnect / name-reclaim path stays scoped to
  `eps-issue-<N>` (do not let the janitor's broadening leak into instances the
  router auto-deletes by name).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; if `background-automation.md` changes, it stays consistent with the
  code.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/gcp_audit.py,src/explore_persona_space/backends/gcp.py
- fingerprint: 49104de5e2d8

<!-- workflow-fix-candidate v1 -->
target_file: scripts/gcp_audit.py,src/explore_persona_space/backends/gcp.py
bug_observed: The GCP stale-VM janitor only inspects eps-issue-* VMs (name~^eps-issue- filter), so the ad-hoc eps-cap-probe2 capacity probe escaped the backstop and ran ~20h unbounded burning credits.
why_workflow_gap: The janitor is documented as the project's credit-leak backstop but its name filter scopes it to router-managed eps-issue-* names only, leaving every other instance in the dedicated GPU project invisible to the 24h reaper.
proposed_change: Broaden the GCP janitor cron to inspect ALL instances in the dedicated eps-persona-gpu-jun2026 project (not just eps-issue-*), so non-eps-issue named VMs are caught by the 24h credit-leak backstop, while keeping the router-managed reconnect/reclaim path scoped to eps-issue-<N>.
diff_sketch: |
  - _AUDIT_NAME_FILTER = "name~^eps-issue-"   # janitor scope == router scope (BUG)
  + _AUDIT_NAME_FILTER = None                 # janitor backstop sweeps the whole dedicated project
  # audit_stale_gcp_vms classifies managed (eps-issue-*) vs un-managed and
  # reaps-or-escalates un-managed stale VMs; router reconnect/reclaim stays eps-issue-<N>.
confidence: high
related_task: n/a (orchestrator chat-mode observation)
<!-- /workflow-fix-candidate -->
