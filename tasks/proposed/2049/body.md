---
title: 'workflow-fix: list-ephemeral hides duplicate-named pods (name-keyed state)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:00425f750fed
created_at: '2026-08-03T18:25:12Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed during #1739 armfill round: three RUNNING/EXITED
  pods named pod-1739 on the live API, only the EXITED one shown by pod.py list-ephemeral;
  ~$8/hr invisible.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a bug hit during a #1739
inline round (emitting agent: orchestrator, PA chat session, 2026-08-03).

## Goal

Key the list-ephemeral state dict on pod_id instead of pod name (or emit one row per live API pod), so duplicate-named pods are all listed

## Workflow gap

- **Bug observed:** pod.py list-ephemeral collapses duplicate-named pods into one row because _load_state() is keyed by pod NAME, hiding RUNNING pods behind an EXITED sibling and masking live spend from the operator listing
- **Why it is a workflow gap:** `pod.py list-ephemeral` is the operator's
  primary view of live pod spend. On 2026-08-03 the live team API showed THREE
  pods named `pod-1739` — `3ysuovu14p879a` (RUNNING 1xH100, the live round),
  `ogaqj4df250xjh` (RUNNING 1xH100, no marker provenance anywhere), and
  `yib7yjxjvmx6iz` (EXITED) — while `list-ephemeral --issue 1739` printed
  ONLY the EXITED one. ~$8/hr of real burn was invisible to the listing, and
  the discrepancy only surfaced because a spend-cap preflight (which reads the
  live API) refused a provision. `_load_state` even has an explicit
  "API only (no metadata)" synthesis branch, so the pods are FETCHED — they are
  lost at the dict-keying step, not at fetch.
- **Confidence (emitter):** high
- verified-at-filing: `sed -n '<def _load_state>,+18p' scripts/pod_lifecycle.py`
  -> returns `dict[str, EphemeralPod]` (6 occurrence(s) of that annotation);
  `cmd_list_ephemeral` (scripts/pod_lifecycle.py:3094) iterates
  `_load_state().values()`, one row per KEY. Empirically confirmed against the
  live team API the same hour: 3 pods named pod-1739 (2 RUNNING) -> 1 row printed
  (the EXITED one). (2026-08-03)

## Proposed change (candidate diff sketch — refine in planning)

- Key the merged state on `pod_id` (unique) rather than pod name; keep name as
  a display column. Alternatively emit one row per live API pod matching the
  managed prefixes, with name collisions disambiguated by id.
- Consider a loud WARN row when >1 live pod shares a managed name, since that
  itself indicates a provisioning-idempotency problem.
- Add a regression test: two live API pods sharing one managed name must both
  appear in the listing.

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py` (`_load_state`, `cmd_list_ephemeral`)
- Check `scripts/pod_audit.py` / `pod.py audit-stale` for the same name-keyed
  assumption before editing.

## Constraints / invariants

- Workflow-surface only. Live API stays authoritative for pod state.
- Must not change terminate/stop targeting semantics (surgical `--name-suffix`
  behavior and the #1485 keep-running refusal are unchanged).
- `scripts/workflow_lint.py` and ruff pass on touched files.

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py
- fingerprint: 00425f750fed
