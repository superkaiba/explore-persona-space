---
title: 'workflow-fix: teardown prose contradicts the compute-kill approval gate'
kind: infra
tags:
- wf-fix
- wf-fix-fp:pod-teardown-prose-vs-kill-approval-gate
created_at: '2026-08-05T15:59:56Z'
has_clean_result: false
origin_prompt: 'Raised on #1739: 3/3 verified-done pilot pods refused automated teardown
  (Automation must NOT set this - surface the pod for approval instead), while CLAUDE.md
  + issue SKILL.md still say verified-done teardown is unconditional / no ask-gate
  / NEVER a user decision.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1739 (emitting agent: the #1739 orchestrator session, 2026-08-05).

The compute-kill approval gate landed on 2026-08-04 (`3a2f364a70`, "compute-kill approval gate: nothing shuts down on its own") and now REFUSES every automated pod termination, instructing automation to surface the pod for approval instead. But `CLAUDE.md` § Pods and the `/issue` SKILL.md completion-side teardown clause still tell sessions the opposite — that verified-done teardown is "unconditional", carries "no ask-gate", and is "NEVER a user decision" — and name the exact command to run. A session that follows the prose walks straight into a mechanical refusal.

## Goal

Reconcile the teardown prose in `CLAUDE.md` and `.claude/skills/issue/SKILL.md` with the live compute-kill approval gate, so sessions are directed to SURFACE a verified-done pod for approval (marker + push, pod left alive) instead of attempting a self-approved terminate.

## Workflow gap

- **Bug observed:** on #1739, three item-A pilot pods finished generation with orchestrator-verified Hub uploads (1000 rollouts + manifest + sentinel per rung, 0 missing). Per the prose the orchestrator ran the prescribed surgical form for each — `pod.py terminate --issue 1739 --name-suffix <slug> --yes` — and all three REFUSED, printing `Automation must NOT set this — surface the pod for approval instead.` The pods stayed RUNNING at ~$8/hr combined until surfaced for user approval.
- **Why it is a workflow gap:** two current workflow surfaces contradict a third, newer, deliberate one. The gate is implemented at the per-backend choke points — `src/explore_persona_space/backends/kill_approval.py` (module docstring: *"Automation must NEVER set these. The correct automated behaviour is to SURFACE the candidate for approval (a marker, a push) and leave it alive."*), with the RunPod copy in `scripts/runpod_api.py:1044` and `scripts/pod_lifecycle.py:3335` — and exists because on 2026-08-04 the stale-pod audit destroyed 77 teammate-owned pods on the shared RunPod team account. It is the authoritative surface and must NOT be weakened. The stale surfaces are the prose that still promises unconditional teardown.
- **Confidence (emitter):** high — the refusal is mechanical and reproducible (3/3 pods, verbatim message), and both sides of the contradiction are current-tree text confirmed by per-target grep.
- verified-at-filing: `grep -c` per target, run at body-compose time 2026-08-05 — `CLAUDE.md`: `no ask-gate`=2, `verified-done teardown is unconditional`=1, `NEVER a user decision`=1 (§ Pods line ~290 + the inline free-analysis carve-out line ~51); `.claude/skills/issue/SKILL.md`: `no ask-gate`=1, `verified-done teardown is unconditional`=1. Guard side: `grep -rn "surface the pod for approval" scripts/ src/` → `scripts/pod_lifecycle.py:3335`, `scripts/runpod_api.py:1044`; module `src/explore_persona_space/backends/kill_approval.py` at `3a2f364a70` (2026-08-04). NOTE: an earlier draft of this candidate named `scripts/pod.py` and a SKILL.md phrase that returned 0 hits — mis-targets, corrected by a repo-wide re-grep before filing (the guard is reached through `pod_lifecycle`/`runpod_api`, not spelled in `pod.py`).

## Proposed change (candidate diff sketch — refine in planning)

- `CLAUDE.md` § Pods "Completion-side teardown (suffixed pods — no ask-gate, #1662)": replace the unconditional-terminate instruction with the surface-for-approval contract — verify uploads, post an approval-request marker naming each pod, its hourly burn, and the surgical per-pod command, fire one PushNotification, LEAVE THE POD RUNNING. Drop "terminating a verified-done pod is NEVER a user decision" and the "standing exception to the ask-before-terminating-pods rule" sentence: the approval gate makes it exactly a user decision now.
- Same rewrite at the `CLAUDE.md` inline free-analysis carve-out ("verified-done teardown is unconditional") and the matching `.claude/skills/issue/SKILL.md` clause.
- Check whether Step 8's PRIMARY-pod auto-terminate is likewise gated (it routes through the same choke point — very likely yes) and rewrite it the same way if so.
- Explicitly state that a session must NEVER pass `--approve` or set `EPS_ALLOW_COMPUTE_KILL=1` / `EPS_ALLOW_POD_TERMINATE=1`.

## Scope / surfaces

- Primary targets: `CLAUDE.md` (2 sites), `.claude/skills/issue/SKILL.md` (1 site).
- Cross-check only, do NOT weaken: `src/explore_persona_space/backends/kill_approval.py`, `scripts/runpod_api.py`, `scripts/pod_lifecycle.py`.

## Constraints / invariants

- Workflow-surface only — no experiment code, no `configs/`, no `tasks/`.
- The fix must NOT weaken the approval gate and must NOT introduce an automation self-approve path. Prose changes only on the automation side.
- Preserve unchanged: verify-uploads-before-teardown ordering; never `pod.py stop` as a durability substitute (#1112); the bare-`--issue`-form warning (destroys every live pod on the issue, refuses under `keep-running`, #1485).
- `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files; the SKILL.md/CLAUDE.md prose-pin tests stay green — note `tests/test_suffixed_pod_completion_teardown_pin.py` exists and likely PINS the very text being changed, so it must be updated in the same round.

## Provenance

- workflow_fix_target: CLAUDE.md
- fingerprint: pod-teardown-prose-vs-kill-approval-gate

Raised as prose by the #1739 orchestrator after 3/3 verified-done pilot pods refused automated teardown; the session surfaced them for approval instead rather than self-approving (approval-request `epm:progress` marker on #1739, 2026-08-05).
