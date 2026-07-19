---
title: 'workflow-fix: plans enumerate off-pod phase file reads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:66df230991e8
- daily-auto-filed
created_at: '2026-07-19T07:06:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): Off-pod phase file reads
  are not required to be enumerated in plans (#1526), and a legitimately VM-side phase
  is structurally outside the pod dispatch chain so upload-verifier r1 FAILs by construction
  (#1426 c0-P4, merged in).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose workflow-fix
follow-up raised on task #1526 (emitting agent: round-1 Alternatives critic;
parked under the recursion guard, routed by the 2026-07-18 /daily Step C
parked-candidate sweep).

## Goal

Extend the planner spec (§9/§10) and/or the upload-verifier so every OFF-POD
phase's file reads must be ENUMERATED in the plan — making them plan-named
and thus Step-2.8-gated — as the mechanical-enforcement complement to
#1526's docs bullet.

## Workflow gap

- **Bug observed:** an off-pod phase's file reads are not required to be
  enumerated in the plan, so an off-pod read of a file the pod upload set
  does not carry is only caught at run time (#1526's incident class: off-pod
  phase reads vs the pod upload set).
- **Why it is a workflow gap:** the gotchas rule ("check off-pod phase reads
  against the pod upload set") is docs-only guidance with no plan-time
  enforcement hook; enumerating off-pod reads in §9/§10 makes them
  plan-named and gate-checkable, closing the docs-vs-enforcement gap the
  CLAUDE.md "describes a rule but the implementing file doesn't enforce it"
  criterion names.
- **Confidence (emitter):** low-medium
- verified-at-filing: `grep -cin 'off-pod' .claude/agents/planner.md .claude/agents/upload-verifier.md` → 0 hits in both targets (absence-of-duty claim confirmed: neither file carries an off-pod read-enumeration requirement); `git log --oneline --since='7 days ago' -- .claude/agents/planner.md` → 4 commits (00e1d8e42d reuse-fitness (l), 29da7c891d out-root mount binding, f5b533aff2, 86e8e1a988), none adds off-pod read enumeration (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

Sketch for the planner: add to planner.md §9 (compute/phase sizing) a per-
OFF-POD-phase requirement to enumerate the files the phase READS (with their
producing phase / upload destination), and to upload-verifier.md a
reconciliation arm: every plan-enumerated off-pod read must resolve against
the verified upload set before the pod is released.

### Merged concern — VM-side phase pre-declaration to the verifier (from #1426, /daily 2026-07-18 c0-P4)

Same two target files, complementary gap: #1426's planned F4 VM-side phase was
STRUCTURALLY outside the pod dispatch chain, so `upload-verifier` r1 FAILed BY
CONSTRUCTION twice (initial + follow-up round), each needing auto-recover +
re-verify. The verifier expects the phase's outputs on the pod / in the upload
set, but a legitimately VM-side phase produces them off-pod. Fix (same
surface): a plan that declares a VM-side / off-pod PHASE either (a) includes it
in the pod dispatch chain, OR (b) PRE-DECLARES it to the verifier via a
`declared-off-pod-phase` (or `declared-VM-side-phase`) plan slot, so the
verifier reconciles that phase's outputs against the VM-side / off-pod
destination instead of FAILing by construction on their absence from the pod.
This pairs with the off-pod-READS enumeration above: reads are enumerated for
input-availability, the phase itself is declared so its OUTPUTS are not
mis-expected on-pod.
verified-at-filing (merge leg): `grep -cin 'vm-side\|VM side\|off-pod\|declared.*phase' .claude/agents/upload-verifier.md` → 0 hits for a declared-off-pod-phase slot (the verifier has no way to know a phase is legitimately off-pod, so it FAILs r1 by construction) (2026-07-19)

## Scope / surfaces

- Primary target: `.claude/agents/planner.md, .claude/agents/upload-verifier.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'upload set\|off-pod' .claude/ CLAUDE.md scripts/`) — the
  gotchas.md "check off-pod phase reads against the pod upload set" entry is
  the docs sibling this mechanizes; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The enumeration duty must not bloat plans for pod-free tasks — scope it to
  plans with a pod + a subsequent off-pod phase.
- `scripts/workflow_lint.py --check-asks` passes; `verify_plan.py` stays
  consistent if it gains a check.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md, .claude/agents/upload-verifier.md
- fingerprint: 53d6679bbee6

Verbatim surfaced prose (task #1526 events.jsonl, 2026-07-19T02:08:29Z):
"Candidate (from round-1 Alternatives critic, confidence low-medium): extend
the planner spec (§9/§10) and/or upload-verifier so every OFF-POD phase's
file reads must be ENUMERATED in the plan (making them plan-named and thus
Step-2.8-gated) — the mechanical-enforcement complement to #1526's docs
bullet. target_file: .claude/agents/planner.md, .claude/agents/upload-verifier.md."
