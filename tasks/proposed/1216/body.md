---
title: 'daily-held: arm the #864 zombie namespace veto'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-09T07:01:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 3): EPM_ZOMBIE_NAMESPACE_VETO
  still ships default-OFF (poll_pipeline.py:628, ''0'') because the #864 §7 live-pod
  gate registered disposition 2: an allocation-free cuInit''d parent (issue813_dispatch.py)
  holds /dev/nvidia-uvm while absent from compute-apps, so the discriminator would
  false-fire on TP coordinators — the #813 false-positive class persists in production
  until the veto is redesigned and armed'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #864 (recursion-guarded workflow-fix session).

## Goal

Arm the #864 namespace veto safely. Route 3: flipping the default changes what the poller classifies as a dead workload on LIVE pods — a false fire mis-triggers stall handling/failover on healthy runs, and the prerequisite is a live-pod validation gate re-run (judgment + destructive-adjacent), not a mechanical edit.

## Workflow gap

- **Bug observed:** EPM_ZOMBIE_NAMESPACE_VETO still ships default-OFF (poll_pipeline.py:628, '0') because the #864 §7 live-pod gate registered disposition 2: an allocation-free cuInit'd parent (issue813_dispatch.py) holds /dev/nvidia-uvm while absent from compute-apps, so the discriminator would false-fire on TP coordinators — the #813 false-positive class persists in production until the veto is redesigned and armed.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/poll_pipeline.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
# poll_pipeline.py: redesign discriminator (allocation-free uvm holders excluded),
# re-run §7 both-directions gate on a live pod (incl. vLLM EngineCore), then:
- ZOMBIE_NAMESPACE_VETO_ENABLED = os.environ.get("EPM_ZOMBIE_NAMESPACE_VETO", "0") != "0"
+ ZOMBIE_NAMESPACE_VETO_ENABLED = os.environ.get("EPM_ZOMBIE_NAMESPACE_VETO", "1") != "0"
```

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- origin: parked candidate on task #864 at 2026-07-02T20:16:17Z

Verbatim parked note:

> source: prose-followup (implementer report (d), 2026-07-02). routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidate: the #864 namespace veto ships default-OFF because the §7 live-pod gate registered disposition 2 — issue813_dispatch.py (a cuInit'd parent/coordinator) holds exact /dev/nvidia-uvm while ABSENT from compute-apps (5 uvm holders vs 4 compute apps on pod-813), the TP-suppression channel. Follow-up needed to ARM the veto: redesign the live-compute discriminator so allocation-free parents do not count (e.g. require uvm-holder ∧ VRAM>floor correspondence, or count only uvm holders whose /proc/<pid> maps to a compute-apps identity), re-run the §7 both-directions gate incl. the vLLM EngineCore sub-case, then flip EPM_ZOMBIE_NAMESPACE_VETO default to 1 (one literal in scripts/poll_pipeline.py). target_file: scripts/poll_pipeline.py. Until armed, the #813 false-positive class persists in production (mitigated: the override only false-fires on >stall-window quiet stretches, and the orchestrator-side SKILL.md routing survives it). For the next non-workflow-fix orchestrator / PM pass to file.
