---
title: 'workflow-fix: phase-partition GPU width — narrow/API phases must not hold
  the peak-width (8xH100) pod'
kind: infra
tags:
- wf-fix
- wf-fix-fp:84f3e574b03b
created_at: '2026-07-01T08:47:23Z'
has_clean_result: false
origin_prompt: 'Fix this: gpu-idle-advisory on #778 — 8x H100 pod held through extract
  + API-bound judge phases at <=5% util; only the finetuning fan-out needs 8-wide.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow gap observed on task #778 (user-directed: "fix this" re a gpu-idle-advisory on the 8× H100 pod).

## Goal

Add a phase-partitioned GPU-WIDTH right-sizing rule so a multi-phase GPU run does not hold the peak-width pod (e.g. 8× H100) through its GPU-light / narrow / API-bound phases.

## Workflow gap

- **Bug observed:** #778 sized ONE 8× H100 pod at the peak-phase width and held it through GPU-light phases — extract (`gpu-idle-advisory`: all 8 GPUs ≤5% util for 38 min) and the API-bound graded-judge phase (Anthropic Batch API, ~0 GPU) — at ~$25/hr. Only Phase 3 (the 24-run finetuning fan-out) actually needs 8-wide.
- **Why it is a workflow gap:** `planner.md §9`, `critic.md` Methodology lens item 10, and the CLAUDE.md "CPU-only phases don't hold GPU pods" section cover only the BINARY case (CPU-only phase → release/off-pod) plus the footprint (>50 GB → `cpu-bigmem`) and compute-character (gradient fit → GPU) carve-outs. There is NO rule for right-sizing GPU WIDTH across the GPU phases of a multi-phase run: a narrow GPU phase (≤7B forward/extract ~1–3 GPUs, ≤7B single-GPU vLLM generation) that holds the run's PEAK width is not covered. The `planner.md §9` sweep-parallelism row even says "one multi-GPU pod with the largest sensible GPU count," which pushes toward holding peak width throughout. Separately, the API-bound judge phase should ALREADY release the GPU pod (existing "judge-API-only scoring MUST NOT hold a GPU pod" rule) — the plan didn't sequence it, so that rule needs a reinforced plan-time hook too.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

- Add a **GPU-width right-sizing** rule mirroring the existing CPU-only-phase threshold: a GPU phase that needs MATERIALLY FEWER GPUs than the run's peak width AND runs longer than ~15–30 min must NOT hold the peak-width pod. Provision the wide pod only for the wide phase; run the preceding narrow / API-bound phases on a narrow pod (smallest intent that fits) or off-pod.
- **Explicitly weigh the tradeoff** (do not make it naive): re-provisioning a second pod costs provisioning latency + doubles RunPod supply-constraint exposure. So the rule is threshold-gated — SHORT narrow phases (< ~15–30 min) may hold the wide pod (churn not worth it); LONG narrow/API-bound phases (or a long batch-judge wait) must release/downsize. State the idle-$ vs re-provision-risk weighing in the plan.
- Reinforce that an **API-bound judge phase releases the GPU pod during the batch wait** (the deadline-bounded `batch_judge` poll is free and off-pod) — the existing rule, now with a plan-time sequencing hook.
- **planner.md §9:** size per-phase GPU WIDTH (not one peak spec for the whole run); name which phase justifies the peak width and how the narrow/API phases avoid holding it.
- **critic.md Methodology lens item 10:** add a REVISE direction — a plan that holds the peak-width pod through a long narrow-GPU or API-bound phase without a stated re-provision-cost justification.
- **CLAUDE.md** "CPU-only phases don't hold GPU pods" section: extend the heading/scope to GPU-WIDTH right-sizing (narrow GPU phases don't hold a WIDE pod), cross-linking #664 (idle-GPU spend-leak) and this incident.

## Scope / surfaces

- Primary: `.claude/agents/planner.md` (§9), `.claude/agents/critic.md` (Methodology lens item 10), `CLAUDE.md` (CPU/GPU-pod-sizing section).
- Grep the surface before editing: `grep -rniE "largest sensible GPU count|CPU-only phases don.t hold|idle multi-GPU|per-phase|GPU pod" .claude/agents/planner.md .claude/agents/critic.md CLAUDE.md`.

## Constraints / invariants

- Workflow-surface only (planning/critic/CLAUDE prose). No experiment-code changes.
- Do NOT contradict the existing sweep-parallelism rule (one multi-GPU pod for N shared-nothing seeds that EACH need the pod) — the new rule is about phases that need DIFFERENT widths, not about splitting a single wide-parallel phase.
- Keep the threshold consistent with the existing ~15–30 min CPU-only-phase floor.
- `scripts/workflow_lint.py` passes; keep CLAUDE.md ↔ planner.md ↔ critic.md consistent.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` with a `workflow_fix_target:` Provenance line — must NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: 84f3e574b03b

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/planner.md, .claude/agents/critic.md, CLAUDE.md
bug_observed: a multi-phase GPU run sizes one pod at the peak-phase width (8x H100) and holds it through GPU-light/narrow phases (extract ~1-3 GPUs, <=7B generation ~1 GPU) and the API-bound judge phase, burning ~$25/hr at <=5% util (gpu-idle-advisory fired 38 min during #778 extract); only the finetuning fan-out needs 8-wide.
why_workflow_gap: planner.md §9 + critic.md item 10 + CLAUDE.md cover CPU-ONLY phases releasing the GPU pod plus footprint/compute-character carve-outs, but have NO rule for right-sizing GPU WIDTH across the GPU phases of a multi-phase run; the sweep-parallelism row even mandates "one multi-GPU pod with the largest sensible GPU count," and the API-bound judge phase (which should already release the pod) was not sequenced to do so.
proposed_change: add a phase-partitioned GPU-width rule mirroring the CPU-only-phase threshold — a GPU phase needing materially fewer GPUs than the run's peak width AND running longer than ~15-30 min must not hold the peak-width pod (provision the wide pod only for the wide phase; run narrow/API phases on a narrow pod or off-pod), weighing idle-$ saved vs re-provision latency/supply-risk; planner sizes per-phase GPU width in §9, critic REVISEs a plan holding peak width through a long narrow/API phase.
diff_sketch: |
  + planner.md §9: size per-phase GPU WIDTH, not one peak spec; name the phase that justifies peak width
  + threshold: narrow GPU phase > ~15-30 min AND << peak width -> narrow pod / off-pod; short ones may hold
  + API-bound judge phase -> release GPU pod during the (free, deadline-bounded) batch wait
  + critic.md item 10: REVISE a plan holding peak-width pod through a long narrow/API phase w/o re-provision-cost justification
  + CLAUDE.md: extend "CPU-only phases don't hold GPU pods" -> GPU-WIDTH right-sizing (narrow GPU phase != wide pod), xref #664
confidence: high
related_task: #778
<!-- /workflow-fix-candidate -->
