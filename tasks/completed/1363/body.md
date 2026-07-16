---
title: 'workflow-fix: MALLOC_ARENA_MAX=2 in the canonical VM-launch env prefix'
kind: infra
tags:
- wf-fix
- wf-fix-fp:77ca2f4f987d
created_at: '2026-07-15T21:19:52Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1315 analyzer r1: glibc arena fragmentation
  OOM on VM dense-fit launches; MALLOC_ARENA_MAX=2 missing from the thread-caps prefix'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1315 (emitting agent: analyzer, round 1).

## Goal

Add MALLOC_ARENA_MAX=2 to the canonical VM-launch env prefix in the thread-caps bullet (code-style.md) and the vectorize-many-cell-fits detached-fit recipe, with a one-line note that RSS growth across passes without any large single allocation is the arena signature.

## Workflow gap

- **Bug observed:** A VM-side batched Gram-eigh bootstrap with <=tens-of-MB per-pass tensors grew to 20-21.7 GB RSS across ~7-9 passes (glibc malloc arena fragmentation under 8 BLAS/torch threads) and was earlyoom-SIGTERMed twice on the shared VM; choom -600 applied at launch did not stick to the uv-spawned python3 child (#1315 analyzer geometry runs, 2026-07-15, VM root at 1.7 GB free).
- **Why it is a workflow gap:** The shared-VM thread-caps rule (code-style.md #847; echoed in the Step 9 entry-guard detached-launch prefix and vectorize-many-cell-fits.md) caps threads but says nothing about MALLOC_ARENA_MAX, so every many-cell dense-fit VM launch inherits the arena-fragmentation OOM trap the caps were meant to prevent.
- **Confidence (emitter):** high (MALLOC_ARENA_MAX=2 held the same 20 GB-ballooning eigh bootstrap at ~1 GB in the #1315 session)
- verified-at-filing: `grep -c "MALLOC_ARENA_MAX" .claude/rules/code-style.md` → 0 hits; `grep -c "MALLOC_ARENA_MAX" .claude/rules/vectorize-many-cell-fits.md` → 0 hits; repo-wide `grep -rln "MALLOC_ARENA_MAX" .claude/ CLAUDE.md scripts/` → 0 files (absence-of-guard claim — the 0-hit in-target results ARE the evidence) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

- `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 uv run python ...`
+ `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python ...`
+ RSS that grows per pass with no large single allocation = glibc arena fragmentation;
+ MALLOC_ARENA_MAX=2 held a 20 GB-ballooning eigh bootstrap at ~1 GB (#1315).

## Scope / surfaces

- Primary targets: `.claude/rules/code-style.md`, `.claude/rules/vectorize-many-cell-fits.md`
- The same env-prefix line also appears in `.claude/skills/issue/SKILL.md` (Step 9 entry guard § Detached VM-side long compute phases) and possibly sibling skills — `grep -rln 'NUMEXPR_NUM_THREADS=8' .claude/ CLAUDE.md scripts/` and update every canonical-prefix hit; list them in the plan. Also consider the choom-not-sticking-to-uv-child observation as a secondary note (the choom sweep runs pgrep -s over the session — investigate whether the uv-spawned child escaped the session sweep).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags run passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/code-style.md, .claude/rules/vectorize-many-cell-fits.md
- fingerprint: 77ca2f4f987d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/code-style.md, .claude/rules/vectorize-many-cell-fits.md
bug_observed: A VM-side batched Gram-eigh bootstrap with <=tens-of-MB per-pass tensors grew to 20-21.7 GB RSS across ~7-9 passes (glibc malloc arena fragmentation under 8 BLAS/torch threads) and was earlyoom-SIGTERMed twice on the shared VM; choom -600 applied at launch did not stick to the uv-spawned python3 child.
why_workflow_gap: The shared-VM thread-caps rule (code-style.md #847) caps threads but says nothing about MALLOC_ARENA_MAX, so every many-cell dense-fit VM launch inherits the arena-fragmentation OOM trap the caps were meant to prevent.
proposed_change: Add MALLOC_ARENA_MAX=2 to the canonical VM-launch env prefix in the thread-caps bullet (and the vectorize-many-cell-fits detached-fit recipe), with a one-line note that RSS growth across passes without any large single allocation is the arena signature.
diff_sketch: |
  - `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 uv run python ...`
  + `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 uv run python ...`
  + RSS that grows per pass with no large single allocation = glibc arena fragmentation;
  + MALLOC_ARENA_MAX=2 held a 20 GB-ballooning eigh bootstrap at ~1 GB (#1315).
confidence: high
related_task: #1315
<!-- /workflow-fix-candidate -->
