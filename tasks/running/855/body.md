---
title: 'workflow-fix: chunk-cap calibration lesson (gotchas + vectorize rule)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2e2d266f748d
created_at: '2026-07-02T09:19:54Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #811 r8: memory cap sized by
  counting explicit temporaries under-estimates autograd+optimizer peak ~6x; calibrate
  from measured real-shape peak; log resolved cap + probed free'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson block
(`gotcha_candidate: yes`) raised on task #811 (emitting agent:
experiment-implementer, crash-fix round 8).

## Goal

Add the measured-peak chunk-cap calibration lesson to
`.claude/rules/gotchas.md` and `.claude/rules/vectorize-many-cell-fits.md`:
size a memory-aware chunk cap's live-tensor factor from a MEASURED
real-shape peak, never from counting the code's explicit temporaries; log
the resolved cap + the probed free value at the cap site.

## Workflow gap

- **Bug observed:** `vectorized_mlp_skill.fit_batched_loco_mlp_multihead`'s
  chunk cap with `live_factor=4` (a count of the explicit temporaries)
  picked c=218 whose REAL peak (~36 GiB — autograd backward graph + AdamW
  moments + allocator retention) re-OOM'd the #811 phase-0 gate on the very
  shape the cap existed to protect (n=480, d_in=3584, A100-80).
- **Why it is a workflow gap:** `.claude/rules/vectorize-many-cell-fits.md`
  names `vectorized_mlp_skill.py` as the canonical many-cell fit helper but
  carries NO memory-sizing guidance, and `gotchas.md` has no entry for the
  temporary-counting under-estimate class — the next agent adding a memory
  cap to a torch fit loop will repeat the ~6× under-estimate.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix measured
  and production-shape smoked in #811 r8).

## Proposed change (candidate diff sketch — refine in planning)

```
+ gotchas.md: new entry "Memory caps for torch fit loops: calibrate from a
+   measured real-shape peak" — explicit-temporary counting under-estimates
+   the autograd+optimizer+allocator peak ~6× (#811: factor-4 cap → c=218 →
+   ~36 GiB re-OOM). Measure ru_maxrss / mem_get_info delta on one chunk at
+   the real shape; conservatism only adds chunk count at constant FLOPs;
+   log resolved cap + probed free (the fix-engaged signal).
+ vectorize-many-cell-fits.md: reference resolve_chunk_cap() (live_factor=26
+   measured, #811 r8) as the canonical memory-sizing companion of the
+   vectorized helper.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`, `.claude/rules/vectorize-many-cell-fits.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'live_factor\|chunk cap\|resolve_chunk_cap' .claude/ CLAUDE.md scripts/`)
  and update every hit; list them in the plan. (At filing time: zero hits in
  `.claude/rules/` — the guidance is absent, which is the gap.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-lessons-index` passes (no LESSONS.md row
  change needed unless a new rule file is created — prefer editing the two
  existing files).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md,.claude/rules/vectorize-many-cell-fits.md
- fingerprint: 2e2d266f748d

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: phase0_gate
lesson: A memory cap calibrated by counting the explicit temporaries in the code (live_factor≈4 tensors) under-estimates the real per-chunk peak ~6× — the autograd backward graph, AdamW moment buffers, and allocator high-water retention dominate the named tensors. Calibrate the factor from a MEASURED peak at the real shape (ru_maxrss / mem_get_info delta on one chunk), and log the resolved cap + the probed free value so the next crash is diagnosable from the log alone.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
