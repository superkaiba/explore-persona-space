---
title: 'workflow-fix: work-conserving dispatcher + per-row IO review lens'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3d6c6431402a
created_at: '2026-07-02T16:11:44Z'
has_clean_result: false
origin_prompt: 'User (2026-07-02): "A lot of recent transcripts show that parallelization/vectorization
  is not properly being used for experiments which is leading to slowdowns -- Read
  the recent transcripts, figure out why this is happening, and propose workflow improvments
  to fix this" -> after the 7-proposal audit report: "Make all seven changes. Figure
  out the best way"'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a user-approved transcript audit (2026-07-02 PM-chat session; user: "parallelization/vectorization is not properly being used ... Make all seven changes"). Emitting agent: orchestrator, via 5 parallel transcript-mining subagents.

## Goal

Extend the compute-shape review lens in BOTH code-reviewer twins with a work-conserving-dispatcher check and add per-row compression/IO in inner loops to the compute-throughput anti-patterns.

## Workflow gap

- **Bug observed:** #813's dispatcher ran two STRICTLY SEQUENTIAL waves — wave 2 (~55% of remaining rows) would not start until wave 1 fully drained — leaving GPUs 1/2/4/7 idle for 6.7h on a billing 8xH100 pod (true remaining ~38-52h vs the projected 18-20h); the same run's per-row `np.savez_compressed` took ~104s = 65% of each ~126s row (plain `savez`: 1.2s, only 1.29x larger, Xet dedup already -59%). Both passed code review. Similarly #778's phase-3 capture looped 25 models x 3 traits one-at-a-time on an 8xH100 pod (~4-5h at 1/8 util).
- **Why it is a workflow gap:** the Step 0.67 compute-shape-vs-dispatcher lens (#806) checks that the dispatcher EXPOSES the plan-declared parallelism (shard flags exist), but not that the SCHEDULE is work-conserving — a dispatcher with 8-way sharding that inserts a full wave barrier still passes; and the "Compute-throughput anti-patterns" list in the reviewer covers batch-1 forwards, CPU-side reductions, and HF generate, but not per-row compression/IO dominating the inner loop.
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/agents/code-reviewer.md` (Step 0.67 + the throughput anti-patterns block) and mirrored in `.claude/agents/codex-code-reviewer.md` (the composed prompt inlines the same rubric):

```
+ Work-conserving dispatcher check (extends Step 0.67): a multi-GPU/multi-
+ worker dispatcher must be WORK-CONSERVING — whenever a worker/GPU is idle
+ and a pending cell exists, it dispatches; flag as Major any strict wave/
+ stage barrier that leaves workers idle while independent cells wait
+ (#813: two sequential waves idled 4/8 H100s for 6.7h). Barriers are
+ acceptable only for genuine cross-cell dependencies, stated in the plan.
+
+ Throughput anti-pattern (d): per-row compression/serialization/upload
+ inside the inner loop when it dominates row wall-time — e.g.
+ np.savez_compressed at ~104s = 65% of a ~126s row (#813; plain savez
+ 1.2s at 1.29x size). Compress/upload out-of-band or batched, or justify
+ the tradeoff in the diff.
```

Keep the `check_compute_shape_review_lens` lint tokens (`Compute-shape-vs-dispatcher`, `compute-shape-mismatch`) intact in both files; extend the lint if the plan adds new pinned tokens.

## Scope / surfaces

- Primary targets: `.claude/agents/code-reviewer.md`, `.claude/agents/codex-code-reviewer.md`
- Secondary: `scripts/workflow_lint.py` `check_compute_shape_review_lens` only if new tokens are pinned.
- Cross-link: open task #869 Fix B/D also edits code-reviewer.md (smoke extrapolation + named-helper check) — distinct concern/fingerprint; coordinate via normal rebase.

## Constraints / invariants

- Both reviewer twins must face the same bar (the #806 invariant the lint pins).
- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- Recursion guard: this session must not auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md, .claude/agents/codex-code-reviewer.md
- fingerprint: 3d6c6431402a

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/code-reviewer.md, .claude/agents/codex-code-reviewer.md
bug_observed: #813's dispatcher ran two strictly sequential waves leaving 4/8 H100s idle 6.7h and per-row savez_compressed consumed 65% of each row's wall-time — neither is covered by the Step 0.67 compute-shape lens or the throughput anti-patterns
why_workflow_gap: the compute-shape lens checks that parallelism is EXPOSED but not that the schedule is work-conserving, and the throughput anti-pattern list omits inner-loop per-row IO/compression
proposed_change: extend the compute-shape review lens in both reviewer twins with a work-conserving-dispatcher check (idle GPU + pending cell => dispatch; no strict wave barriers) and add per-row compression/IO in inner loops to the compute-throughput anti-patterns
diff_sketch: |
  + Step 0.67 extension: strict wave barrier idling workers while
  +   independent cells wait = Major (work-conserving requirement)
  + anti-pattern (d): per-row compression/serialization/upload dominating
  +   inner-loop row time (savez_compressed 104s of a 126s row, #813)
confidence: high
related_task: n/a (user-chat transcript audit, 2026-07-02)
<!-- /workflow-fix-candidate -->
