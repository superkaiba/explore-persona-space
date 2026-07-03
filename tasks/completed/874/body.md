---
title: 'workflow-fix: compute-character statement on plannerless launch paths'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fff0f4dc2da6
created_at: '2026-07-02T16:11:33Z'
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

Require a one-paragraph compute-character statement (ops arithmetic + batched-helper confirmation) before launching any fit or statistical battery on the two plannerless paths: inline user-chat free-analysis and the same-issue follow-up loop.

## Workflow gap

- **Bug observed:** #667's inline free-analysis reused #722's serial per-cell `fit_cell` (300-epoch MLP per cell x 84 cells — the MLP already established uninformative by #722, so pure cost) and ran ~1-3h serial, caught only by an ad-hoc pace watch; #778's same-issue follow-up re-ran the parent's serial 1000-draw null battery (2+h, projected 4-6h vs ~15-30 min batched), caught only by the user, twice; #722's own originating follow-up was planned as "0 GPU-h free-analysis ... No gates" and burned 19.5 CPU-h.
- **Why it is a workflow gap:** the inline free-analysis carve-out (CLAUDE.md § Routing experiment intent) and the same-issue follow-up loop (SKILL.md step 9b) intentionally skip the planner+critic stack — which is where ALL compute-character review lives (planner §9, critic Methodology item 10(iii)). "0 GPU-h" is being read as "no compute review needed", but 0 GPU-h work can still burn tens of CPU-hours or hold a pod. There is currently NO surface on these paths that asks "is the inner loop batched?".
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

In CLAUDE.md's "User-chat inline free analysis" carve-out and SKILL.md's same-issue follow-up loop + step 9a-ter (zero-GPU floor):

```
+ Compute-character pre-launch statement (one paragraph, NOT a planner
+ round): before launching any fit, sweep, or statistical battery
+ (permutation/bootstrap/null draws, per-cell/per-fold fits, per-row
+ model calls) on this path, the executor states: (1) the ops arithmetic
+ (cells x folds x draws x epochs, and the projected wall-time it implies);
+ (2) which BATCHED helper implements the inner loop (name the function),
+ or why the work is genuinely not batchable; (3) for reused parent code,
+ that its inner loop + device routing were inspected (artifact-reuse
+ item (i)). A projected wall-time > ~1h without a batched inner loop is
+ a stop: vectorize first (vectorize-many-cell-fits.md), then launch.
+ "0 GPU-h" does not mean "0 compute review".
```

The statement rides the existing markers (the `epm:progress` note on the inline path; the followup-scope/run markers on the loop path) so it is auditable.

## Scope / surfaces

- Primary targets: `CLAUDE.md` (§ Routing experiment intent — the inline free-analysis carve-out), `.claude/skills/issue/SKILL.md` (same-issue follow-up loop + step 9a-ter)
- Grep: `grep -n 'free.analysis' CLAUDE.md .claude/skills/issue/SKILL.md` and cover every launch point on the plannerless paths.
- Cross-link: open task #869 owns planner-§9-side sizing; this task covers the paths that never reach a planner.

## Constraints / invariants

- The carve-outs' routing semantics are UNCHANGED (inline stays inline; the follow-up loop stays on the parent issue) — this adds a one-paragraph pre-launch obligation, not a gate or an AskUserQuestion.
- `scripts/workflow_lint.py --check-asks` passes (no new ask sites); CLAUDE.md and SKILL.md stay consistent.
- Recursion guard: this session must not auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: CLAUDE.md, .claude/skills/issue/SKILL.md
- fingerprint: fff0f4dc2da6

<!-- workflow-fix-candidate v1 -->
target_file: CLAUDE.md, .claude/skills/issue/SKILL.md
bug_observed: #667's inline free-analysis and #778's same-issue follow-up launched serial fits with no compute gate anywhere — the 0-GPU-h carve-outs skip planner and critic and were read as needing no compute review
why_workflow_gap: all compute-character review lives in the planner/critic stack, which these two paths intentionally bypass, so serial fits launch unexamined exactly where reuse of parent code is most common
proposed_change: require a one-paragraph compute-character statement (ops arithmetic cells x folds x draws + batched-helper confirmation) before launching any fit or statistical battery on the inline free-analysis and same-issue follow-up paths
diff_sketch: |
  + pre-launch paragraph: ops arithmetic + projected wall; named batched
  +   helper (or why not batchable); reused-code inner-loop inspection.
  + projected > ~1h without a batched inner loop => vectorize first.
  + "0 GPU-h" does not mean "0 compute review".
confidence: high
related_task: n/a (user-chat transcript audit, 2026-07-02)
<!-- /workflow-fix-candidate -->
