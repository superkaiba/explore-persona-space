---
title: 'workflow-fix: external-stream presumption for Step 3.6 / checkpoint-per-phase'
kind: infra
tags:
- wf-fix
- wf-fix-fp:dcfecd21b24b
created_at: '2026-07-08T01:10:07Z'
has_clean_result: false
origin_prompt: 'user chat 2026-07-07: ''isn''t there an indication somewhere to always
  checkpoint stuff?'' -> audit found code-reviewer Step 3.6''s trigger blind to network-bound
  streaming loops (#1092 attempt-3 3h stream loss)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1092 (emitting agent: orchestrator/PM chat, user-prompted audit).

## Goal

Extend the long-loop restartability enforcement (code-reviewer Step 3.6 + code-style checkpoint-per-phase) with an EXTERNAL-STREAM presumption so network-bound streaming/harvest phases can no longer evade the checkpoint requirement via trivial per-row kernels and unsizeable wall-time.

## Workflow gap

- **Bug observed:** #1092 P0 attempt 3 accumulated a ~3h HF-datasets streaming pool in memory and lost it on a downstream crash (KeyError in topic labeling); the round-8 crash-fix then added the stream cache reactively (`issue1092_build_corpus.py` `_stream_with_cache`, "the 3h-stream protection").
- **Why it is a workflow gap:** `.claude/agents/code-reviewer.md` Step 3.6 exists precisely to catch uncheckpointed long loops (born from #823, where five review rounds missed ~20h of unpersisted ridge fits), but its TRIGGER has a blind spot: it keys on (a) projected wall-time from plan §9 / smoke-extrapolated per-call cost and (b) a ">~500 serial calls of a NON-TRIVIAL kernel" presumption. A network-bound streaming loop has a trivial per-row kernel (parse+filter, ~ms) and wall-time dominated by network throughput that neither §9 prose nor the diff can size reliably — so NEITHER prong fires, and #1092's multi-round review passed the uncheckpointed 3h stream. Same failure family as #823 (rule exists, enforcement misses), one layer deeper (the enforcement gate itself has a typed blind spot).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/agents/code-reviewer.md Step 3.6 Trigger:
+ - EXTERNAL-STREAM presumption: any loop consuming an external streaming
+   source (HF `datasets` streaming=True, API pagination, web harvest,
+   S3/HTTP row iteration) with a target volume >~10^4 rows OR an unbounded
+   stop condition is PRESUMED over the ~1h floor REGARDLESS of per-row
+   kernel triviality — network throughput is not sizeable from the diff,
+   and realized stream walls (#1092: ~3h) routinely exceed §9 prose.
+   Required: per-chunk durable persistence (atomic JSONL append) + a
+   fingerprint-gated resume keyed on the filter/recipe code marker
+   (the #1092 round-8 `_stream_with_cache` shape is the reference impl).

.claude/rules/code-style.md § "Checkpoint per phase" (intra-phase grain):
+ - The ~1h floor is PRESUMED met by any external-stream loop (network-bound
+   row iteration; see the external-stream presumption in code-reviewer
+   Step 3.6) — checkpoint it regardless of projected wall-time.
```

## Scope / surfaces

- Primary targets: `.claude/agents/code-reviewer.md` (Step 3.6 trigger + incident list), `.claude/rules/code-style.md` (§ Checkpoint per phase intra-phase grain clause)
- Grep the workflow surface for the pattern before editing (`grep -rln 'non-trivial kernel\|checkpoint-per-phase\|Checkpoint per phase' .claude/ CLAUDE.md scripts/`) and update every hit that states the ~1h trigger; list them in the plan. Consider whether `.claude/rules/plan-compute-sizing.md` (many-call floor) and `.claude/agents/experiment-implementer.md` need the same clause.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; rule text and reviewer-gate text stay consistent with each other.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md,.claude/rules/code-style.md
- fingerprint: dcfecd21b24b

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/code-reviewer.md,.claude/rules/code-style.md
bug_observed: #1092 P0 attempt 3 accumulated a ~3h HF-datasets streaming pool in memory and lost it on a downstream crash; Step 3.6 never fired because its trigger keys on projected wall-time and a >500-serial-calls-of-non-trivial-kernel presumption, both of which a trivial-kernel network-bound streaming phase evades
why_workflow_gap: The dedicated enforcement gate for the checkpoint-per-phase rule (born from #823) has a typed blind spot for network-bound streaming loops, so the rule fails exactly where wall-time is least sizeable from the diff
proposed_change: Extend code-reviewer Step 3.6 trigger and code-style checkpoint-per-phase with an external-stream presumption: any network-bound streaming/harvest loop (HF datasets streaming, API pagination, web crawl) is presumed over the ~1h floor regardless of kernel triviality and requires per-chunk persistence plus fingerprint-gated resume
diff_sketch: |
  see ## Proposed change above
confidence: high
related_task: #1092
<!-- /workflow-fix-candidate -->
