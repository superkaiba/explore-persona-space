---
title: 'workflow-fix: read-hygiene clause for heavy-read agent specs (autocompact-thrash
  guard)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:379df0cc2f3f
created_at: '2026-07-07T20:45:16Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised by the /issue 1090 orchestrator: two
  subagent autocompact-thrash deaths during read phases; specs lack a context-budget
  read-hygiene guard'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1090 (emitting agent: orchestrator /issue session).

## Goal

Add a mandatory read-hygiene clause (Read limit <=150 lines on files >400 lines; grep-then-slice navigation; never page a full task body / plans/v* / eval JSON bodies) to the heavy-read agent specs so subagents stop autocompact-thrashing on large-artifact tasks.

## Workflow gap

- **Bug observed:** Two subagents on task #1090 (a fact-checker `planner`-type and an `experiment-implementer`, both sonnet-pinned) autocompact-thrashed to death at ~19-20 tool calls during their READ phase ("context refilled to the limit within 3 turns of the previous compact, 3 times in a row"); no durable output; each death cost a full dispatch round (~10-12 min + ~150-600K tokens). The recovery that worked was an orchestrator-composed brief carrying explicit hard read bounds (Read limit <=150 always; no `task.py view <N>` full-body; digests for eval JSONs; --name-only git forms).
- **Why it is a workflow gap:** the agent specs for heavy-read roles (experiment-implementer, planner/fact-checker, analyzer, critic) carry context-safety guidance for REFUSAL triggers (banks, raw completions) but NO structural guard against context-budget overflow on large artifacts (a mature task's body.md + plans/v4-v5 + events.jsonl are each 30-50KB; one unsliced `task.py view` or full-file Read chain fills a subagent context). The fix belongs in the specs so every brief inherits it, instead of each orchestrator re-deriving it after a death.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## Read hygiene (context-budget guard)
+ - Every Read of a file >400 lines carries limit <=150; navigate by Grep (-n, head_limit<=20) then slice.
+ - Never run `task.py view <N>` bare on a task with a clean-result body — use `--json` piped through jq for the specific field, or read body.md in slices skipping fenced sample blocks.
+ - eval_results/** and raw-completion files: digests only (jq keys/length, wc -l), never full-body Reads.
+ - git: --name-only / --stat / -1 --oneline forms only inside a subagent context.
```

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`
- Sibling hits (same missing guard): `.claude/agents/planner.md`, `.claude/agents/analyzer.md`, `.claude/agents/critic.md`, `.claude/agents/implementer.md` — grep the workflow surface for existing read-slice guidance before editing and update every heavy-read spec; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md
- fingerprint: 379df0cc2f3f

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/experiment-implementer.md, .claude/agents/planner.md, .claude/agents/analyzer.md, .claude/agents/critic.md, .claude/agents/implementer.md
bug_observed: Two subagents on task 1090 (fact-checker planner-type and experiment-implementer, both sonnet) autocompact-thrashed to death at ~19-20 tool calls during their read phase; no durable output; each death cost a full dispatch round
why_workflow_gap: heavy-read agent specs carry refusal-trigger context-safety but no context-BUDGET read-hygiene guard; orchestrators re-derive the bounds only after a death
proposed_change: Add a mandatory read-hygiene clause (Read limit <=150 lines on files >400 lines; grep-then-slice navigation; never page a full task body / plans/v* / eval JSON bodies) to the heavy-read agent specs so subagents stop autocompact-thrashing on large-artifact tasks
diff_sketch: |
  + ## Read hygiene (context-budget guard)
  + - Read limit <=150 on >400-line files; Grep-then-slice navigation
  + - never bare `task.py view <N>` on clean-result tasks; jq field extraction or sliced body reads
  + - eval_results/raw-completion files digest-only; git --name-only/--stat forms only
confidence: medium
related_task: #1090
<!-- /workflow-fix-candidate -->
