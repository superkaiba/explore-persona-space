---
title: 'workflow-fix: context-hygiene rule for harmful-content bank files in agent
  specs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ec5e63eaf202
created_at: '2026-07-02T20:49:22Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #866: repeated spurious usage-policy
  refusal-kills from raw benchmark-bank text paged into agent context; see body Provenance
  block'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #866 (emitting agent: orchestrator).

## Goal

Add an explicit context-hygiene rule to the implementer / experiment-implementer /
analyzer agent specs: never page raw item text of harmful-content benchmark
question banks (e.g. files under `src/explore_persona_space/artifacts/query_banks/`)
or raw completions into agent context; verify such files via counts / key names /
sha256 / index ranges only, and keep report wording neutral.

## Workflow gap

- **Bug observed:** Task #866 round-1 implementer sessions were refusal-killed
  repeatedly (41, 20, and 2 tool uses) by spurious usage-policy false positives
  ("violative cyber content") after raw benchmark-bank text was paged into
  context during verification, and later at prompt level from trigger-dense
  brief vocabulary. Four sessions lost on one task.
- **Why it is a workflow gap:** CLAUDE.md § "Spurious usage-policy refusals"
  clause (d) covers raw-completion files, but the agent specs
  (`implementer.md`, `experiment-implementer.md`) carry no such rule, and the
  clause does not name benchmark QUESTION-BANK files — so agents verify bank
  files by reading their text and get killed.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ In `.claude/agents/implementer.md` + `.claude/agents/experiment-implementer.md`
+ (and align `analyzer.md`'s existing raw-completion clause):
+ "Context hygiene for harmful-content data files: NEVER read/print the raw
+  item text of safety-benchmark question banks or raw completion files
+  (query_banks/*.json, raw_completions/). Verify via python printing counts,
+  key names, sha256, index ranges only. Reference items by bank+index.
+  Keep report/marker wording neutral — no verbatim items."
+ In CLAUDE.md § Spurious usage-policy refusals (d): extend "raw-completion
+ files" to "raw-completion files AND benchmark question banks".

## Scope / surfaces

- Primary target: `.claude/agents/implementer.md, .claude/agents/experiment-implementer.md, .claude/agents/analyzer.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'raw harmful-content\|raw-completion' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/implementer.md, .claude/agents/experiment-implementer.md, .claude/agents/analyzer.md
- fingerprint: ec5e63eaf202

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/implementer.md, .claude/agents/experiment-implementer.md, .claude/agents/analyzer.md
bug_observed: Task #866 round-1 implementer sessions refusal-killed twice (41 and 20 tool uses) by spurious usage-policy trips after paging raw advbench/strongreject bank text into context during verification
why_workflow_gap: CLAUDE.md spurious-refusals clause (d) covers raw-completion files only and lives outside the agent specs; implementer/experiment-implementer specs carry no context-hygiene rule for harmful-content benchmark banks
proposed_change: Add a context-hygiene rule to implementer/experiment-implementer/analyzer agent specs: never page raw item text of harmful-content benchmark banks or raw completions into context; verify via counts/keys/sha + grep-by-offset only; neutral report wording
diff_sketch: |
  + NEVER read/print raw item text of safety-benchmark question banks or raw
  + completion files; verify via counts/keys/sha256/index ranges only;
  + reference items by bank+index; neutral report wording.
confidence: high
related_task: #866
<!-- /workflow-fix-candidate -->
