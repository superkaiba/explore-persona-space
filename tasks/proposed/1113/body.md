---
title: 'workflow-fix: gotchas entry — tiny-real probe for real-corpus streaming filters'
kind: infra
tags:
- wf-fix
- wf-fix-fp:faaf1240e761
created_at: '2026-07-07T17:47:29Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from task #1092 round 7 (epm:failure-lesson
  v1, 2026-07-07T17:46:15Z)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #1092 (emitting agent: experiment-implementer, round 7).

## Goal

Add a `.claude/rules/gotchas.md` entry: real-corpus streaming filters (WildChat/LMSYS full-name language fields, per-dataset moderation shapes) must be verified against REAL rows with a bounded tiny-real streaming probe + per-filter reject counters before any production corpus launch.

## Workflow gap

- **Bug observed:** issue #1092 P0 streamed WildChat-1M ~40 min and kept 0 of ~1M rows: the language filter compared full names ('English') against 'en'; a required-kwarg TypeError sat armed one branch later; the plan's redaction/moderation filters were unimplemented — synthetic-fixture smokes stayed green throughout.
- **Why it is a workflow gap:** gotchas.md carries the codebase-trap ledger new implementers read; nothing there warns that real chat corpora store full-name language fields / per-dataset moderation shapes, or that synthetic-fixture smokes cannot catch row-shape bugs (the data-ingestion sibling of the #906 mock-seam entry).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new entry "Real-corpus streaming filters: verify against REAL rows (tiny-real probe)":
+ WildChat/LMSYS `language` = FULL names ('English'), not ISO codes; WildChat: top-level + per-turn `redacted`, per-turn `toxic`, `openai_moderation {categories, flagged}`, `detoxify_moderation` continuous-only; verify per dataset via the datasets-server rows API before writing filters.
+ Bounded tiny-real streaming probe (kept cap + TOTAL-streamed cap) with kept>0 assert per dataset; per-filter reject counters in the stream done-line; real-shape fixture test (network-boundary-only fakes).
+ Cross-ref: agent-memory feedback_real_corpus_streaming_filters_tiny_real_probe.md (#1092), feedback_tiny_real_cpu_e2e.md (#906).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'tiny-real' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: faaf1240e761

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: P0 corpus build (issue1092_build_corpus.py)
lesson: Real-corpus streaming filters must be verified against REAL rows before any production launch — WildChat/LMSYS store FULL language names ('English', not 'en') and per-dataset moderation shapes (per-turn openai_moderation {categories, flagged}; WildChat per-turn redacted/toxic bools; detoxify has no boolean), so a filter chain written from assumed field semantics can reject 100% of rows while every synthetic-fixture smoke stays green. Run a bounded tiny-real streaming probe (kept cap + a TOTAL-streamed-rows cap so a broken chain terminates) and log per-filter rejection counters in the done line so the next 0-kept run names its rejecting filter instantly.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
