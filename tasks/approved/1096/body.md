---
title: 'workflow-fix: llm-judging.md judge max_tokens sizing rule (truncation censoring
  mimics selection artifact)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-07T05:56:47Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #1090 free-analysis round: 64-token judge cap
  truncates reason-first responses; drop rule censors 10-40% of draws asymmetrically'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on
task #1090 (emitting agent: experiment-implementer, Step 9a-ter free-analysis round).

## Goal

Add a judge-response max_tokens sizing guideline to `.claude/rules/llm-judging.md`:
reason-then-score rubrics (rule 7) need a response-token budget sized for the
reasoning; a 64-token cap silently truncates reason-first judge responses before
the score token, and the drop-never-coerce rule (rule 9) then discards them —
censoring 10-40% of draws with an arm-asymmetric pattern that mimics a selection
artifact.

## Workflow gap

- **Bug observed:** #1090's Tier-2 sycophancy judging dropped 473/1000 base vs
  307/1000 trained draws as parse errors; a refresh at max_tokens=300 recovered
  98.8%, proving the drops were 64-token truncation of reason-first responses
  (0 refusals). The asymmetric censoring initially read as a possible
  selection artifact on the headline install delta.
- **Why it is a workflow gap:** llm-judging.md rules 7 (reason-then-score) and 9
  (drop-never-coerce) interact to censor draws when the caller's max_tokens is
  too small, and no rule warns about sizing the response budget; the trap
  plausibly recurs at every graded-judge call site with a reasoning rubric.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ New guideline in §C (prompt/rubric design) or §B: "Size judge max_tokens for
+ the rationale, not the score: a reason-then-score rubric needs >=~300 response
+ tokens; a truncated response parses as an error and rule 9 drops it —
+ arm-asymmetric truncation censoring mimics a selection artifact (incident
+ #1090: 47%/31% draw drops at a 64-token cap, 98.8% recovery at 300). Report
+ the per-arm drop rate alongside every judged DV (extends rule 18)."
+ Cross-ref: src/explore_persona_space/eval/graded_judge.py judge_graded now
+ threads a max_tokens kwarg (default 64 for cache-stability; callers with
+ reasoning rubrics pass ~300).

## Scope / surfaces

- Primary target: `.claude/rules/llm-judging.md`
- Grep `max_tokens` in `.claude/rules/*.md` + `docs/api_throughput_guidelines.md` for sibling hits; update consistently.

## Constraints / invariants

- Workflow-surface only; workflow_lint passes.

## Provenance

- workflow_fix_target: .claude/rules/llm-judging.md
- fingerprint: (compute at filing)

Verbatim prose follow-up: "c-cells judged at 64 tokens systematically lose ~10% (formatting) to ~40% (sycophancy) of draws — future judge_graded callers with reason-heavy rubrics should pass a larger max_tokens."
