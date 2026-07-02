---
title: 'workflow-fix: judge caches must key on rubric+question+completion (llm-judging.md)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7589708532b4
created_at: '2026-07-02T18:51:37Z'
has_clean_result: false
origin_prompt: 'interpretation-critic #810 r1 prose follow-up: llm-judging.md should
  add a guideline that any judge cache MUST key on (rubric/behavior, question, completion),
  not content alone — JudgeCache._hash_key gap corrupts multi-behavior judged runs
  reusing completions across rubrics'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #810 (emitting agent: interpretation-critic).

## Goal

Add an llm-judging.md guideline: any judge cache MUST key on (rubric/behavior id, question, completion), never completion content alone.

## Workflow gap

- **Bug observed:** `JudgeCache._hash_key` (src/explore_persona_space/eval/batch_judge.py:107-110) omits the rubric/behavior from the cache key, so refusal-rubric judgments leaked into harmful_compliance E0 scores via the shared content-keyed cache dir on task #810 — 64% of high harmful_compliance scores were refusal-rubric judgments; 99.1% of flagged rows share their exact (reasoning, score) pair with the refusal file (vs 18.7% baseline), proving cache leak over a prompt bug.
- **Why it is a workflow gap:** `.claude/rules/llm-judging.md` is the binding recipe every judged-DV plan loads, and it is silent on cache keying — any future multi-behavior judged run that reuses completions across rubrics with a shared cache will silently corrupt scores the same way. The library fix itself is out of workflow-fix scope (src/ eval code); the rule-file guideline is in scope and is what makes planners/critics catch it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ In .claude/rules/llm-judging.md § G (Reproducibility / reporting) or a new
+ § C item: "Judge-cache keying: any judge result cache MUST include the
+ rubric/behavior identifier (plus question + completion) in its key —
+ NEVER completion content alone. A content-only key silently returns
+ another behavior's judgment for the same completion when completions are
+ reused across rubrics (incident #810: JudgeCache._hash_key omitted the
+ rubric; refusal judgments leaked into harmful_compliance E0, 64% of high
+ scores). Planners name the cache key fields; critics REVISE a shared
+ multi-rubric cache without a rubric-bearing key."
```

## Scope / surfaces

- Primary target: `.claude/rules/llm-judging.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'judge cache\|JudgeCache\|cache-dir\|cache_dir' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. (Enforcement pointers in `planner.md` §6 / `critic.md` Statistics & Measurement lens may warrant a one-line cross-reference; the spawned planner decides.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`. The
  `batch_judge.py` library fix is a SEPARATE follow-up (out of scope here).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/llm-judging.md
- fingerprint: 7589708532b4

Surfaced prose (interpretation-critic #810 round 1, verbatim): ".claude/rules/llm-judging.md should add a guideline that any judge cache MUST key on (rubric/behavior, question, completion), not content alone — the JudgeCache._hash_key gap will silently corrupt any multi-behavior judged run reusing completions across rubrics (the library fix itself is out of workflow-fix scope; the rule-file guideline is in scope)."
