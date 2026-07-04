---
title: 'daily-fix: rubric-bearing JudgeCache._hash_key'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fe8ef634bf02
- daily-auto-filed
created_at: '2026-07-04T21:35:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-2: JudgeCache._hash_key still content-only
  (rule 22 / #882 deferred the code fix); #810 contamination evidence'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-02 backfill problem sweep (route 2 —
behavior/logic change, independent review required). Emitting context:
transcript mining of the 2026-07-02 sessions.

## Goal

Add rubric/judge-prompt hash + judge model id to JudgeCache._hash_key so cached judgments cannot leak across rubrics

## Workflow gap

- **Bug observed:** JudgeCache._hash_key keys on (question, completion) only; refusal-rubric judgments leaked into harmful_compliance E0 scores on #810 (64 percent of high scores shared exact reasoning/score pairs with the refusal file)
- **Why it is a workflow gap:** llm-judging.md rule 22 (#882) documented the keying requirement but explicitly deferred the batch_judge.py code fix as a separate follow-up that was never filed; the per-rubric cache_dir is only an accepted surrogate
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
- def _hash_key(question: str, completion: str) -> str:
-     content = f"{question}\n---\n{completion}"
+ def _hash_key(question, completion, rubric_key) -> str:
+     # rubric_key = sha256 of the full judge prompt (+ judge model id)
+     content = f"{rubric_key}\n---\n{question}\n---\n{completion}"
  (thread rubric_key through get()/put(); bust or namespace existing
  content-keyed cache dirs so stale cross-rubric entries cannot be read)
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/eval/batch_judge.py`
- Grep the workflow surface for the pattern before editing and update every
  hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/eval/batch_judge.py
- fingerprint: fe8ef634bf02
- related_task: #810
- source: /daily 2026-07-02 backfill problem sweep (route 2)
