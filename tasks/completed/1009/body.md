---
title: 'workflow-fix: gotchas — vLLM bank length-validation at load'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c5383820c5b5
created_at: '2026-07-04T11:49:18Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #952 r3 (see body Provenance)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #952 (emitting agent: experiment-implementer, round 3).

## Goal

Add a `.claude/rules/gotchas.md` entry: length-validate any external/real-corpus query bank fed to vLLM at LOAD time against `max_model_len − generation cap`, tokenizing the FORMATTED prompt with the same chat template as the generate call; drop matched pairs together; record drops digest-only.

## Workflow gap

- **Bug observed:** One 8,377-token row of 460 in a real-corpus bank hard-crashed vLLM (`ValueError: decoder prompt > max_model_len`) at the first production chunk, seconds after a fully-passing synthetic smoke — burning a GPU provision (#952 attempt 2, GCE att-20260704-103316).
- **Why it is a workflow gap:** gotchas.md carries the LMSYS 8192-truncation caveat but no rule for EXTERNAL banks; every future experiment consuming a real prompt corpus through vLLM re-hits the class (smokes sample small subsets and never catch rare overlong rows).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ **vLLM query banks: length-validate at LOAD time.** Real-corpus banks contain
+ rare overlong rows a smoke subset never samples; vLLM hard-raises on the first
+ formatted prompt over max_model_len mid-production. Filter at the bank LOADER
+ (same chat template + tokenizer as the generate call; budget = max_model_len −
+ max_tokens), drop matched pairs together, record drops digest-only (index +
+ token count + category). Worked example: run_952.filter_bank_rows_by_length (#952).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing: `grep -rn "max_model_len" .claude/rules/ CLAUDE.md` for placement next to the existing truncation caveats.

## Constraints / invariants

- Workflow-surface only; lessons-index untouched (gotchas.md is already indexed).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: c5383820c5b5
