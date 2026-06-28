---
title: 'Fire failure-lesson capture on root-cause-confirmed + add supersedes (close #664/r13 gap)'
kind: infra
tags: []
created_at: '2026-06-28T08:47:38Z'
has_clean_result: false
origin_prompt: 'PM-session audit of #664 vLLM bug-bounce: the r13 tokenizer-reload root cause was never durably captured while the chunk-it mis-diagnosis was promoted to gotchas.md. Fire failure-lesson capture on root-cause-confirmed (not only round-resolved) + add a supersedes: field to retract mis-diagnoses.'
---

## Overview / Motivation

The `epm:failure-lesson` capture step fires only when "a crash-fix round
RESOLVES the failure" (round resolved + relaunch succeeded). This missed the
#664 r13 root cause: the real cause of the multi-relaunch "vLLM hang" was
per-call `AutoTokenizer.from_pretrained` reloads inside `_render()` (fixed
with `lru_cache`), found in r13 — but r13 immediately hit the next p2 bug, so
the "this round resolved X, here's the lesson" capture never fired. Result:
the WRONG intermediate lessons (chunk-it / vLLM-EngineCore-deadlock) were
captured durably into `gotchas.md`, and the CORRECT root cause is in no
durable file. Worse, there is no retraction path: when an early lesson is a
mis-diagnosis, nothing supersedes it.

## Goal

(1) Fire failure-lesson capture on ROOT-CAUSE-CONFIRMED, not only on
round-resolved-and-relaunch-succeeded. (2) Add a `supersedes:` field so a
corrected diagnosis retracts/annotates the earlier wrong lesson in the durable
record.

## Scope / surfaces

- `.claude/skills/issue/SKILL.md` — the "Failure-lesson capture" step:
  add the root-cause-confirmed trigger (a round that identifies the true cause
  posts the lesson even if a subsequent, distinct failure follows) and the
  `supersedes: <prior-lesson-slug|marker-ts>` field.
- `.claude/agents/experiment-implementer.md` + `.claude/agents/experimenter.md`
  — the failure-lesson block spec: add `supersedes:` and the
  root-cause-confirmed firing condition.
- `.claude/workflow.yaml` — the `epm:failure-lesson` marker schema: add
  `supersedes` (optional).
- Consumer side: when a lesson carries `supersedes:`, the agent-memory write /
  gotcha promotion must update-or-annotate the superseded entry, not just
  append (coordinate with the deterministic consolidator task).

## Constraints / invariants

- Workflow-surface only.
- Backward compatible: lessons without `supersedes:` behave exactly as today.
- `scripts/workflow_lint.py --check-asks` + ruff pass; if `workflow.yaml` or
  `CLAUDE.md` change, keep them consistent with the rule/agent files.

## Acceptance

- A two-round fixture (round A posts a mis-diagnosis lesson; round B posts the
  corrected lesson with `supersedes:`) results in the durable record carrying
  the corrected cause and the superseded one marked/annotated, not both as
  co-equal gotchas.
- A root cause confirmed in a round that then hits a NEW distinct failure
  still produces a captured lesson.

## Provenance note

Surfaced during a PM-session audit of #664's vLLM bug-bounce: the r13
tokenizer-reload root cause was never durably captured while the chunk-it
mis-diagnosis was promoted to gotchas.md. Related (deferred): the #664-specific
gotchas.md correction, held on `predicate-664-lands`.
