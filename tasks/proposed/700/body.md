---
title: 'rerun-discipline: confirm a fix''s code-path is reached + differential-diagnose
  a hang before reprovisioning'
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:08Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: autonomous sessions blindly re-run failed launches without confirming the fix engaged or diagnosing the failure.

## Goal

Add workflow guardrails so an autonomous experiment session does not blindly re-run a failed launch without proving the fix's code path is reached and differential-diagnosing a hang first.

## Problem (from /daily 2026-06-27)

1. #664 OOM crash-loop relaunched with IDENTICAL strategy (r4 fp32 `_convert_to_fp32`, 57GiB; session bfd0775d); chunk-500/50 relaunched on a FALSIFIED hypothesis — no `[vllm-chunk]` log line meant the hang preceded the first chunk (session 2c432067).
2. The ~8-launch #664 saga burned substantial GPU-hours on an undiagnosed vLLM `generate()` hang before anyone ran py-spy (sessions 2c432067 / b3489bdb).

## Proposed change

- experiment-implementer.md — before re-running a "fixed" failure, assert the expected log line proving the fix's code path is actually reached (a missing marker means the fix didn't engage, so don't reprovision).
- experimenter.md / .claude/rules/gotchas.md — a hang at vLLM `generate()` triggers py-spy / enforce_eager / prefix_caching differential diagnosis on ONE pod BEFORE reprovisioning a fresh multi-GPU pod.

## Scope / target files

- `.claude/agents/experiment-implementer.md`
- `.claude/agents/experimenter.md`
- `.claude/rules/gotchas.md`

## Constraints

- Workflow-surface only — no experiment code, configs, or tasks/.
- Lint gate green (`workflow_lint.py`); touched-file ruff clean if any.
- Keep CLAUDE.md / workflow.yaml consistent if a rule reference is added.
