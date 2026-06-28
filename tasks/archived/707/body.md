---
title: spend-approval park must stay sticky across resume (not un-parked by a config-drift
  cap bump)
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-06-28T07:14:04Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog (NEEDS THOMAS'S GREENLIGHT — plan-gate approval contract): a once-parked spend gate auto-approved itself on resume.

## Goal

A once-parked user-facing spend gate must NOT auto-approve itself on resume (e.g. via a config-drift cap bump).

## Problem (from /daily 2026-06-27)

#653's spend-approval gate (132 GPU-h > 100h cap) silently un-parked on resume because the respawned session's env carried `EPM_PLAN_AUTOAPPROVE_GPU_HOURS=150` (session 62c6451a). A once-parked spend gate auto-approving on a cap bump is a spends-money path that bypasses the human approval the gate exists to enforce.

## Proposed change

NEEDS THOMAS'S GREENLIGHT — touches the plan-gate approval contract (public-contract change). A user-facing spend park should require an explicit user-approval marker to clear, NOT a config-drift cap value. The respawned session should re-read the park state and require the explicit approval marker rather than re-evaluating the GPU-h cap against a possibly-bumped env var.

## Scope / target files

- `.claude/skills/issue/SKILL.md` Step 2c (plan-approval gate)
- `spawn_session.py` env handling

## Constraints

- Architectural / public-contract change to the plan-approval gate — DO NOT auto-dispatch; PM surfaces to Thomas for greenlight.
- Workflow-surface only.
- If implemented later, lint gate green; keep CLAUDE.md / workflow.yaml gate descriptions consistent.
