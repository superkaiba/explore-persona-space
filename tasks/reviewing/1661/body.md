---
title: 'daily-fix: agent-spec-size ratchet in pre-commit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7e5383b890db
- daily-auto-filed
created_at: '2026-07-24T06:50:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): a direct-to-main agent-file
  edit over its byte ratchet cap reds the fleet because the size check runs only in
  Step 9c gates, not at commit time'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Incident chain: `.claude/agents/research-pm.md` regrew past its 47,000-byte grandfather ratchet cap via a DIRECT-TO-MAIN commit (47,861 B), redding `test_workflow_lint_default_exits_zero` in every Step 9c gate fleet-wide until #1618 trimmed it (fix commit `8ede3ddb1b`). The lint caught it only downstream — at other sessions' gates — because the direct-to-main path runs no Step 9c.

## Goal

Catch agent-spec-size ratchet breaches AT COMMIT TIME: add the size-cap check to the pre-commit hook set so a direct-to-main workflow-surface edit fails on the committer's machine instead of redding the fleet.

## Workflow gap

- **Bug observed:** the AGENT_SPEC_SIZE ratchet is enforced by `workflow_lint.py`'s no-flags run (Step 9c gates + `test_workflow_lint_default_exits_zero`), but the pre-commit config carries no such hook — a direct-to-main edit (the incident's path) bypasses every gate and turns the breach into fleet-wide red.
- **Why it is a workflow gap:** direct-to-main workflow-surface edits are a supported path (orchestrator workflow fixes commit directly); the enforcement point must sit on the commit, not only on other sessions' gates.
- **Confidence:** high
- verified-at-filing: `grep -n "agent_spec_size\|AGENT_SPEC" .pre-commit-config.yaml` → 0 hits (absence claim, in-target); incident fix commit `8ede3ddb1b` rev-parse-verified; current `wc -c .claude/agents/research-pm.md` = 46,785 B (under cap post-#1618) (2026-07-24 UTC).

## Proposed change (refine in planning)

Add a pre-commit hook (files-scoped to `.claude/agents/*.md`) invoking the existing `workflow_lint.py` agent-spec-size check (or a thin wrapper) so any staged agent file over its ratchet cap blocks the commit with the cap + measured size named.

## Scope / surfaces

- Primary target: `.pre-commit-config.yaml` (+ a `workflow_lint.py` entrypoint flag if needed)

## Constraints / invariants

- The hook must be fast (single-file stat class, no full lint run); recursion guard applies.

## Provenance

- fingerprint: 7e5383b890db

- workflow_fix_target: .pre-commit-config.yaml
