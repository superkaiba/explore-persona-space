---
title: 'daily-fix: self-derive WORKFLOW_INVARIANT count pin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7bc4514f3df4
- daily-auto-filed
created_at: '2026-07-22T06:45:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): hardcoded WORKFLOW_INVARIANT
  count pin collides whenever two workflow-fix PRs land in the same window'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from the #1584 Step 10d merge-conflict incident (transcript c2f2fe1f).

## Goal

Make the Step-9c `WORKFLOW_INVARIANT` registry count self-derived (or union-merge-friendly) so two concurrent workflow-fix PRs registering tests no longer collide on a hardcoded integer pin.

## Workflow gap

- **Bug observed:** #1584's PR #1353 squash-merge failed twice ("Pull Request has merge conflicts") because #1575 had just merged bumping the invariant count pin 41→42 while #1584's branch bumped the same pin — the resolution needed a fresh-context conflict subagent to compute the semantic union 42→43 (~15 min, 09:17→09:32Z).
- **Why it is a workflow gap:** a hardcoded COUNT pin that every registrant bumps is a guaranteed collision point whenever ≥2 workflow-fix PRs land in the same window — and multiple same-window workflow-fix PRs are now the nightly norm (12 landed 2026-07-21 alone).
- **Confidence:** high (the two `is_error:true` merge failures + resolution are quoted in the transcript).
- verified-at-filing: `grep -rn 'WORKFLOW_INVARIANT' scripts/select_step9c_tests.py tests/ | grep -icE 'len\(|count'` → the pin is a hardcoded integer asserted in the pin test, not derived (2026-07-22; presence of the hardcoded-pin pattern confirmed in-target). NOT a duplicate of open #865 (worktree-blind diffing) nor of tonight's `step9c-selector-self-diff-zero` filing (selector-self-mapping miss) — three distinct bugs on the same file, each its own fingerprint.

## Proposed change (candidate diff sketch — refine in planning)

Replace the hardcoded registry-count integer with a dynamically computed `len(WORKFLOW_INVARIANT)` assertion (or restructure the pin into a union-merge-friendly per-entry format), preserving whatever regression-detection the count pin was providing (e.g. pin the SET of registered test paths instead of its cardinality).

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Sibling: its pin test (locate via `grep -rln 'WORKFLOW_INVARIANT' tests/`).

## Constraints / invariants

- Workflow-surface only; the selector's safe-by-direction contract preserved; pin test still catches accidental deregistration.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 7bc4514f3df4

- workflow_fix_target: scripts/select_step9c_tests.py

Origin evidence: transcript c2f2fe1f, 09:17:16Z + 09:31:44Z ("GraphQL: Pull Request has merge conflicts (mergePullRequest) MERGE FAILED"), 09:32:42Z ("MERGED (squash, PR #1353)... resolved as the semantic union 42→43 by a fresh-context subagent").
