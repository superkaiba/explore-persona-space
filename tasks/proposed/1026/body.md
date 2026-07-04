---
title: 'daily-fix: upload-verifier in flight blocks finalize/status-advance'
kind: infra
tags:
- wf-fix
- wf-fix-fp:898da74cb743
- daily-auto-filed
created_at: '2026-07-04T22:11:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 backfill route-2: #778 flipped to interpreting +
  finalized the pod on an agent-level PASS fallback while the upload-verifier was
  still running (its v2 verdict later came back FAIL); only an explicit upload-verification
  PASS may satisfy the Step 8 gate'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-01 backfill problem sweep (route 2) from the #778 r4 upload round (sessions 0b873a86 / f0f20be3, 18:13–18:59Z).

## Goal

Close the fallback loophole that let a session flip status to `interpreting` and finalize (terminate) the pod while the upload-verifier was still running.

## Workflow gap

- **Bug observed:** on #778 (2026-07-01), status advanced to `interpreting` and the pod was finalized on an "agent-level PASS fallback since no artifact declaration" while the upload-verifier was still in flight; its `epm:upload-verification v2` verdict then came back FAIL (the FAIL itself was a marker-string defect — a phantom WandB entity URL — data was safe, but the ordering violation is real: with a genuine upload gap the pod would already have been gone).
- **Why it is a workflow gap:** the standing contract (CLAUDE.md § agents-vs-skills composition diagram + upload-verifier spec) is "no interpretation PUBLISHED before upload-verification PASS, and pod termination strictly requires PASS" — but the /issue Step 8 flow apparently permits an agent-level fallback verdict to satisfy the gate when the verifier has produced no artifact declaration yet, i.e. the gate can be satisfied by absence-of-verdict.
- **Confidence (emitter):** medium (exact fallback wording came from session prose; the planner should locate the precise Step 8 / finalize text or code path).

## Proposed change (refine in planning)

In `.claude/skills/issue/SKILL.md` Step 8 (and the finalize path it drives): an in-flight upload-verifier is BLOCKING for status-advance and pod-finalize — a missing/undeclared artifact list is a WAIT (or a verifier re-spawn), never a PASS-equivalent; only an explicit `epm:upload-verification` PASS satisfies the gate. Add the negative case to whatever mechanical check guards the transition.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 8 results-landed batch / gate ordering)
- Check whether `scripts/dispatch_issue.py` / the finalize helper encodes the same fallback; fix both if so.

## Constraints / invariants

- Workflow-surface only; `workflow_lint.py --check-asks` / `--check-references` pass; do not introduce a new user gate (the fix is ordering, not approval).
