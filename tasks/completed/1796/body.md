---
title: 'daily-fix: c39 trigger regex misses inverse-direction reads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b4136176b817
- daily-auto-filed
created_at: '2026-07-29T07:09:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): c39''s trigger vocabulary
  (`off-pod`/`vm-side`) cannot fire on inverse-direction cross-phase reads (a GCE/pod
  phase consuming VM-produced inputs), so the #1773 seam stays mechanically un-nudged
  after #1782''s rule generalization'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a formal candidate block parked on task #1782 (ts 2026-07-29T03:29:16Z, fp b4136176b817; emitted by the Phase-2 Methodology critic during #1782's plan review). #1782 generalized the c39 fenced-block SCHEMA to direction-agnostic cross-phase reads (merged 2026-07-28/29, commit `0d3b2cf46b`, "block name + c39 semantics unchanged"); the TRIGGER regex extension was deliberately out of its scope because it requires the c33-precedent corpus re-scan calibration.

## Goal

Extend `verify_plan.py` c39's trigger regex with inverse-direction vocabulary (a GCE/pod phase consuming VM-produced inputs) after a full persisted-plan corpus re-scan calibrates the false-positive rate.

## Workflow gap

- **Bug observed:** c39's trigger vocabulary (`off-pod`/`vm-side`) cannot fire on inverse-direction cross-phase reads (a GCE/pod phase consuming VM-produced inputs), so the #1773 seam stays mechanically un-nudged even after #1782's rule generalization.
- **Why it is a workflow gap:** the check's regex encodes the #1535 direction only; extending trigger vocabulary is a semantics change requiring the persisted-plan corpus re-scan per the c33 calibration precedent, deliberately out of #1782's scope.
- **Confidence (emitter):** low
- verified-at-filing: `grep -n '_C39' scripts/verify_plan.py` → `_C39_TRIGGER_RE = re.compile(r"(?i)\boff-pod\b|\bvm-side\b")` at scripts/verify_plan.py:6840 — exactly the two-direction-blind form the candidate describes (2026-07-29 UTC, read AFTER #1782's `0d3b2cf46b` landed, so the generalization commit did not extend the trigger). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` → `0d3b2cf46b` (schema generalization, trigger untouched), `12311b2bb6` (c43), `4e04015e89` (c38), `7809272e2f`, `4f74756cb9` — none extend the c39 trigger.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- _C39_TRIGGER_RE = re.compile(r"(?i)\boff-pod\b|\bvm-side\b")
+ _C39_TRIGGER_RE = re.compile(r"(?i)\boff-pod\b|\bvm-side\b|\bgit-clone lane\b|\bvm-produced\b")
+ (corpus re-scan + new WARN-text tests per the c33 calibration contract)
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py` (c39 trigger, ~L6840) + its tests
- REQUIRED: persisted-plan corpus re-scan to calibrate false-positive rate before landing (c33 precedent); record the scan result in the plan.

## Constraints / invariants

- c39 stays WARN-only, conditional, experiment-only; block name + semantics unchanged (only trigger vocabulary widens).
- Workflow-surface only; recursion guard applies to the spawned session.

## Provenance

- sha-verify (filing-time, #1467): `b4136176b817` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: b4136176b817

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_plan.py
bug_observed: c39's trigger vocabulary (`off-pod`/`vm-side`) cannot fire on inverse-direction cross-phase reads (a GCE/pod phase consuming VM-produced inputs), so the #1773 seam stays mechanically un-nudged after #1782's rule generalization
why_workflow_gap: the check's regex encodes the #1535 direction only; extending trigger vocabulary is a semantics change requiring the persisted-plan corpus re-scan per the c33 calibration precedent, deliberately out of #1782's scope
proposed_change: extend c39's trigger regex with inverse-direction vocabulary (e.g. `git-clone lane`, `stages? .* from HF`, `VM-produced`) after a full persisted-plan corpus re-scan calibrates false-positive rate
confidence: low
related_task: #1782
<!-- /workflow-fix-candidate -->
