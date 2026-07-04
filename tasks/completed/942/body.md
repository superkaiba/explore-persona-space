---
title: 'workflow-fix: audit-availability check misses quota-held denial family + store/maps
  artifact classes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c6e46922bf9f
created_at: '2026-07-03T17:51:10Z'
has_clean_result: false
origin_prompt: '<!-- workflow-fix-candidate v1 -->

  target_file: scripts/verify_task_body.py

  bug_observed: The audit-availability check (check 25) PASSed vacuously on #813''s
  body, which asserts data artifacts "remain on the pod under an HF public-storage
  quota hold (upload 403)" while an exact-count upload-verification PASS + independent
  list_repo_files prove all 24,206 artifacts are on HF.

  why_workflow_gap: `_AUDIT_DENIAL_RE` omits the "quota-held / quota hold / upload
  403 / remain on the pod" denial family, and `_AUDIT_ARTIFACT_CLASSES` omits the
  activation-store / reduced-store / maps artifact classes, so a false availability-denial
  near a data artifact escapes the mechanical gate.

  proposed_change: Extend `_AUDIT_DENIAL_RE` with the quota-hold/403/"remain on the
  pod" phrasings and add unreduced-store / reduced-store / maps entries (with HF-path
  convention) to `_AUDIT_ARTIFACT_CLASSES` so the check resolves them against the
  data repo.

  confidence: medium

  related_task: #813

  <!-- /workflow-fix-candidate -->'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #813 (emitting agent: interpretation-critic).

## Goal

Extend `_AUDIT_DENIAL_RE` with the quota-hold/403/"remain on the pod" phrasings and add unreduced-store / reduced-store / maps entries (with HF-path convention) to `_AUDIT_ARTIFACT_CLASSES` so check 25 resolves them against the data repo.

## Workflow gap

- **Bug observed:** The audit-availability check (check 25) PASSed vacuously on #813's body, which asserts data artifacts "remain on the pod under an HF public-storage quota hold (upload 403)" while an exact-count upload-verification PASS + independent list_repo_files prove all 24,206 artifacts are on HF.
- **Why it is a workflow gap:** `_AUDIT_DENIAL_RE` omits the "quota-held / quota hold / upload 403 / remain on the pod" denial family, and `_AUDIT_ARTIFACT_CLASSES` omits the activation-store / reduced-store / maps artifact classes, so a false availability-denial near a data artifact escapes the mechanical gate.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
_AUDIT_DENIAL_RE = re.compile(r"(?:not\s+uploaded|...|cannot\s+be\s+audited"
+   r"|quota[- ]?held|quota\s+hold|upload\s+403|remain(?:s|ing)?\s+on\s+the\s+pod"
+   r"|pod[- ]side\b.*(?:quota|held))", re.IGNORECASE)
_AUDIT_ARTIFACT_CLASSES = { ...,
+   "unreduced": re.compile(r"unreduced\s+(?:activation\s+)?store", re.I),
+   "reduced": re.compile(r"reduced\s+(?:c_C/v_A\s+)?summ|reduced\s+store", re.I),
+   "maps": re.compile(r"\bfitted[- ]maps?\b|\bmap\s+factored\s+forms?\b", re.I),
}
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '_AUDIT_DENIAL_RE\|_AUDIT_ARTIFACT_CLASSES' .claude/ CLAUDE.md scripts/ tests/`) and update every hit;
  list them in the plan (expect `tests/test_verify_task_body.py` coverage too).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: c6e46922bf9f

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: The audit-availability check (check 25) PASSed vacuously on #813's body, which asserts data artifacts "remain on the pod under an HF public-storage quota hold (upload 403)" while an exact-count upload-verification PASS + independent list_repo_files prove all 24,206 artifacts are on HF.
why_workflow_gap: `_AUDIT_DENIAL_RE` omits the "quota-held / quota hold / upload 403 / remain on the pod" denial family, and `_AUDIT_ARTIFACT_CLASSES` omits the activation-store / reduced-store / maps artifact classes, so a false availability-denial near a data artifact escapes the mechanical gate.
proposed_change: Extend `_AUDIT_DENIAL_RE` with the quota-hold/403/"remain on the pod" phrasings and add unreduced-store / reduced-store / maps entries (with HF-path convention) to `_AUDIT_ARTIFACT_CLASSES` so the check resolves them against the data repo.
confidence: medium
related_task: #813
<!-- /workflow-fix-candidate -->
