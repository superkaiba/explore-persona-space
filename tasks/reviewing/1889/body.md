---
title: 'workflow-fix: inline lint gate blind to untracked payload files'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b63d128dcc37
- urgent-main-red
created_at: '2026-07-30T18:55:26Z'
has_clean_result: false
origin_prompt: 'map963k-reuse correction marker 2026-07-30T18:52:31Z on #1739: pre-commit/inline
  gate misses brand-new untracked scripts because test_shared_vm_thread_caps enumerates
  git-tracked files only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose-surfaced candidate raised on task #1739 (emitting agent: map963k-reuse inline subagent, correction-round marker 2026-07-30T18:52:31Z).

## Goal

Make the inline payload lint gate visible to brand-new UNTRACKED payload files: stage the payload (or pass it explicitly) before running mapped tests that enumerate tracked files only.

## Workflow gap

- **Bug observed:** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` enumerates git-TRACKED files via `git ls-files`, so a brand-new, still-untracked script carrying a heavy-import-before-load_dotenv violation passes a manually-run gate/pytest and lands red on the branch. Realized twice on 2026-07-30: the map963k round committed two violating scripts at `606278aa38` (self-caught + fixed at `b21f109337`), the #1388 fleet-red class.
- **Why it is a workflow gap:** the inline payload lint gate (`scripts/inline_lint_gate.py`) runs the mapped tests against the working tree WITHOUT ensuring the payload files are visible to tracked-file-enumerating invariant tests — a structurally invisible violation class for exactly the NEW files the gate exists to certify. (A staged file IS caught — the 15:44Z run this session caught a staged new script — so the fix is to guarantee visibility, e.g. verify each payload path is tracked/staged before the pytest leg, or thread the payload list into the enumerating tests via env.)
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "ls-files" tests/test_shared_vm_thread_caps.py` → 2 hits (lines 842, 910: `["git","ls-files"]` enumeration) AND `grep -n "git add\|ls-files\|untracked\|staged" scripts/inline_lint_gate.py` → 0 hits (gate has no payload-visibility handling) (2026-07-30). Per-target: both named files confirmed as the mechanism sites.

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/inline_lint_gate.py, before running mapped tests: for each payload path,
+ verify `git ls-files --error-unmatch <path>` OR `git diff --cached --name-only`
+ contains it; if untracked, either fail loud with a "stage your payload first"
+ instruction or export EPM_EXTRA_ENTRYPOINT_FILES=<paths> consumed by the
+ tracked-file enumeration in tests/test_shared_vm_thread_caps.py (union).

## Scope / surfaces

- Primary target: `scripts/inline_lint_gate.py, tests/test_shared_vm_thread_caps.py`
- Grep the workflow surface for other tracked-only enumerating invariant tests before editing (`grep -rln "ls-files" tests/`) and cover them uniformly if the mechanism generalizes.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes.
- This session runs under a workflow_fix_target Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/inline_lint_gate.py, tests/test_shared_vm_thread_caps.py
- fingerprint: b63d128dcc37

Surfaced prose (verbatim, from the map963k-reuse correction marker): "Why the pre-commit gate missed it: that test enumerates git-TRACKED files, and the round's scripts were still untracked when the gate ran, so a brand-new script's violation is structurally invisible to the pre-commit inline payload lint gate. Reported as a workflow-surface gap."
