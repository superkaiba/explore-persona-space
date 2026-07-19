---
title: 'workflow-fix: lint bash function-invocation || rc=$? errexit-suppression in
  scripts/*.sh'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9ee07db6a7b7
created_at: '2026-07-18T16:11:16Z'
has_clean_result: false
origin_prompt: 'code-reviewer #1426 sampled-rollout r1 Critical: run_seed "$s" ||
  rc=$? masks every per-seed failure to rc=0 (errexit suppressed in-function); Mechanizable:
  yes — grep/lint for multi-step functions invoked via || rc=$? under set -e in scripts/*.sh'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1426 (emitting agent: code-reviewer, sampled-rollout round r1, mechanizable: yes).

## Goal

Add a workflow_lint check flagging a same-file bash FUNCTION invoked via `func || rc=$?` (or `|| true`) under `set -e` in `scripts/*.sh` — errexit is suppressed inside the function body, so mid-function failures collapse to the last command's rc.

## Workflow gap

- **Bug observed:** the #1426 sampled-rollout dispatch shipped `run_seed "$s" || rc=$?` — a Gate-1 terminal failure (driver return 3), the manifest-validation SystemExit, and fit failures all collapsed to rc=0, letting unvalidated fits + partial uploads + the success sentinel `[phase=done]` proceed. Caught only by the adversarial code-reviewer (round-1 FAIL).
- **Why it is a workflow gap:** the shape is a recurring bash footgun (errexit suppression in `||`/`&&`/`if` contexts) that no mechanical gate catches; workflow_lint already names `|| rc=$?`-then-ignore as a "residual evasion shape" in another check's docstring (lines 1199/3851) but has no function-invocation detector.
- **Confidence (emitter):** high (reviewer reproduced the failure shape live on this VM)
- verified-at-filing: `grep -rn '|| rc=\$?' scripts/*.sh` → 13 hits in 13 files (2026-07-18), ALL current hits are single-external-command captures (safe — the captured rc IS the command's rc); 0 current function-invocation instances on main (the #1426 instance lives on the unmerged issue-1426-sampled branch, being fixed in its revision round); the check targets the CLASS to prevent recurrence. `grep -n 'rc=\$?' scripts/workflow_lint.py` → docstring mentions only (no existing check for this class).

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_sh_function_rc_capture(...):
+     # per scripts/*.sh: collect function names (^\s*(function\s+)?(\w+)\s*\(\)\s*\{)
+     # flag lines matching ^(\s*)(<fname>)\b.*\|\| (rc=\$\?|true) when set -e is
+     # active in the file; single external commands (non-function LHS) unflagged;
+     # inline waiver # RC_CAPTURE_EXEMPT: <reason ≥10 chars>
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (+ its tests in tests/test_workflow_lint.py)
- Decide bundling into the no-flags default run per the sibling checks' policy.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` + no-flags run stay green; new check must pass on the live tree (13 current single-command uses unflagged).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 9ee07db6a7b7

Surfaced prose (code-reviewer #1426 r1): "Mechanizable: yes (grep/lint for multi-step functions invoked via `|| rc=$?` under set -e in scripts/*.sh)."
