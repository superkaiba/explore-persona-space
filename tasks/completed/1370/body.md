---
title: 'workflow-fix: footer Reused bullets must carry pinned paths (verify Check
  35)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:69a2799b4a60
created_at: '2026-07-15T23:44:26Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 candidate on #1315: footer reuse bullets without
  pinned paths invisible to Check 35'
workflow: v1
---
## Overview / Motivation
Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised by the clean-result-critic on task #1315 (round 1).
## Goal
Add a footer-side trigger to verify_task_body.py's cross-issue reuse-pins check (Check 35, #1256): when the `**Repro:**` footer contains a `- Reused ... from [#M](...)` bullet, FAIL (or WARN) unless the bullet contains an HF `/tree/<sha>` / `@<sha>` URL or an `eval_results/issue_M/...` path.
## Workflow gap
- **Bug observed:** On #1315 the check graceful-skipped ("no cross-issue reuse pins in committed result-JSON metadata") even though the body's footer carried two `- Reused ... from [#1090]` bullets with NO pinned HF path/revision — the exact defect the check exists to catch went unchecked; caught only by LM lens judgment.
- **Why it is a workflow gap:** The check triggers only on reuse pins found in committed result-JSON metadata, so a footer reuse bullet authored without any pinned path is invisible to it.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "cross-issue reuse pins" scripts/verify_task_body.py` → 2 hits (lines 675, 4943 — Check 35 presence confirmed); no footer-side trigger exists in the check body (absence claim, inspected at filing) (2026-07-15)
## Proposed change (candidate diff sketch — refine in planning)
+ # in check_cross_issue_reuse_pins (or a sibling check):
+ footer = _extract_repro_footer(body)
+ for bullet in re.findall(r"^- Reused .*?from \[#\d+\]\(.*?\).*$", footer, re.M):
+     if not re.search(r"(/tree/[0-9a-f]{7,40}|@[0-9a-f]{7,40}|eval_results/issue_\d+/)", bullet):
+         fails.append(f"footer Reused bullet lacks a pinned path: {bullet[:80]}")
## Scope / surfaces
- Primary target: `scripts/verify_task_body.py` (Check 35 region, line ~4943); update `tests/test_verify_task_body.py` with fixture pairs.
## Constraints / invariants
- Workflow-surface only; existing green bodies must not newly FAIL (consider WARN-first rollout); recursion guard applies (EPM_WORKFLOW_FIX_SESSION=1).
## Provenance
- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 69a2799b4a60
