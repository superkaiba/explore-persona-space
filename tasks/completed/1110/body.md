---
title: 'workflow-fix: verify H1 == frontmatter title (set-title never syncs the body
  H1)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:507a7468937c
created_at: '2026-07-07T10:38:50Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r2 formal candidate on #825: H1 vs frontmatter
  title divergence has no mechanical check; fingerprint 507a7468937c'
workflow: v1
---
## Overview / Motivation

Auto-filed from a formal workflow-fix candidate raised by the clean-result-critic
(round 2, fold base-separator-control) on task #825. Confidence: HIGH.

## Goal

Add a verify_task_body.py check (sentinelled bodies only) FAILing when the whitespace-normalized H1 text differs from the frontmatter `title`.

## Workflow gap

- **Bug observed:** On #825 the analyzer retitled via `task.py set-title` (frontmatter + REGISTRY updated) but never synced the body H1, so the dashboard list title and the body H1 diverged for a full critic round with no mechanical catch — the verifier's title check only validates the confidence tag on the H1.
- **Why it is a workflow gap:** the title lives in two places (frontmatter via set-title, H1 via set-body) with no check asserting they agree, so every retitle-without-H1-edit silently ships a stale headline; all 6 recent awaiting_promotion bodies satisfy equality, confirming it is the intended invariant.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_h1_matches_frontmatter_title(body, fm):
+     h1 = next((l[2:].strip() for l in body.splitlines() if l.startswith("# ")), "")
+     ft = " ".join(str(fm.get("title", "")).split())
+     if " ".join(h1.split()) != ft:
+         return FAIL("H1 title differs from frontmatter title — sync via set-title + set-body")
+     return PASS
```
Consider also: a set-title flag (or default) that rewrites the body H1 in the same flock+commit.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (+ optional `scripts/task.py` set-title H1 sync)
- Regression test in tests/test_verify_task_body.py.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 507a7468937c

(Verbatim candidate block preserved in the emitting critic's epm:clean-result-critique v12 marker on task #825.)
