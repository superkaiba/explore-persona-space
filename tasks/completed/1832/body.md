---
title: 'workflow-fix: verify_task_body WARN for quantitative result sections with
  zero figures'
kind: infra
tags:
- wf-fix
- wf-fix-fp:720a71791e74
created_at: '2026-07-29T16:46:22Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1769 r1 prose follow-up (b): figure-less quantitative
  headline result passed the mechanical verifier silently'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1769 (emitting agent: clean-result-critic, round 1, 2026-07-29).

## Goal

Add a `verify_task_body.py` WARN when a quantitative `### <result>` H3 under `## Results` (number-dense prose or a table) carries zero inline `![...]` images, so a figure-less headline result no longer passes the mechanical verifier silently.

## Workflow gap

- **Bug observed:** task #1769's headline Result 5 (the α=2 three-treatment lattice carrying the H1 title claim) had ZERO inline figure and a `> **Figure.**` caption mislabeling a table; `verify_task_body.py` PASSed (WARNs only) and the gap was caught only by the LM clean-result-critic (round-1 REVISE, lens 3/9).
- **Why it is a workflow gap:** check 4 requires ≥1 image per body SECTION and the per-result caption checks only audit results that already HAVE figures — no check binds a figure-less quantitative result section, though the v4 spec's one-result-one-figure three-beat requires one per `### <result>`.
- **Confidence (emitter):** medium.
- verified-at-filing: `grep -n "image\|!\[" scripts/verify_task_body.py` → check 4 (hero image, line ~2494, section-level ≥1) + caption checks keyed to existing images; no per-result zero-image rule; `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → 3 recent commits, none touching per-result figure presence (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_result_sections_have_figures(...):  # WARN, not FAIL
+     for each `### <result>` under `## Results`:
+         if the section prose is quantitative (≥3 numeric tokens or a GFM table)
+         and contains no `![alt](url)` image line:
+             WARN "quantitative result section '<h3>' carries no inline figure
+                   (v4 one-result-one-figure three-beat)"
+ # WARN (not FAIL) so grandfathered/edge bodies never hard-block; the
+ # clean-result-critic remains the judgment layer.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Add the paired rows in `tests/test_verify_task_body.py`; keep v3/v2/legacy bodies exempt (v4-sentinel-gated per the forward-only rule).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- WARN-only (never a new hard FAIL on grandfathered shapes); `verify_task_body.py` stays consistent with `.claude/skills/clean-results/SPEC.md` (update SPEC.md mechanical-checks list alongside).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 720a71791e74

Surfaced prose (verbatim, clean-result-critic round 1 on #1769): "add a `verify_task_body.py` WARN for a quantitative `### <result>` (number-dense prose) carrying zero `![...]` images — check 21 only audits results that already have figures, so a figure-less headline result passes silently." — filer note: the check the critic cites as 21 is the caption/params family; the gap statement itself verified genuine.
