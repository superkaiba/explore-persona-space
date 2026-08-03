---
title: 'workflow-fix: verify_task_body per-result multi-figure (non-pair) check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a7bcd9d11aea
created_at: '2026-07-30T10:52:44Z'
has_clean_result: false
origin_prompt: 'clean-result-critic fu1 re-gate on #1769, lens 9 mechanizable finding:
  extend verify to FAIL >1 non-paired inline figure per ### <result>'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a mechanizable clean-result-critic finding raised on task #1769 (emitting agent: clean-result-critic, fu1 re-gate round, 2026-07-30).

## Goal

Add a `verify_task_body.py` check flagging a single `### <result>` H3 under `## Results` that carries MORE THAN ONE inline `![...]` figure when the figures are not a declared raw+processed (aggregate + per-unit) pair — mechanizing the lens-9 one-result-one-figure rule's multi-figure side.

## Workflow gap

- **Bug observed:** task #1769's fu1 fold placed two distinct-analysis figures (`fig_dose_ladder.png` + `fig_alpha3_lattice.png`) inside one `### <result>` section; `verify_task_body.py` PASSed (the only flag was an unrelated word-count WARN) and the violation was caught only by the LM clean-result-critic (fu1 re-gate REVISE, lens 9, `mechanizable: yes`).
- **Why it is a workflow gap:** the v4 spec's one-result-one-figure three-beat allows >1 figure per `### <result>` only for a raw-alongside-processed pair, but no mechanical check counts inline figures per result section — the existing figure checks are per-figure (caption shape, panel drift, count claims), never per-section cardinality — so a two-analysis section passes silently and burns an LM-critic round.
- **Confidence (emitter):** high (the critic tagged it `mechanizable: yes` and named the fix).
- verified-at-filing: `grep -n "one-result-one-figure\|one_figure" scripts/verify_task_body.py` → 11 hits, ALL per-figure helper functions (`_panel_drift_for_one_figure`, `_prose_numerics_for_one_figure`, `_beat_claims_for_one_figure`, `_count_claims_for_one_figure`) — no per-`###` figure-count rule; `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → 1 commit (218a6d0425, #1493 N/A-verdict for infra/batch/survey — unrelated) (2026-07-30)

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_result_sections_single_figure(...):
+     for each `### <result>` under `## Results` (v4-sentinel bodies only):
+         n = count of inline `![alt](url)` lines in the section
+         if n > 1 and not a declared raw+processed pair (heuristic: one
+             figure's alt/caption names a per-unit/per-question companion of
+             the other, or the prose declares the pair):
+             WARN (or FAIL for v4) "### '<h3>' carries N inline figures that
+                   are not a raw+processed pair (one-result-one-figure)"
+ # WARN-vs-FAIL and the pair heuristic are planning decisions; the
+ # forward-only rule keeps v3/v2/legacy bodies exempt.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Add paired rows in `tests/test_verify_task_body.py`; keep v3/v2/legacy bodies exempt (v4-sentinel-gated per the forward-only rule); update `.claude/skills/clean-results/SPEC.md` mechanical-checks list alongside.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Never a new hard FAIL on grandfathered (non-v4) shapes; `verify_task_body.py` stays consistent with `.claude/skills/clean-results/SPEC.md`.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py

Surfaced finding (verbatim, clean-result-critic fu1 re-gate on #1769): "Lens 9 — `### The dose ladder places the CJK collapse between α=2 and 3…` carries TWO figures (`fig_dose_ladder.png` + `fig_alpha3_lattice.png`) that are two distinct analyses, not a raw+processed pair — the only sanctioned >1-figure exception. [...] `mechanizable: yes` — extend verify check 21/38 to FAIL >1 non-paired inline figure per `### <result>`."
- fingerprint: a7bcd9d11aea
