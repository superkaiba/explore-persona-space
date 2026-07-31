---
title: 'workflow-fix: figure-sidecar opaque-code check misses bare P/M candidate codes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:724e1927a288
created_at: '2026-07-31T15:55:50Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1900 r1 prose follow-up (a): figure-text opaque-code
  token classes miss bare P\d+/M\d+ candidate codes in sidecar legend/title text (scripts/verify_task_body.py)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1900 (emitting agent: clean-result-critic, prose follow-up).

## Goal

Extend the figure-sidecar opaque-code check token classes to flag bare P<digit>/M<digit> candidate and panel codes in sidecar legend/title text.

## Workflow gap

- **Bug observed:** figures with plan-internal candidate codes (P1|P7 legend, P7-residualized titles) passed verify_task_body while the clean-result-critic caught them on task 1900 round 1.
- **Why it is a workflow gap:** the no-opaque-condition-codes rule is mechanically enforced for body prose, and the figure-text arm reads sidecar legend/title text (via `_read_figure_meta_json`, near the check-26 docstring block at ~L497-520), but its token classes do not include bare `P\d+` / `M\d+` candidate/panel codes — so reader-facing figure text with plan codes ships past the mechanical gate and burns an LM-critic round.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "opaque\|sidecar" scripts/verify_task_body.py` → 8+ hits incl. L497 (`check_figure_panel_prose_vs_sidecar`), L520 ("must not carry opaque") — the check site exists; the gap is evidenced by the live incident: `verify_task_body.py --issue 1900` PASSed (n_fail=0) twice on 2026-07-31 while figures/issue_1900/{mediation_forest,residualized_race_content,residualized_race_marker}.png carried "P1 | P7"-style legend/title codes the clean-result-critic then flagged (epm:clean-result-critique v1 on #1900, Must-Fix 1). `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` → 5 commits, none touching the opaque-code token classes (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

```
  # in the figure-sidecar opaque-code scan (the token-class list used against
  # sidecar legend/title/text fields):
- OPAQUE_TOKEN_RES = [ ... existing condition-code classes ... ]
+ OPAQUE_TOKEN_RES = [ ... existing classes ...,
+     r"\b[PM]\d+[ab]?\b",   # bare candidate/panel codes (P1, P7, p3b, M4) in
+                            # rendered legend/title text — #1900 incident
+ ]
  # tier: WARN (consistent with sibling figure-text checks); allowlist
  # legitimate uses (e.g. "P100 GPU", percentiles like "p97.5") via context
  # or an explicit negative lookahead — the planner decides the guard shape.
```

Caveat for the planner: `p97.5`-style percentile text and hardware names must not false-positive; consider requiring the code to appear in a legend/title field (not caption body) or match only `\b[PM]\d{1,2}\b` with a percentile lookahead exclusion.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'opaque' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan (the clean-result SPEC + critic lens reference may
  document the token classes and must stay in sync).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  tests/test_verify_task_body.py extended with a pinning test (positive: bare
  P7 in a sidecar title flags; negative: p97.5 percentile text does not).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 724e1927a288

Surfaced prose (verbatim, clean-result-critic round 1 on #1900): "two `verify_task_body.py` gaps surfaced above as mechanizable — (a) figure-text opaque-code token classes miss bare `P\d+`/`M\d+` candidate codes in sidecar legend/title text (`scripts/verify_task_body.py`)"
