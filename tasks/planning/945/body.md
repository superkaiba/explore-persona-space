---
title: 'workflow-fix: figure-sidecar opaque-code lint for clean-results'
kind: infra
tags:
- wf-fix
- wf-fix-fp:70282c477377
created_at: '2026-07-03T18:29:12Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #920 r1 mechanizable sketch: sidecar/rendered-text
  scan for @L<digits> + regime-code slug tokens in clean-result figure titles/labels
  (slug-titled figure leaked through 3 review passes to the final gate)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a mechanizable-fix sketch surfaced by the `clean-result-critic` on task #920 round 1 (emission suppressed agent-side under AUTO_REVIEW_DISABLED; routed by the orchestrator).

## Goal

Add a `verify_task_body.py` check (or an `audit_clean_results_body_discipline.py` pass) that scans clean-result figures' sidecar meta.json / rendered text for opaque config-code tokens — `@L<digits>` layer pins and bare regime-code slugs (e.g. `ctx_blk_max`, `ans_uhdr_max`) — in figure titles/labels, WARNing (or FAILing under the no-opaque-condition-codes rule) before the 9a-bis gate.

## Workflow gap

- **Bug observed:** on #920, `winning_cell_scatter.png` reached the 9a-bis clean-result gate titling itself `ctx_blk_max@L12 × ans_uhdr_max@L12` after THREE prior review passes each noted it only as a cosmetic nit (interp-critic r2, humanize critic, code-review) — it finally cost a REVISE round at the final gate (both ensemble critics flagged it, Lens 3).
- **Why it is a workflow gap:** the no-opaque-condition-codes rule (memory `feedback_no_opaque_condition_codes.md` + SPEC figure lenses) has no mechanical enforcement on FIGURE-side text; only prose is scanned, so slug-labeled figures repeatedly leak to the final gate.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_figure_label_codes(body, figure_meta_paths) -> CheckResult:
+     # For each inline figure, read its sidecar meta.json (title/labels fields
+     # written by paper_plots) and scan for r"@L\d+" and r"\b(ctx|ans|pos)_[a-z0-9_]+\b"
+     # style regime-code tokens; WARN listing each figure + offending token.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Also audit `scripts/audit_clean_results_body_discipline.py` + the SPEC figure-lens text; grep `.claude/` for the no-opaque-codes rule surfaces and cross-link.

## Constraints / invariants

- Workflow-surface only. WARN-tier by default (figures without sidecars skip cleanly); ruff + existing verifier tests pass.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 70282c477377

Surfaced prose (clean-result-critic #920 r1, verbatim): "`mechanizable: yes` (sidecar-text scan for `…@L\d+` / bare regime-code tokens) — not emitted as a workflow-fix candidate this turn per the `AUTO_REVIEW_DISABLED=1` sentinel; the sketch is in the verdict for the orchestrator to route."
