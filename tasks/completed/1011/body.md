---
title: 'workflow-fix: verify_task_body WARN on orphaned committed per-context companion
  figures'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eb379c0f76e4
created_at: '2026-07-04T15:32:49Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #928 r3 prose follow-up: WARN-level verify_task_body.py
  heuristic — round-committed figures/issue_<N>/*percontext* PNG exists at a body-cited
  figure SHA but is unreferenced by any body image (glob committed round figures,
  diff against body image URLs, WARN on per-unit-named orphans)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #928 (emitting agent: clean-result-critic, round-3 delta-scoped re-gate, 2026-07-04).

## Goal

Add a WARN-level heuristic to `scripts/verify_task_body.py`: a round-committed `figures/issue_<N>/*percontext*|*per_context*` PNG exists at a body-cited figure SHA but is unreferenced by any body image URL.

## Workflow gap

- **Bug observed:** the #928 clean-result body embedded only the pooled aggregate figure for the new MLP result while the round-committed per-context companion PNG (`figures/issue_928/mlp_indiv_percontext_delta.png` at 72a8770009, a SHA the body already cited) existed unreferenced; no mechanical check flags orphaned per-unit figures, so the gap reached the round-3 clean-result-critic as a Lens 11 blocker instead of being caught pre-gate.
- **Why it is a workflow gap:** Lens 11 (underlying data alongside every aggregate) has an easily mechanizable sub-case — "the per-unit companion was produced and committed but not embedded" — that the mechanical pre-pass (`verify_task_body.py`) does not cover, forcing an LM-critic round + analyzer bounce for a purely mechanical omission.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_orphaned_per_unit_figures(body, issue_n):
+     # Collect body-cited figure SHAs from raw.githubusercontent URLs.
+     # For each cited SHA: git ls-tree <sha> -- figures/issue_<N>/ and glob
+     #   *percontext*|*per_context*|*per-context* PNGs.
+     # WARN on any such PNG absent from the body's ![...](...) image URL set.
+     # WARN-level only (an exemption stated in prose is acceptable); never FAIL.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Add/extend tests in `tests/test_verify_task_body.py` pinning the WARN fires on an orphaned per-context PNG and stays silent when the PNG is embedded or no per-context PNG exists at the cited SHAs.
- Grep the workflow surface for related lens text before editing (`grep -rln 'percontext\|per-unit companion' .claude/ scripts/`) and keep `clean-result-critic` Lens 11 wording consistent (the LM lens remains the substantive owner; this is the mechanical WARN backstop).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- WARN-level, never a hard FAIL (per-unit exemptions stated in prose are legitimate; the LM critic adjudicates substance).
- Must not require network: use local `git ls-tree`/`git cat-file` against the cited SHAs; if a cited SHA is not locally reachable, skip silently (no false WARN).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: eb379c0f76e4

Verbatim surfaced prose (clean-result-critic, #928 round 3): "Follow-up (orchestrator should consider): a WARN-level `verify_task_body.py` heuristic — 'a round-committed `figures/issue_<N>/*percontext*` PNG exists at a body-cited figure SHA but is unreferenced by any body image' — would have caught this mechanically; sketch is in the marker's Lens 11 bullet." Marker Lens 11 sketch: "glob committed round figures, diff against body image URLs, WARN on per-unit-named orphans."
