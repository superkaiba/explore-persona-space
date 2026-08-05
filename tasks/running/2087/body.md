---
title: 'workflow-fix: AST lint for discarded upload-helper returns (scripts/)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f4319be4f5f9
created_at: '2026-08-05T17:11:30Z'
has_clean_result: false
origin_prompt: 'epm:code-review v6 on #2054, Major + Mechanizable-yes: AST lint flagging
  Expr-statement calls to _upload_folder_filtered/_upload under scripts/ (waiver for
  deliberate fail-soft callers); 6 pre-existing offenders enumerated (phase_b:255,
  phase_c:445, phase_d:489, fits:949, ladder:982, capture:943)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #2054 (emitting agent: code-reviewer, review round 10, marker
`epm:code-review` v6).

## Goal

Add an AST lint check (bundled into the no-flags default run) that flags
Expr-statement (discarded-return) calls to `_upload_folder_filtered` /
`_upload` under `scripts/`, with a waiver comment for deliberate fail-soft
callers.

## Workflow gap

- **Bug observed:** six discard-shaped upload-helper call sites reached main
  across issue2054 phase scripts despite full review rounds; a
  fail-soft-by-return upload helper whose return is discarded logs success and
  exits 0 on silent durability loss.
- **Why it is a workflow gap:** `src/explore_persona_space/orchestrate/hub.py`'s
  `_upload_folder_filtered` is fail-SOFT by return on every failure shape
  (missing token → warn + `""`; incomplete expected-set verify →
  `logger.error` + `""`; terminal `except Exception` → `""`). Any caller that
  discards the return converts a durability failure into a false-success exit
  0 — the exact class `.claude/rules/upload-policy.md` bans ("'upload returned
  no path' is a TRACKED GAP ... never a warning-and-continue"). Reviews caught
  the class only by hand (task #2054 review rounds 9-10, which fixed 3 sites
  and enumerated 6 more pre-existing on main); no mechanical check exists. The
  adjacent `check_no_upload_or_true` (#841) covers the SHELL-side suppression
  (`|| true` on upload lines) but not this Python-side discarded-return shape.
- **Confidence (emitter):** high (reviewer marked the finding `Mechanizable:
  yes` with the check sketch; 6 live offenders enumerated by exact grep +
  git-provenance-probed as pre-existing on main).
- verified-at-filing: `grep -rnE '^\s*_upload_folder_filtered\(|^\s*_upload\(' scripts/`
  → 12+ bare-statement call-shape hits in 11 files at the main checkout
  (2026-08-05; issue2054_capture.py:943, issue2054_phase_a.py:1067/:1089,
  issue2054_phase_c.py:445, plus issue2054_phase_b/phase_d/fits/ladder sites
  enumerated in `epm:code-review` v6 on #2054, plus issue1481_marker.py /
  issue825_gen_conversations.py sites whose local `_upload` helpers need
  per-site fail-soft classification — the grep is a line-anchored
  approximation; the AST check is the true classifier). Absence probe:
  `grep -inE 'upload.*(discard|return)|discard.*upload' scripts/workflow_lint.py`
  → 1 incidental hit at :4309, the docstring of the SHELL-side
  `check_no_upload_or_true` (#841) — no Python-discarded-return check exists.
  Landed-fix history: `git log --oneline --since='7 days ago' -- scripts/workflow_lint.py`
  → 5 commits, none touching this class.

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_upload_return_discard(paths: list[Path]) -> list[str]:
+     # AST pass over scripts/**.py: flag ast.Expr nodes whose value is a
+     # Call to _upload_folder_filtered / _upload (name or attribute form) —
+     # the return carries the success/failure verdict and MUST be consumed
+     # (capture-and-raise per hub.py upload_raw_completions_to_data_repo).
+     # Waiver: a trailing `# lint: upload-fail-soft-ok — <reason>` comment
+     # on the call line exempts deliberate fail-soft callers.
+     # Grandfather: seed an allowlist from today's offenders (or fix them
+     # in the same round) so the no-flags run stays green on main.
  # bundle into the no-flags default run + add pin tests in
  # tests/test_workflow_lint.py (live-tree pass + synthetic offender FAIL)
```

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Secondary: `tests/test_workflow_lint.py` (pin tests: synthetic offender
  FAILs, waivered caller passes, live tree green).
- Grep the workflow surface for the pattern before editing
  (`grep -rnE '^\s*_upload_folder_filtered\(|^\s*_upload\(' scripts/`) and
  decide per site: fix (capture-and-raise), waiver, or grandfather-allowlist;
  list them in the plan. NOTE: the 6 issue2054 sites may already be fixed by
  task #2054's in-flight r11 round by the time this session runs — re-grep at
  plan time; the lint must be green on the tree it lands on (the #1145/#931
  class), so fix-or-waive every residual offender in the same round.

## Constraints / invariants

- Workflow-surface only — the lint + its tests; residual offender fixes in
  `scripts/issue*_*.py` are allowed ONLY as the minimal capture-and-raise
  one-liners needed to land the check green (the #2054 r10 pattern,
  hub.py:2161), never behavioral restructures.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; the no-flags default run stays green on the landing tree.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: f4319be4f5f9

Origin candidate (synthesized from `epm:code-review` v6 on task #2054, Major
finding + "Mechanizable: yes" line): "the v5-sketched AST lint (flag
Expr-statement calls to `_upload_folder_filtered` / `_upload` under
`scripts/`, waiver comment for deliberate fail-soft callers) would have caught
all 6; still recommended as a workflow-fix candidate for whoever owns the next
lint pass."
