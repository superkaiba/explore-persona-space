---
title: 'workflow-fix: verify_task_body check-31 per-unit pattern set gains pair'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1402fa9ef2e1
created_at: '2026-07-23T00:22:40Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/verify_task_body.py\n\
  bug_observed: check 31's per-unit filename pattern set misses the perpair naming\
  \ used by the issue1415 per-pair small-multiples figures, so the mechanical check\
  \ cannot see an embedded per-unit view under that name\nwhy_workflow_gap: _PER_UNIT_FIG_RE\
  \ omits `pair` from its alternation while per-pair is the project's most common\
  \ per-unit grain; the critic had to catch a Lens-11 break the mechanical check should\
  \ have keyed on\nproposed_change: add perpair (and per-pair) to verify_task_body.py\
  \ check 31's per-unit figure filename pattern set so an embedded per-pair panel\
  \ is recognized\ndiff_sketch: |\n  - per[-_]?(context|unit|cell)\n  + per[-_]?(context|unit|cell|pair)\n\
  \  + check-31 fixture for *_perpair_*.png\nconfidence: high\nrelated_task: #1415\n\
  <!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `mechanizable: yes` prose
follow-up in the clean-result-critic's verdict on task #1415
(epm:clean-result-critique v5, Lens 11 finding; emitting agent:
clean-result-critic; router: orchestrator).

## Goal

Add perpair (and per-pair) to verify_task_body.py check 31's per-unit figure filename pattern set so an embedded per-pair panel is recognized.

## Workflow gap

- **Bug observed:** check 31's per-unit filename pattern set misses the perpair naming used by the issue1415 per-pair small-multiples figures, so the mechanical check cannot see an embedded per-unit view under that name.
- **Why it is a workflow gap:** `_PER_UNIT_FIG_RE = re.compile(r"(?<![a-z0-9])per[-_]?(context|unit|cell)")` (scripts/verify_task_body.py:2445) omits `pair` from the alternation; per-PAIR is this project's most common per-unit grain (the pair is the inference unit on the #1415 line), so both arms of check 31 (orphaned-committed-per-unit WARN and the critic's Lens-11 keying) are blind to `position_profile_perpair_*.png`-style figures. The clean-result-critic caught #1415's Lens-11 break manually; the mechanical check should have.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "perpair\|per_pair\|percell" scripts/verify_task_body.py` → 0 hits for pair-form patterns; `sed -n '2445p'` shows the alternation `(context|unit|cell)` (2026-07-23). Presence claim on the named target confirmed (the regex exists at the cited line; `pair` absent).

## Proposed change (candidate diff sketch — refine in planning)

- _PER_UNIT_FIG_RE = re.compile(r"(?<![a-z0-9])per[-_]?(context|unit|cell)", re.IGNORECASE)
+ _PER_UNIT_FIG_RE = re.compile(r"(?<![a-z0-9])per[-_]?(context|unit|cell|pair)", re.IGNORECASE)
(+ pin test: a `*_perpair_*.png` committed-but-unembedded fixture WARNs; an embedded one PASSes; verify the `indiv` exclusion note stays intact)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Check `tests/test_verify_task_body.py` check-31 fixtures; update the SPEC.md check-31 prose if it enumerates the patterns.

## Constraints / invariants

- Workflow-surface only. Existing check-31 semantics (WARN never FAIL; the #1510 exemption idioms; the `indiv` exclusion) unchanged.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 1402fa9ef2e1

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: check 31's per-unit filename pattern set misses the perpair naming used by the issue1415 per-pair small-multiples figures, so the mechanical check cannot see an embedded per-unit view under that name
why_workflow_gap: _PER_UNIT_FIG_RE omits `pair` from its alternation while per-pair is the project's most common per-unit grain; the critic had to catch a Lens-11 break the mechanical check should have keyed on
proposed_change: add perpair (and per-pair) to verify_task_body.py check 31's per-unit figure filename pattern set so an embedded per-pair panel is recognized
diff_sketch: |
  - per[-_]?(context|unit|cell)
  + per[-_]?(context|unit|cell|pair)
  + check-31 fixture for *_perpair_*.png
confidence: high
related_task: #1415
<!-- /workflow-fix-candidate -->
