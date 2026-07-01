---
title: 'workflow-fix: verify_task_body.py should check per-figure meta.json commit
  against body URL'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f00cf191e100
created_at: '2026-07-01T13:56:26Z'
has_clean_result: false
origin_prompt: 'Round-2 interp-critic on #761 flagged a recurring defect class: per-figure
  meta.json commit drifting from the commit embedded in the body''s raw-GitHub URL
  for the same figure. verify_task_body.py + audit_clean_results_body_discipline.py
  both PASSed a v2 body carrying this drift because neither cross-checks the sidecar
  meta.json against the body URL SHA per-figure. Add the missing mechanical check.'
---
## Goal

Extend `scripts/verify_task_body.py` to cross-check every per-figure sidecar `meta.json` `commit` field against the commit embedded in the body's raw-GitHub URL for that same figure. When the two disagree, FAIL with an actionable pointer.

## Workflow gap

- **Bug observed:** the analyzer's task #761 v2 body linked figures at raw-GitHub URL SHA `38a3b740d7` for three of four figures — correct pin; but a NEW `layer_robustness.png` added in round 2 was linked at `c969f2281f` in the body while its sidecar `layer_robustness.meta.json` retained the stale `commit: 3f5d228398` (the JSON+script commit, one before the figure landed). The existing `verify_task_body.py` figure check + the discipline audit BOTH PASSed the v2 body because they cross-check the body URL against the `## Reproducibility` commit line, not against each figure's per-file `*.meta.json`. The mismatch was caught only by the Codex twin's Lens 6 by loading the meta.json directly.
- **Why it is a workflow gap:** this is a recurring defect class — the round-1 Codex critique already flagged the identical shape (all three original figure meta.json files carried a stale commit SHA vs the body URLs), the analyzer fixed those three, and then re-introduced the exact same class of drift on the newly-added `layer_robustness.png`. A recurring mechanizable defect that the LM-side critic catches per-round belongs in the mechanical gate.
- **Confidence (emitter):** medium — the class is well-defined + the check is straightforward, but the fix must be robust to (a) figures referenced by the body without a `.meta.json` sidecar (skip cleanly), (b) sidecars that carry a `commit_sha` alias vs `commit`, (c) short-SHA vs full-SHA equality (prefix match).

## Proposed change (candidate diff sketch — refine in planning)

```python
# In scripts/verify_task_body.py, add a new check (Check 24 or after existing figure checks):
#
# For every `figures/issue_<N>/<fig>.png` referenced by a raw-GitHub URL in the body:
#   1. Parse the body URL's SHA (the segment between /blob/ or /raw/ and /figures/).
#   2. Read `figures/issue_<N>/<fig>.meta.json` if it exists.
#   3. Compare its `commit` (or `commit_sha`) field against the body URL SHA (prefix-safe).
#   4. FAIL when they disagree; PASS when they match; NO-OP when the sidecar doesn't exist
#      (not every figure needs a sidecar).
#
# Register the check in the discipline audit too so it fires in both mechanical gates.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (add the new check).
- Secondary target: `scripts/audit_clean_results_body_discipline.py` (mirror the check so the discipline audit fires it too, or make one call the other).
- Test: add `tests/test_verify_task_body.py::test_figure_meta_commit_matches_body_url` covering PASS / FAIL / no-sidecar-skip cases.

## Constraints / invariants

- Workflow-surface only — no experiment code touched.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Existing `verify_task_body.py` checks + discipline audit invariants unaffected.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: f00cf191e100

Surfaced by the round-2 `interpretation-critic` on task #761 as a prose follow-up (verbatim): "Consider adding a `verify_task_body.py` check that, for every `figures/issue_<N>/*.meta.json`, asserts its `commit` field equals the commit embedded in the body's `raw.githubusercontent.com/.../<sha>/figures/issue_<N>/<same-figure>.png` URL. Concrete, likely to recur across any figure-bearing clean-result. `mechanizable: yes`."

Related concrete recurrence on #761 round 2:
- meta.json says `commit: 3f5d228398`
- body URL says `c969f2281f...`
- Only `c969f2281f...` actually contains the figure blob
- Both existing mechanical gates PASSed regardless
