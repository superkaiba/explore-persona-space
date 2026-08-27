---
title: 'main-red: test_workflow_lint_prod_import_lockfile::test_live_tree_clean pins
  7 live-tree WARNs, main emits 11 (dangling first-party import roots unreconciled)'
kind: infra
tags:
- main-red
- wf-fix
created_at: '2026-08-27T09:39:38Z'
has_clean_result: false
origin_prompt: 'Auto-filed by the #2620 /issue session: Step 9c compare stripped this
  as a pre-existing main-red on a workflow-invariant test and instructed emit-or-verify
  a routable urgency: main-red workflow-fix-candidate (#1713/#1742).'
workflow: v1
---
# `tests/test_workflow_lint_prod_import_lockfile.py::test_live_tree_clean` is red on main: the live-tree WARN pin expects 7 rows, the tree now emits 11

urgency: main-red

## The defect

`test_live_tree_clean` pins the EXPECTED WARN roster of
`workflow_lint.py::check_prod_import_lockfile` on the live tree (assert at
`tests/test_workflow_lint_prod_import_lockfile.py:545`). On current main the check emits 11
WARN rows against the pinned 7 (`assert 11 == 7`), so the test fails on a pristine main
checkout — reproduced 2026-08-27 at the repo root (97.9 s, single node) AND classified by
`step9c_baseline.py compare` as a stripped pre-existing main-red on a workflow-invariant
test during #2620's Step 9c gate (compare rc=0, `new: []`).

The 11 realized WARN rows include dangling first-party import roots the pin does not
carry — observed in the gate log: `issue2474_fit` (9 sites), `issue500_predictors` (2),
`issue541_personas`, `issue541_predictors`, `issue541_upload_lib`, `issue621_analyze`,
`issue_521_prep_turner_corpus`, `_issue506_common` (4) — plus the two known extra/group
rows (`liger_kernel` via 'gpu', `umap` via 'viz'). New dangling roots evidently landed on
main after the pin was last reconciled.

## The fix

Reconcile the pin with the live tree the right way per the test's own design: for each
NEW dangling first-party root, either (a) the importing script is stale/needs its import
fixed or its module landed under a different name — fix the import; or (b) the dangling
state is legitimate latent breakage worth WARN-tracking — fold the row into the pinned
expected set. Do NOT blanket-bump 7 → 11 without per-root disposition. Every Step 9c gate
fleet-wide currently eats a ~98 s pre-existing red + a compare pristine-oracle run on this
file until this lands.

## Acceptance

- `uv run pytest tests/test_workflow_lint_prod_import_lockfile.py::test_live_tree_clean -q`
  passes at the repo root on main.
- Each newly-pinned WARN row carries a one-line per-root disposition in the commit message
  (import fixed vs legitimately pinned).

## Provenance

Auto-filed by the #2620 /issue session at its Step 9c compare (URGENT-PARK-REQUIRED:
stripped pre-existing main-red on a workflow-invariant test; emit-or-verify routable
'urgency: main-red' workflow-fix-candidate, #1713/#1742). Compare artifacts:
/tmp/step9c-compare-issue-2620.{json,err}; gate junit /tmp/step9c-junit-issue-2620.xml.
