---
title: 'workflow-fix: test_gates_full_shape red fleet-wide — clarify_experiment_ask
  gate missing from pinned set'
kind: infra
tags:
- wf-fix
created_at: '2026-08-25T00:41:55Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from the #2342 round-1 implementer (fingerprint
  gates-full-shape-clarify-experiment-ask-drift)'
workflow: v1
---
## Gap

Commit b116c2e872 added the `clarify_experiment_ask` gate to `.claude/workflow.yaml` § gates without updating the pinned expected set in `tests/test_workflow_yaml.py::test_gates_full_shape`, leaving that test red fleet-wide (reproduced rc=1 on the pristine main checkout, 2026-08-24: "Extra items in the left set: 'clarify_experiment_ask'"). Every Step 9c gate run and every local `tests/test_workflow_yaml.py` run now carries this baseline red, eroding the signal of the gates-shape pin.

candidate-fingerprint: gates-full-shape-clarify-experiment-ask-drift
target_file: .claude/workflow.yaml

## Asked change

Add `clarify_experiment_ask` to the test's expected gates set (or, if the gate entry is malformed, correct the workflow.yaml entry) and re-run `tests/test_workflow_yaml.py`; also check whether the gate entry needs the standard fields the shape test asserts.

## Evidence

- Worktree + main-checkout runs both FAIL with "Extra items in the left set: 'clarify_experiment_ask'".
- `git log -S clarify_experiment_ask -- .claude/workflow.yaml` attributes b116c2e872.

## Provenance

Surfaced as a `workflow-fix-candidate v1` by the #2342 round-1 implementer (pre-existing baseline red discovered during its gate-scope verification; not payload-attributed to #2342's diff).
