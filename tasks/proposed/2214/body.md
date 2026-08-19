---
title: 'daily-fix: test_issue1315_dispatch pollutes the fu3 context registry — order-dependent
  Step 9c gate bounce'
kind: infra
tags:
- from-2063-sweep
created_at: '2026-08-10T04:16:55Z'
has_clean_result: false
parent_id: 2063
origin_prompt: 'Flagged by #2063 implementer round: tests/test_issue1315_dispatch.py:328-332
  registers fu3 contexts without restore; test_issue1090_fu3_dispatcher fails under
  randomized ordering.'
workflow: v1
---
## Goal

Fix the pre-existing order-dependent test pollution that can stochastically
bounce any Step 9c test-verdict gate whose selection contains both files:
`tests/test_issue1315_dispatch.py:328-332` calls `register_fu3_contexts()`
without restoring the registry, and
`tests/test_issue1090_fu3_dispatcher.py::test_conv_context_is_wildchat_family`
then fails its registry-cleanliness assert at entry under randomized ordering.

## Context

Found during #2063's implementer round (constants-only diff; all involved
files byte-identical to origin/main, so the bug is pre-existing on main).
Deterministic reproduction:

```
uv run pytest -q -p no:randomly tests/test_issue1315_dispatch.py tests/test_issue1090_fu3_dispatcher.py
```

The polluter registers fu3 contexts into a module-global registry and never
removes them; the victim asserts the registry is clean at entry. Under
pytest-randomly, any ordering that runs the polluter first fails the victim —
a fleet-wide stochastic Step 9c gate bounce for unrelated diffs (the gate
selects both files whenever a diff touches the issue1090/1315 family or the
pin-sweep pulls them in).

## Fix sketch

Add registry restoration to the polluter (a `try/finally` popping the
registered fu3 context keys, or a fixture with teardown — match the registry
API in `scripts/issue1090_fu3_worker.py` / the module that owns
`register_fu3_contexts`). Keep the victim's cleanliness assert (it is the
detector, not the bug). Verify with the deterministic repro above run in BOTH
orders plus a `-p randomly` seed sweep.
