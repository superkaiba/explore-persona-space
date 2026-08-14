---
title: 'daily-fix: test-order polluter test_issue1482_densesae'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-08-06T07:03:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): import side effect makes
  test_backend_poll module-mode node fail in multi-file runs (8-round bisect, minimal
  pair reproduces)'
workflow: v1
---
# daily-fix: test-order pollution — test_issue1482_densesae_fullwidth.py makes a test_backend_poll node fail in multi-file runs

## Workflow gap

`tests/test_backend_poll.py::test_ensure_scripts_dir_bootstrap_resolves_runpod_api_in_module_mode`
fails in large gate runs but passes alone. The #2105 session bisected a 170-file pool in 8
rounds down to a single polluter: `tests/test_issue1482_densesae_fullwidth.py`
(byte-identical to origin/main), and confirmed the minimal two-file pair reproduces
("1 failed, 12 passed"). Collection-import pollution on main — every gate that happens to
collect both files re-classifies the failure.

verified-at-filing: the bisect + minimal-pair reproduction are the miner's probed
readbacks of session 2ba4ca62 rows 463–469 (2026-08-06T06:30–06:31Z, the session's own
pytest outputs). The #2105 session was still LIVE mid-investigation at mining time
(status `reviewing`) and had posted no workflow-fix filing for it as of
2026-08-06T07:0xZ (`task.py view 2105 --json` events scan — no `epm:workflow-fix-*`
rows); if it files its own between now and dispatch, the wf-fix fingerprint dedup should
catch the collision — planner should re-check #2105's events before implementing.

unverified hypothesis — verify at plan time: the pollution mechanism is a module-level
import side effect in test_issue1482_densesae_fullwidth.py (sys.path / sys.modules
mutation at import) that perturbs `runpod_api` module-mode resolution — miner-inferred,
not probed.

## Proposed change

Reproduce with the minimal pair, identify the import-time side effect, and make it
hermetic (move the mutation into a fixture with teardown, or import lazily inside tests).
Deliverable: the pair runs green in both orders and the full-suite node is stable.

## Provenance

- workflow_fix_target: tests/test_issue1482_densesae_fullwidth.py
- origin: /daily 2026-08-05 problem sweep — miner 7 P17.
