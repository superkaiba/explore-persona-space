---
title: 'workflow-fix: ruff-policy pin red on main — autonomous_session_watch.py gate_push_pass
  C901 (c0d76f99c9)'
kind: infra
tags:
- wf-fix
- main-red
created_at: '2026-08-17T12:12:11Z'
has_clean_result: false
origin_prompt: 'surfaced by #2155 MF-A re-execution unit 2026-08-17 (pre-existing
  main-red, 0 introduced by #2155)'
workflow: v1
---
# workflow-fix: ruff-policy pin red on main — autonomous_session_watch.py gate_push_pass C901 (complexity 18 > 15), introduced by c0d76f99c9

## Provenance

workflow_fix_target: scripts/autonomous_session_watch.py
urgency: main-red
failing_test: ruff-policy pin (C901 arm) on scripts/autonomous_session_watch.py:35796
wf_fix: true
Surfaced by task #2155's MF-A re-execution unit (2026-08-17): the ruff-policy pin FAILs at current origin/main — `autonomous_session_watch.py:35796 C901 gate_push_pass is too complex (18 > 15)` — introduced by main commit `c0d76f99c9` ("mission-control rung 0: flag-gated async gate mode (EPS side)") and unfixed at the tip checked (`ea193dc743`). #2155's copy is byte-identical to base (0 introduced); every session running the ruff-policy gate on a branch containing this file inherits the red.

## Goal

Restore the ruff-policy pin to green on main: reduce `gate_push_pass` complexity below the C901 threshold (extract helpers along its natural branches), or — if the owning mission-control line justifies it — add the sanctioned per-function suppression per the repo's existing C901 conventions (check how sibling over-threshold functions in this file are handled). No behavior change; the async-gate flag logic stays intact.

## Acceptance criteria

1. `ruff check` C901 arm green on scripts/autonomous_session_watch.py at main.
2. Behavior-preserving: the watcher's gate-push pass tests (tests/test_autonomous_session_watch.py) stay green.
3. Coordinate with the mission-control rung-0 line (the c0d76f99c9 author session) if it is still live — one implementer per file set.

Estimated GPU-hours (total): 0
