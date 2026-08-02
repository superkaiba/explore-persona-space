---
title: 'workflow-fix: inline lint gate purges stale pycache before its pytest sweep'
kind: infra
tags:
- wf-fix
- wf-fix-fp:679928dfbef3
created_at: '2026-07-31T22:46:10Z'
has_clean_result: false
origin_prompt: 'boundary-impl, #1345 session 2026-07-31: three nondeterministic gate
  BLOCK/INCONCLUSIVEs traced to stale scripts/__pycache__ pyc served to the gate pytest;
  purging pycache before the gate makes it deterministic.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1345 (emitting agent: boundary-impl).

## Goal

Make the inline lint gate's pytest leg deterministic against stale `scripts/__pycache__/*.pyc`: purge the relevant pycache (or run the gate pytest with PYTHONDONTWRITEBYTECODE=1 + -B semantics) before the mapped-test sweep.

## Workflow gap

- **Bug observed:** three lint-gate runs returned BLOCK/INCONCLUSIVE on failures not reproducible outside the gate; each traced to a stale `scripts/__pycache__/*.pyc` whose recorded source mtime+size matched a newer file (the rapid-Edit + ruff-format-hook same-second rewrite trigger already documented in agent memory), so the gate's plain pytest imported the OLD module while direct runs (which set PYTHONDONTWRITEBYTECODE) recompiled. Purging pycache immediately before the gate made it deterministic.
- **Why it is a workflow gap:** the gate is the mandatory certification path for direct-to-main code; a nondeterministic BLOCK costs a full re-run cycle (~3-8 min each) and erodes trust in real BLOCKs. The emitter's remedy is mechanical and cheap.
- **Confidence (emitter):** high (three occurrences, each resolved by the purge).
- verified-at-filing: `grep -cn "pycache\|DONTWRITEBYTECODE" scripts/inline_lint_gate.py` -> 0 hits (absence-of-guard claim; 0 in-target hits IS the evidence); landed-fix history 7d on the file: 638093ec4f (cert-retry settle pass) — does not touch bytecode staleness (2026-07-31). Related distinct behavior, NOT this filing: the mid-gate payload md5 change INCONCLUSIVE is fail-safe by design and stays.

## Proposed change (candidate diff sketch — refine in planning)

+ In inline_lint_gate.py, before the pytest sweep: remove scripts/__pycache__
+ entries for payload-adjacent modules (or export PYTHONDONTWRITEBYTECODE=1 and
+ pass -B / a clean PYTHONPYCACHEPREFIX to the gate's pytest invocation), so a
+ stale-mtime-matched .pyc can never serve old code to the gate.

## Scope / surfaces

- Primary target: `scripts/inline_lint_gate.py`
- Sibling knowledge: `.claude/agent-memory/experiment-implementer/` stale-pycache memory (4432a82f5c) documents the same trigger for smokes.

## Constraints / invariants

- Workflow-surface only. The gate's verdict semantics (BLOCK/INCONCLUSIVE/PASS) unchanged; only import determinism changes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/inline_lint_gate.py
- fingerprint: 679928dfbef3

Verbatim surfaced prose (boundary-impl, 2026-07-31): "Three of my lint-gate runs came back BLOCK/INCONCLUSIVE on failures I could not reproduce, every time traced to a stale scripts/__pycache__/*.pyc read by the gate's plain pytest (my own runs set PYTHONDONTWRITEBYTECODE and so recompiled). Purging pycache immediately before the gate makes it deterministic."
