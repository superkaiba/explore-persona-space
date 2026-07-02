---
title: 'Fix: restore issue734_figures functions dropped by 4109e30f17 (pinned test
  failing on main)'
kind: infra
tags: []
created_at: '2026-07-02T09:34:05Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #851 implementer concern 734-figures-regression: commit
  4109e30f17 overwrote scripts/issue734_figures.py, dropping _load_corrected_reread/hero1_install_recovery
  added by e022921732; tests/test_issue734_corrected_reader.py fails on pristine main.'
---
## Overview / Motivation

Auto-filed from task #851 (implementer concern `734-figures-regression`, 2026-07-02). The full pytest suite on pristine `main` has exactly one failing test, root-caused during #851's tree-wide verification.

## Goal

Restore the `_load_corrected_reread` / `hero1_install_recovery` functionality in `scripts/issue734_figures.py` so `tests/test_issue734_corrected_reader.py` passes on `main` again.

## Root cause (already diagnosed — verify, then fix)

- Commit `e022921732` added `_load_corrected_reread` / `hero1_install_recovery` to `scripts/issue734_figures.py`, pinned by `tests/test_issue734_corrected_reader.py`.
- Commit `4109e30f17` later OVERWROTE `scripts/issue734_figures.py`, dropping those functions while the pinning test remained.
- Result: `uv run pytest tests/test_issue734_corrected_reader.py` fails identically on pristine `main` (verified 2026-07-02 during #851's stash-compare pass; full suite 3401 passed / 1 failed).

## Proposed fix

Reconcile the two versions of `scripts/issue734_figures.py`: re-introduce the `e022921732` functions (git history has the exact bodies — `git show e022921732:scripts/issue734_figures.py`) into the current file WITHOUT dropping whatever `4109e30f17` legitimately changed. If the two versions are divergent forks of the same figure script, merge them; the pinned test defines the required surface.

## Scope / constraints

- Experiment-adjacent figures code (`scripts/issue734_figures.py`) + no test edits expected (`tests/test_issue734_corrected_reader.py` is the contract; fix the script, not the test — unless investigation shows the test pins behavior #734 deliberately retired, in which case say so in the plan and propose the test change explicitly).
- Acceptance: `uv run pytest tests/test_issue734_corrected_reader.py` green on the branch; no other test newly failing; ruff clean on the touched file.
- 0 GPU-hours; VM-local.

## Provenance

- Surfaced by: task #851 implementer round 1 (concern id `734-figures-regression`), tree-wide stash-compare verification.
- Pre-existing on `main` — NOT introduced by #851's diff.
