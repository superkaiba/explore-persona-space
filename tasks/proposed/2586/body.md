---
title: Clear pre-existing red test_no_new_torch_before_dotenv_vm_entrypoints on main
  (4 non-grandfathered violators red the Step 9c gate for every scripts/*.py diff)
kind: infra
tags: []
created_at: '2026-08-25T19:38:42Z'
has_clean_result: false
origin_prompt: 'Found by #2584''s implementer during gate-scope verification: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  fails on pristine origin/main with 4 non-grandfathered violators (3 landed 08-24/08-25
  after the grandfather list''s last touch 2026-08-09).'
workflow: v1
---
## Goal

Clear the pre-existing red `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` on pristine `origin/main`, so the Step 9c gate's baseline-compare stops carrying a standing red that every round touching `scripts/*.py` inherits (the test is glob-scan-selected for any such diff).

## Measurement

On a clean `main` checkout: `uv run pytest -q tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` FAILS naming 4 non-grandfathered violators (VM entrypoints importing heavy modules — torch/numpy — before `load_dotenv()`):

| violator | landed |
|---|---|
| `scripts/issue1901_mlpdense_fold_analysis.py` | pre-dates the grandfather list's last touch |
| `scripts/issue1901_mlpdense_fold_figures.py` | `fd8222d6f7`, 2026-08-24 |
| `scripts/issue2254_firstk_ctxext_sensitivity.py` | `0ebe4e84ac`, 2026-08-25 |
| `scripts/issue2378_lenmatch_fig.py` | `fc8e426aca`, 2026-08-25 |

All 4 violate at the base SHA; the grandfather list was last touched `8471503fc1` (2026-08-09), before the newest 3 landed. Full git evidence: the `epm:results` v1 marker on task #2584 (section (c), "Pre-existing failure root-cause").

## Remedy

Per file, either add `load_dotenv()` (from `explore_persona_space.orchestrate.env`) before the heavy imports — the #847 pattern and the invariant's intent — or grandfather the file in the test's list with a stated reason. Prefer the real fix; grandfathering a 2026-08-25 file defeats the invariant. State the choice per file.

## Sequencing constraint

Task #2584 (in flight as of filing, expected to merge within the hour) edits `scripts/issue1901_mlpdense_fold_analysis.py` on branch `issue-2584`. Cut the worktree AFTER #2584 lands on main (or sync that file from `origin/main` before editing it) to avoid a same-file collision.

## Acceptance

- `uv run pytest -q tests/test_shared_vm_thread_caps.py` green on a main-synced tree (no NEW failures vs the plan-time baseline elsewhere in the file).
- No heavy-import regression: each fixed script still runs its own happy path (import-smoke suffices for figure scripts).
- The remedy is stated per file (fix vs grandfather, with reason).

## Provenance

Found by task #2584's implementer during the #1288 gate-scope verification (round 1): the diff-linked `tests/test_shared_vm_thread_caps.py` failed locally, and the root-cause probe showed the failure is pre-existing on pristine `origin/main` — reported per #2584's plan kill criterion (out-of-scope offenders are reported, never absorbed).
