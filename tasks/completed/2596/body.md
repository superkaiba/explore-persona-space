---
title: 'test_no_new_torch_before_dotenv_vm_entrypoints red on main: 4 scripts missing
  load_dotenv, reds Step 9c gates fleet-wide'
kind: infra
tags: []
created_at: '2026-08-26T04:02:52Z'
has_clean_result: false
origin_prompt: Surfaced during /issue 2546 round-12 lint attribution; orchestrator
  independently reproduced rc=1 on clean main naming scripts/issue1901_mlpdense_fold_analysis.py,
  issue1901_mlpdense_fold_figures.py, issue2254_firstk_ctxext_sensitivity.py, issue2378_lenmatch_fig.py.
workflow: v1
---
# test_no_new_torch_before_dotenv_vm_entrypoints fails on clean main — reds any Step 9c gate that selects it

## Goal

Restore `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` to green on `main` by adding the missing `load_dotenv` calls to the four violating entrypoints, so the Step 9c test gate stops reporting a pre-existing fleet red as if it were the task's own payload failure.

## Evidence — reproduced on clean main by the orchestrator, not inherited from a report

```
$ git rev-parse --abbrev-ref HEAD
main
$ uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q
rc=1
FAILED tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
```

Violators named by the failure, four scripts, all missing a `load_dotenv` before their torch import:

- `scripts/issue1901_mlpdense_fold_analysis.py`
- `scripts/issue1901_mlpdense_fold_figures.py`
- `scripts/issue2254_firstk_ctxext_sensitivity.py`
- `scripts/issue2378_lenmatch_fig.py`

The test is on `main` and so are the violators, so this is not a feature-branch artifact.

## Why it matters beyond one task

The Step 9c test-verdict gate selects tests by changed-file mapping, so any task whose payload maps onto `test_shared_vm_thread_caps.py` inherits a red that has nothing to do with its own diff. That forces every affected session to spend a review round establishing "pre-existing, payload-external" attribution — which is exactly the verification the gate exists to make unnecessary. Task #2546 paid that cost across four consecutive review rounds (rounds 9-12), each one re-deriving the same attribution: the round-9 breadcrumb, round 10's re-fenced 1500 s lint run, round 11's branch-vs-main skew check, and round 12's independent re-confirmation.

Two failure modes it creates, both worse than the red itself:

1. **Attribution work repeated per session** — every affected task re-establishes the same fact, and a reviewer who skips that step either blocks a clean payload or waves through a genuine failure.
2. **Desensitization** — a standing red in a gate trains sessions to treat that gate's failures as background noise, which is how a real regression in the same file would ship. This is the same dynamic the `.claude/rules/repo-root-uncommitted-state.md` § watcher-pass note warns about for recurring alerts.

## Scope

- Add the missing `load_dotenv` (`orchestrate.env.load_dotenv`, per the lint rule `--check-dotenv-before-hf-import` and CLAUDE.md § Upload Policy: "New direct-upload scripts use `orchestrate.env.load_dotenv`, never bare `dotenv`") to each of the four scripts, placed before the torch/HF import the test checks.
- Re-run the test to green on `main`.
- Confirm the four scripts still run (or at minimum still import) after the insertion — a `load_dotenv` at the wrong position can shadow an intentional env override.

Do NOT weaken, skip, or xfail the test — it guards a real property (env must load before torch so thread caps and HF cache redirects apply on the shared VM). The fix is the four missing calls.

## Provenance

Surfaced by the `/issue 2546` round-12 implementer while attributing its own lint/pin-sweep results, and then independently reproduced on clean `main` by the orchestrator before filing (rc=1, same four scripts). Related standing rule: `.claude/rules/gotchas.md` § dotenv-before-torch; lint twin `workflow_lint.py --check-dotenv-before-hf-import`.
