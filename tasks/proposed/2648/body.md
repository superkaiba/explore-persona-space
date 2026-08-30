---
title: Clear the torch-before-dotenv red on main that reds every Step 9c gate
kind: infra
tags: []
created_at: '2026-08-30T17:50:40Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2645''s implementer: test_no_new_torch_before_dotenv_vm_entrypoints
  fails on origin/main independent of any round diff, so every session reaching Step
  9c must re-derive it as pre-existing.'
workflow: v1
---
---
kind: infra
---

# Clear the torch-before-dotenv red on main that reds every Step 9c gate

## Goal

Make `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` green on `main`, so it stops appearing as an unexplained failure in every session's Step 9c gate run.

## The finding

Surfaced during #2645's implementation round, on a file set that has nothing to do with it. The test fails on `origin/main` itself, independent of any round diff: it failed identically at the main checkout, and the two offending scripts were verified byte-identical to `origin/main` and untouched by that round.

Reported offenders, to be re-verified before any edit rather than trusted from this body:

- `scripts/issue2617_standardized_ctx_answer.py`
- `scripts/issue779_ctxansviz_pc_specimens.py`

## Why it matters, and why it is not merely cosmetic

The invariant being violated is real, not a lint preference. Torch freezes its thread pool from `OMP_NUM_THREADS` at IMPORT time, so a VM entrypoint that imports torch BEFORE calling `load_dotenv()` gets an uncapped pool. On this shared VM that is roughly 32 runnable threads per job, and with several concurrent jobs the box has been driven to load 186-226 while each job realized only 5-6 cores. The test exists to keep new VM entrypoints from reintroducing that, which is why the fix should be to satisfy the invariant rather than to widen the test's allowlist.

The second cost is the one that makes this worth a task: an unexplained red in a gate teaches sessions to skim past gate output. Every session reaching Step 9c currently sees a failure it must independently re-derive as pre-existing before proceeding. #2645's implementer paid exactly that cost, and each subsequent session pays it again.

It is not in the known-red ledger, and the reason is circular: the ledger had gone 78 hours stale precisely because the nightly refresh #2645 repaired had not run for 18 nights. Once that refresh runs again, tonight's pass may absorb these two into the ledger, which would suppress the noise WITHOUT fixing the underlying invariant violation. Check the ledger's current state before starting — if the entries have been absorbed, that changes this task's urgency but not its correctness, since a ledgered red is a silenced red, not a fixed one.

## Approach

Per-file, since the two scripts may differ in shape: move the `load_dotenv()` call ahead of the torch import, or restructure so the entrypoint's torch import happens inside a function called after the dotenv load. The canonical requirement is in `.claude/rules/code-style.md` under the shared-VM CPU thread caps entry.

Prefer fixing the scripts over widening the test. If a script genuinely cannot be restructured, say why explicitly in the body rather than reaching for an allowlist entry.

## Acceptance

1. `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` passes on `main`.
2. Each offender either satisfies the invariant, or carries a recorded reason it cannot.
3. Both scripts still run — confirm the entrypoints import and their argparse surface is intact, since the fix reorders module-level statements.
4. No new lint error naming either file.

## Provenance

Surfaced by #2645's implementer while running the Step 9c-selected test union for an unrelated file set (a cron wrapper file-mode fix), and deliberately not fixed there: repairing an unrelated pre-existing main red inside a mode-fix round would have widened that round's scope well past its own acceptance criteria. Filed with `--no-dispatch`: the account was session-limited at filing time, so an auto-spawned session would have died on the limit immediately. Dispatch when convenient via `spawn_session.py spawn-issue --issue <N> --auto`.
