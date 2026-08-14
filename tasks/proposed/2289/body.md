---
title: '#2223''s merged r2/r3/r5 scripts red 5 fleet-wide gate nodes on origin/main
  — incl. the no-flags workflow_lint bundle'
kind: infra
tags: []
created_at: '2026-08-14T12:20:38Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'surfaced by #2288''s code-reviewer (thread-caps) + #2288''s Step 9c
  gate run (live-hf-retry-routing); both verified pre-existing on origin/main, not
  attributable to #2288'
workflow: v1
---
---
kind: infra
---

# #2223's merged r2/r3/r5 scripts red 5 fleet-wide gate nodes on origin/main — incl. the no-flags workflow_lint bundle

## Goal

Restore `origin/main` to a state where a freshly-cut worktree passes the Step 9c
workflow-invariant gate. Right now `origin/main` (tip `6d29131458`) carries four
`scripts/issue2223_*.py` files whose content reds **5 gate nodes for every
session**, one of which is the no-flags `workflow_lint.py` bundle — the very
instrument the Step 9c gate, the Step 10d pre-push gate, and the inline-payload
lint gate all run. This is the #1388 class: lint-red code landed on main breaks
the fleet's gate instrument, and every session must then hand-adjudicate 5 reds
through `step9c_baseline.py compare` before it can read its own verdict.

## The two defects

**Defect 1 — bare HF Hub call in LIVE code (the fleet-wedge).** Reproduced in
isolation on a main-synced tree:

```
$ uv run python scripts/workflow_lint.py --check-live-hf-retry-routing
workflow_lint: [live-hf-retry-routing] scripts/issue2223_r5_pubtopic.py:218: bare HF Hub call in LIVE code — route through hub.retry_transient, waive with `# NO_RETRY: <reason>`, or (pre-existing file this round never touched — snapshot staleness, #1568) regen on a main-synced tree: `workflow_lint.py --regen-hf-routing-snapshot`: hf_hub_download(
workflow_lint: FAIL (1 error(s))
rc=1
```

Reds 4 nodes in `tests/test_workflow_lint.py`:
`test_workflow_lint_default_exits_zero` (the no-flags bundle),
`test_check_live_hf_retry_routing_live_tree_passes`,
`test_workflow_lint_check_live_hf_retry_routing_cli_exits_zero`,
`test_regen_hf_routing_snapshot_live_tree_subset_of_current`.

The correct fix is almost certainly **(a) route the call through
`hub.retry_transient`** (or waive it inline with `# NO_RETRY: <reason>` if the
call site genuinely should not retry) — NOT (b) `--regen-hf-routing-snapshot`.
The snapshot-regen escape in the message exists for a PRE-EXISTING file whose
snapshot went stale (#1568); `issue2223_r5_pubtopic.py` is a file NEW on main
making a genuinely unrouted `hf_hub_download(` call, so a regen would bless the
bare call and silently retire the check's coverage of it. Confirm which case
applies before choosing — if the call is genuinely new, fix the call site.

**Defect 2 — module-top torch import before `load_dotenv()`.** Reds
`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`,
flagging all four scripts:
`scripts/issue2223_analyzer_figs.py`, `scripts/issue2223_r2_analysis.py`,
`scripts/issue2223_r3_domain_ci.py`, `scripts/issue2223_r5_pubtopic.py`.

Torch freezes its thread pool from `OMP_NUM_THREADS` at IMPORT, so a VM entrypoint
that imports torch before `orchestrate.env.load_dotenv()` runs uncapped — ~32
runnable threads per job on a box shared across ~15 concurrent sessions
(`.claude/rules/code-style.md` § Shared-VM CPU thread caps, #847). The fix is a
`load_dotenv()`-before-heavy-import reorder in each of the four scripts.

## Provenance

Surfaced by the #2288 `code-reviewer` (defect 2) and by #2288's own Step 9c gate
run (defect 1, which the reviewer's narrower run did not reach). Verified NOT
attributable to #2288: a byte-parity probe confirms `issue-2288` never touched
any `scripts/issue2223_*.py`, and `git cat-file -e origin/main:...` confirms the
files are present on `origin/main` — a pristine tree cut from `origin/main`
fails all 5 nodes. #2288's own gate run measured
`5 failed, 5847 passed, 12 skipped in 2722.64s`, and all 5 failures are these.

## Acceptance

1. `uv run python scripts/workflow_lint.py` (no flags) exits 0 on a tree cut
   from `origin/main`.
2. `uv run pytest tests/test_workflow_lint.py tests/test_shared_vm_thread_caps.py`
   passes on that tree.
3. Whichever remedy is chosen for defect 1, the choice is justified in the
   round's marker: routing/waiving the call site vs regenerating the snapshot are
   NOT interchangeable (a regen retires the check's coverage of that call).
4. Defect 2's reorder does not change any script's behavior — the scripts are
   already-run analysis drivers for a promoted #2223 result; the fix is import
   ordering only, not logic.
5. State whether any OTHER `scripts/issue*.py` on main carries the same two
   shapes (one `--check-live-hf-retry-routing` run + one thread-caps run answer
   this), so the fix is not a whack-a-mole round.

## Why this is urgent rather than debt

The no-flags lint bundle is a gate instrument, not just a test: while it is red
on main, every session's Step 9c verdict requires a `compare --run-pristine`
adjudication to separate its own diff's reds from these 5, and the Step 10d
pre-push lint gate cannot return a clean verdict for anyone. #1388 is the
precedent — two inline-landed lint-red scripts broke the Step 9c gate fleet-wide.
