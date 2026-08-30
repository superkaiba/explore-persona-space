---
title: 'main-red: HF-routing/hub-retry lint + torch-before-dotenv scan fail on pristine
  origin/main, redding Step 9c fleet-wide'
kind: infra
tags:
- main-red
created_at: '2026-08-30T20:18:25Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2646 Step 9c step-1d compare: 7x URGENT-PARK-REQUIRED
  demanding a routable ''urgency: main-red'' workflow-fix-candidate (#1713/#1742).
  Reproduced on a pristine origin/main worktree.'
workflow: v1
---
## Goal

Restore `origin/main` to green on the three `workflow_lint` HF-routing/hub-retry
checks and on `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`,
so the `/issue` Step 9c test-verdict gate stops failing fleet-wide on
main-inherited red.

This is the routable `urgency: main-red` workflow-fix-candidate the Step 9c
step-1d compare demanded seven times during #2646's gate run
(`URGENT-PARK-REQUIRED: … stripped pre-existing main-red on a
workflow-invariant test; emit (or verify existing) a routable
'urgency: main-red' workflow-fix-candidate (#1713/#1742)`).

## Evidence (measured 2026-08-30, not inferred)

Step 9c gate on #2646 (`issue-2646`, base `2fca4437de8`): **7 failed, 8219
passed, 12 skipped** in 1:08:58. All 7 are main-inherited; #2646's payload is
5 files and touches none of the offenders.

```
FAILED tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
FAILED tests/test_workflow_lint.py::test_workflow_lint_default_exits_zero
FAILED tests/test_workflow_lint.py::test_check_hub_verify_retry_repo_tree_is_clean
FAILED tests/test_workflow_lint.py::test_workflow_lint_check_hub_verify_retry_cli_exits_zero
FAILED tests/test_workflow_lint.py::test_check_live_hf_retry_routing_live_tree_passes
FAILED tests/test_workflow_lint.py::test_workflow_lint_check_live_hf_retry_routing_cli_exits_zero
FAILED tests/test_workflow_lint.py::test_regen_hf_routing_snapshot_live_tree_subset_of_current
```

**Authoritative main-red proof.** A detached worktree cut at `origin/main`
reproduces the thread-caps failure independently of any branch:
`git worktree add --detach /tmp/probe origin/main` then
`uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
→ **1 failed in 48.92s**. So this is main's state, not a branch artifact.

**Offender set (two overlapping groups, all `payload=no` / `on_main=yes` for #2646).**

Heavy-import-before-`load_dotenv` VM entrypoints (10):

```
scripts/issue2617_standardized_ctx_answer.py        (heavy import L20, no load_dotenv)
scripts/issue2643_gradient_pursuit.py              (L23, none)
scripts/issue2643_marker_panel.py                  (L29, none)
scripts/issue2643_refusal_panel.py                 (L20, none)
scripts/issue2643_sae_map.py                       (L31, none)
scripts/issue779_ctxansviz_pc_specimens.py         (L27, none)
scripts/issue779_ctxansviz_separate_pca_dashboard.py (L22, none)
scripts/issue779_ctxansviz_separate_pca_fit.py     (L22, none)
scripts/issue779_ctxansviz_separate_umap_dashboard.py (L21, none)
scripts/issue779_ctxansviz_separate_umap_fit.py    (L23, none)
```

Bare Hub calls outside `HF_ROUTING_FROZEN_SNAPSHOT` / un-routed through
`hub.retry_transient` — `workflow_lint: FAIL (10 error(s))` across
`scripts/issue2643_{marker_panel,refusal_panel,sae_map}.py` and five
`scripts/issue779_ctxansviz_*.py` (bare `.list_repo_tree(` and
`hf_hub_download(` sites).

**A rebase does NOT fix it.** `origin/main` is 51 commits past #2646's base,
and the grandfather list is IDENTICAL on both (0 entries for these offenders
either side). The `test_regen_hf_routing_snapshot_live_tree_subset_of_current`
message's own discriminator applies: none of the offenders is
`workflow_lint.py` / `verify_plan.py` / `backends/gcp.py`, so the regen walker
did NOT diverge — genuinely-new offenders landed on main and the snapshot was
never regenerated.

## Acceptance criteria

1. `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
   passes on a pristine `origin/main` worktree.
2. `uv run python scripts/workflow_lint.py --check-live-hf-retry-routing`,
   `--check-hub-verify-retry`, and the no-flags run report no error naming the
   offender set above.
3. `uv run pytest tests/test_workflow_lint.py` green on pristine `origin/main`.
4. The chosen remedy is stated per offender: route through
   `orchestrate.env.load_dotenv()` before the heavy import / through
   `hub.retry_transient`, OR a justified waiver
   (`# NO_RETRY: <reason>` / grandfather-list entry) with the reason recorded.
   Do NOT blanket-grandfather to silence the check without stating why each
   site is safe.

## Scope notes

- Prefer the real fix (load_dotenv ordering, retry routing) over widening the
  grandfather list; the checks exist because uncapped torch/BLAS threads and
  un-retried Hub listings are live failure classes (#847, #920).
- `scripts/workflow_lint.py --regen-hf-routing-snapshot` on a main-synced tree
  is the mechanical half for the snapshot, but regenerating the snapshot only
  records the offenders as known — it does not make the calls retried.

## Distinct from the already-filed siblings (do NOT dedupe onto them)

- **#2497 / #2567** — the compare's SCAN-NEW-VIOLATION *mis-attribution* (the
  `tb=short` traceback location line entering the branch-attribution set).
  That is an instrument defect; THIS task is the underlying main red the
  instrument is reporting. #2646's run reproduced #2497's shape exactly: the
  compare reported `branch adds violation path(s) absent on pristine main:
  ['tests/test_shared_vm_thread_caps.py']` — the test file, not any of the 10
  real violation paths.
- **#2528** — `status` reporting fresh on a content-stale ledger.
- **#2645** — `cron_step9c_ledger_refresh.sh` committed non-executable, which
  is why the known-red ledger was 92.2h stale (sha `fceea4a40dbc`,
  `failing_tests: 2`) against the 7 failures actually observed.
- **#2513** — a different main-red (`test_argcheck.py`).

## Provenance

Surfaced by #2646's Step 9c gate (`kind: infra`, the CONCERN:: forwarder
fail-loud fix). #2646's own payload is gate-clean: `inline_lint_gate.py`
PASSED twice, certifying all payload paths by content hash, with
`touched_ruff_errors: 0` and the worktree at 101 ruff findings vs base 104.
