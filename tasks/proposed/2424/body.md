---
title: 'Step 5a sibling-sync probe INPUT-SET gap: synced sibling SCRIPTS are never
  import-probed, breaking UNSYNCED covering tests at collection (fork-era src symbol;
  #2205 issue2094 crash)'
kind: infra
tags:
- workflow-fix
- step10d-lint-gate
created_at: '2026-08-20T13:04:28Z'
has_clean_result: false
parent_id: 2205
origin_prompt: 'Workflow-fix auto-filed by the #2205 orchestrator: 10d gate crash
  round 3 — Step 5a synced scripts/issue2094_figures.py (tip-era, imports figsize_iclr_full)
  while tests/test_issue2094_figures.py needed no sync and was therefore never probed;
  fork-era src/paper_plots.py lacks the symbol; TG_RC=2 collection ImportError ->
  verdict crash. Probe iterates only synced tests/test_issue*_*.py, so synced SCRIPTS
  are unguarded.'
workflow: v1
---
## Goal

Close the Step 5a sibling-sync **probe INPUT-SET** gap: the #2208 import-satisfiability probe iterates only files in `SIBLING_SYNCED` matching `tests/test_issue*_*.py`, so a synced sibling **SCRIPT** whose tip-era body imports a `src/` symbol added after the branch fork point is never probed — and it breaks the COLLECTION of an **UNSYNCED** test that imports it. The pair-atomic design assumes script and test move together; that assumption fails whenever the covering test needs no sync (its fork-era content already equals origin/main), which leaves the synced script's post-fork `src/` dependency completely unguarded.

Concretely (`.claude/skills/issue/steps/09-step-5.md`, sibling arm):

```
for f in "${SIBLING_SYNCED[@]}"; do
  case "$f" in
    tests/test_issue*_*.py)      # <-- scripts/issue*_* are NEVER probed
```

Candidate remedies (pick and justify one):
(a) extend the probe to synced SCRIPTS by probing the tests that COVER them — resolve via `select_step9c_tests.py --map-files` over the synced-script list, then collection-probe (or runtime-probe) that mapped set;
(b) sync the `src/` modules a synced sibling script imports, pair-atomically with it (widens the sync surface — needs care: `src/` is deliberately outside SPECS);
(c) treat any synced sibling script with a post-fork `src/` import as unsafe and skip it (fail-safe toward status-quo staleness, cheapest and most consistent with the existing revert remedy).

## Why (incident)

#2205 Step 10d round 3 (2026-08-20), the THIRD consecutive gate crash on this branch and the second distinct sibling-sync variant in one session:

- Step 5a synced `scripts/issue2094_figures.py` (status M) to tip. `tests/test_issue2094_figures.py` was NOT synced — its fork-era content already equalled origin/main, so it never entered `SIBLING_SYNCED` and was never probed.
- The tip-era script added `from explore_persona_space.analysis.paper_plots import figsize_iclr_full`. This worktree's `src/.../paper_plots.py` is fork-era (4 insertions / 234 deletions vs main) and has no such symbol.
- The gate's TG gated leg died at COLLECTION: `ImportError: cannot import name 'figsize_iclr_full'`, `TG_RC=2` -> `TG_CRASH=yes` -> verdict `crash`.
- Cost: a full gate run (~1h50m wall; TG legs sized `recommended-timeout-s=6540` over 85 mapped tests) plus diagnosis, then a pair-atomic revert of the two synced `scripts/issue2094_*` files and a second full gate run.
- All lint legs were GREEN in that same run (`GT_RC=0 BASE_RC=0 GATED_RC=0`) and the TG baseline was green (`TG_BASE_RC=0`) — the branch payload (`scripts/verify_plan.py` c46) was untouched by the failure. Pure vintage skew, blocking an innocent merge.

## Distinctness from the sibling filings (dedup evidence)

- **#2416** (same file, same session) covers probe **DEPTH** — collection-only vs runtime — for files that ARE synced tests (its incident: `test_issue1739_claim4.py` calling `ridge_layer_batched_auto(train_rows=...)` against fork-era src, a RUNTIME TypeError). A fix satisfying #2416's acceptance (i)/(ii) by runtime-probing the synced TESTS still misses this shape entirely: issue 2094 contributed NO synced test, only scripts. Different axis (input set vs depth), different failure stage (collection vs runtime), non-overlapping remedy.
- **#2420** covers the Step 5a `FAMILY_OF` map omitting workflow.yaml-derived pin tests. Unrelated arm (family sync, not the sibling arm).

The two should probably be scoped together by whoever picks them up — the shared root cause of #2416, #2420 and this task is that Step 5a syncs sibling `scripts/` + `tests/` but never their `src/` dependencies — but they are not the same bug and a fix for one does not close the others.

## Acceptance

- A synced sibling SCRIPT with a post-fork `src/` import is either not synced, reverted pair-atomically, or otherwise prevented from reaching the gate — such that the #2205 issue2094 shape cannot produce a `crash` verdict. State which remedy and why.
- The remedy holds when the covering test needs NO sync (the exact blind spot: `SIBLING_SYNCED` contains the script but not the test).
- Regression fixture reproducing the #2205 issue2094 shape: synced sibling script + fork-era `src/` stub missing the imported symbol + an unsynced covering test -> today a collection ImportError inside the gate; post-fix the declared remedy fires.
- Bonus (cheap, same arm): the probe currently emits nothing on success, so a silent no-op is indistinguishable from a probe that ran clean. Emit a per-file probed/skipped count.

## Provenance

Surfaced by the #2205 orchestrator at the Step 10d lint gate, round 3 (2026-08-20). Filed per `.claude/rules/workflow-fix-on-bug.md` (workflow-surface gap: `.claude/skills/issue/steps/09-step-5.md` sibling-issue file-sync arm, import-satisfiability probe input set). Sibling filings from the same branch: #2416 (probe depth), #2420 (family map), #2421 (contention-aware gate fences).
