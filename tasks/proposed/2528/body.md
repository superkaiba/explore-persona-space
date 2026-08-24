---
title: 'workflow-fix: step9c_baseline.py status reports ''fresh'' on a content-stale
  ledger (2,134 commits behind main) — the 1d compare then classifies pre-existing
  main-side reds as NEW'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T08:38:19Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2315''s Step 9c gate: two red nodes absent from the known-red
  ledger both FAIL in the main checkout with no branch code involved; ledger main_sha
  is 2,134 commits behind origin/main yet status prints fresh (time-based predicate).
  Distinct from #2489 (cron file mode), which is today''s proximate cause.'
workflow: v1
---
# `step9c_baseline.py status` reports `fresh` on a content-stale ledger, manufacturing false-NEW gate blockers

## Goal

Make the Step 9c known-red ledger's staleness signal CONTENT-based, not time-based, so a ledger
whose `main_sha` has fallen far behind `origin/main` cannot read `fresh` and cannot cause the 1d
compare to classify pre-existing main-side reds as NEW.

## The gap

`scripts/step9c_baseline.py status` decides freshness from the ledger's refresh TIMESTAMP. It does
not compare the ledger's recorded `main_sha` against the current `origin/main`. So a ledger that is
recent in wall-clock terms but hundreds or thousands of commits behind main reports `fresh`, and
every main-side red introduced since the recorded `main_sha` is ABSENT from `failing_tests`. The
Step 9c 1d compare then classifies those reds as NEW — attributing another branch's (or main's own)
failure to the round under review, and blocking a clean round.

This is distinct from #2489. #2489 is the proximate cause of today's instance (the nightly refresh
script is committed at mode 100644, so the cron has never executed) and fixing it will re-freshen
the ledger. The predicate gap survives that fix: any future cron failure, credential lapse, or
simply a day of heavy main churn silently reproduces the same false-NEW regime, with `status` still
printing `fresh`. Fixing the file mode removes today's symptom; making staleness content-derived
removes the class.

## Measured evidence (2026-08-24, from #2315's Step 9c gate)

- Ledger `main_sha` = `e54ca2c1f3d141ab9ef224d2690dc7e4d1d8aaa8`, refreshed 2026-08-23T09:41:50Z,
  7 recorded failing tests. Current `origin/main` = `29539a51d3fd`.
  `git rev-list --count <ledger_sha>..origin/main` = **2,134 commits** of lag.
- `uv run python scripts/step9c_baseline.py status` prints `fresh` and exits 0 at that lag.
- Two nodes went red in #2315's gate:
  `tests/test_argcheck.py::test_bind_fleet_census_positive_coverage` and
  `tests/test_no_ungated_upload_call_sites.py::test_no_new_ungated_upload_call_sites`.
  Neither is in #2315's 3 changed files; both are ABSENT from the ledger.
- Run in the MAIN checkout with no branch code on the path, **both FAIL** (`2 failed in 16.37s`);
  the upload node fails on `scripts/issue1739_r2v2_run.py`, a sibling issue's script on main.
  So both are pre-existing, and ledger absence was the only thing suggesting otherwise.

#2315 stripped them on the direct main-checkout evidence rather than on ledger absence, but that
took an ad-hoc manual re-run per node — exactly the work the ledger exists to make unnecessary.

## Suggested direction (not prescriptive — the planner owns the design)

- `status` grows a content arm: resolve the ledger's `main_sha`, compare against fetched
  `origin/main`, and report STALE past a commit-count and/or touched-test-file threshold, keeping
  the existing time arm as an independent trigger. Exit codes already carry a `3 = stale` meaning.
- The 1d compare, on a STALE ledger, should refuse to classify absence as NEW silently: either
  degrade the NEW verdict to an explicitly-labelled `unclassifiable / baseline-stale` arm (the
  `classify-new-nodes` subcommand already has a `--out-unclassifiable` pristine-oracle-needed arm
  this could route into), or require a per-node direct main-checkout probe before blocking.
- Fail-safe direction: a stale or unresolvable ledger must never SILENTLY block a round on absence,
  and must never silently pass a genuine NEW red either — it should say which regime it is in.

## Provenance

Surfaced by #2315's own Step 9c gate round (2026-08-24) while adjudicating its two red nodes.
Root-cause sibling: #2489 (`cron_step9c_ledger_refresh.sh` mode 100644 — filed, `proposed`).
Ledger origin: #1022; nightly refresh cron: #2114. Corroborating evidence was also posted as an
`epm:progress` note on #2489.
