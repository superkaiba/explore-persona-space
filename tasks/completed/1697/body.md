---
title: 'daily-fix: --map-files timeout dispersion margin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4f67014630ef
- daily-auto-filed
created_at: '2026-07-26T07:05:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The Step 10d TG leg for
  #1682 was timeout-killed at a 780s bound AFTER reaching 100 percent with 0 FAILED
  against a 751.7s measured baseline wall (bound = 1.04x measured); the map was dominated
  by 25 test_workflow_lint_* sibling files that carry no surcharge.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 Step C parked-workflow-fix-candidate routing pass
(`.claude/rules/workflow-fix-on-bug.md` § Recursion guard escape valve). The candidate was
parked on task #1682 at 2026-07-25T17:42:11Z because that session ran under the
`workflow_fix_target` recursion guard.

**Read the premise-correction below before planning.** The parked candidate's stated
mechanism ("the `--map-files` sizing path applies no workflow-lint surcharge") is
REFUTED by a compose-time semantic probe. The INCIDENT is real and reproducible; the
diagnosis is not. Filing the candidate verbatim would send this session chasing a
non-existent missing-surcharge branch.

## Goal

Give the `--map-files` `recommended-timeout-s` a real dispersion margin, and make the
slow-test surcharge cover the `tests/test_workflow_lint_*.py` sibling family rather than
only the exact `tests/test_workflow_lint.py` key.

## Workflow gap

- **Incident (verified):** the #1682 Step 10d TG leg was timeout-killed at a **780 s**
  bound AFTER reaching `[100%]` with **0 FAILED**, against a **751.7 s** measured
  baseline wall — bound = **1.04×** measured. It forced a crash-verdict gate re-run with
  a hand-doubled bound (recorded `epm:progress v7` on #1682).
- **It happened TWICE today, in two independent sessions** (surfaced by the `/daily`
  2026-07-25 transcript sweep, so this is not a single-observation filing):
  - #1675 (session `5cfcf606`) @ 2026-07-25T13:50:28Z — same **780 s** selector-derived
    bound, measured wall **728.19 s** (93%), gate reported `crash` on
    `440 passed in 728.19s (0:12:08)`; hand-doubled to 1560 s and re-ran.
  - #1682 (session `838b76a5`) @ 2026-07-25T16:35:50Z — same **780 s** bound, measured
    wall **751.70 s** (96%), `459 passed`; hand-doubled to 1560 s and re-ran.
  Cost: ≈35 min + ≈34 min = **~69 min of pure gate re-run** in one afternoon. Both
  sessions self-identified it as "the #1634/#1646 undersized-bound class". The
  hand-doubled 1560 s figure both sessions independently chose is a useful prior for the
  dispersion factor, but it is a guess — prefer a measured basis (see below).
- **Premise correction (compose-time semantic probe):** the surcharge is NOT missing on
  the map path. `recommended_timeout_s(tests, floor=...)` sums `SLOW_TESTS` surcharges on
  BOTH paths, and `--map-files` calls it with `floor=MAP_TIMEOUT_FLOOR_S`. Probed
  directly at HEAD:
  - `recommended_timeout_s(['tests/test_workflow_lint.py', ...], floor=MAP_TIMEOUT_FLOOR_S)`
    → **2580** (the 2400 s surcharge DOES apply on the map path).
  - `SLOW_TESTS` == `{'tests/test_workflow_lint.py': 2400}` — a single exact-path key.
  The real cause is that `tests/test_workflow_lint.py` was **NOT in the #1682 map at
  all**. Reproduced: `select_step9c_tests.py --map-files <#1682 touched files>` yields
  **26 tests, of which 25 are `test_workflow_lint_*.py` SIBLINGS**
  (`test_workflow_lint_walks.py`, `_v2_checks.py`, `_judge_model_check.py`, …) plus
  `test_guard_piped_git_push.py` etc. — and `grep -x 'tests/test_workflow_lint.py'` over
  that map returns nothing. Those 25 siblings carry **no** surcharge, so the bound is the
  flat `TIMEOUT_BASE_S + TIMEOUT_PER_FILE_S × n` model (today: `120 + 30×26 = 900`; at
  incident time, 22 tests → **780**, matching the observed bound exactly).
- **So the two real defects are:**
  1. **No dispersion margin on the map path.** A bound of 1.04× the measured wall is
     below the project's own `≥2×` p90-dispersion default
     (`.claude/rules/plan-compute-sizing.md`), so a green leg dies on ordinary
     run-to-run variance. The flat `120 + 30/file` model tracks file COUNT, not cost.
  2. **The surcharge key is exact-path.** The `test_workflow_lint_*` sibling family
     dominates any map touching `scripts/workflow_lint.py` and is collectively slow, but
     matches no `SLOW_TESTS` key.
- **Why it is a workflow gap:** this is the #1634/#1646 gate-timeout-sizing class
  recurring on the map path. #1646 (`fa742e77b3`, 2026-07-24 04:51 PT) landed the
  surcharge — BEFORE this incident — which is exactly why "the surcharge is missing" is
  the wrong diagnosis.
- **Confidence (emitter):** high on the incident and the corrected mechanism; medium on
  the right remedy shape (dispersion factor vs sibling-family surcharge vs measured
  per-file costs — the planner decides).
- verified-at-filing: per-target probes on `scripts/select_step9c_tests.py` —
  `grep -n 'SURCHARGE\|surcharge\|2400'` → **7 hits**, incl. `SLOW_TESTS`
  `"tests/test_workflow_lint.py": 2400` at line 709 and `recommended_timeout_s` at
  line 784 (so the presence claim binds, and its CONTEXT shows the surcharge already
  implemented — hence the premise correction above, per rule clause (c)). Semantic probe
  (clause (a')): imported the module and called `recommended_timeout_s` directly, results
  quoted above. Map reproduction: `git show --name-only 841304c2d0` → 5 touched files →
  `--map-files` → 26 tests, `tests/test_workflow_lint.py` NOT among them. Landed-fix
  history check `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py`
  → 8 commits (#1688 `ab45d65777` today, #1673, #1662, #1656, #1651, #1649, #1646
  `fa742e77b3`, #1645); none adds a dispersion factor or a sibling-family surcharge.
  SHA checks: `git rev-parse --verify --quiet` resolves `fa742e77b3`, `841304c2d0`,
  `ab45d65777`. (2026-07-25)

## Proposed change (planner's call — two candidate shapes, not mutually exclusive)

```
A) dispersion margin on the map path:
   recommended_timeout_s(..., floor=MAP_TIMEOUT_FLOOR_S) -> apply a >=2x
   dispersion factor (or raise TIMEOUT_PER_FILE_S) so the bound is not ~1.0x
   the measured wall; matches the p90 x2 default in plan-compute-sizing.md.

B) sibling-family surcharge:
   SLOW_TESTS lookup becomes prefix/glob-aware for tests/test_workflow_lint*.py,
   or the family gets its own (smaller, per-file) surcharge — measure first,
   do not guess the constant.
```

Prefer a MEASURED basis for whichever constant lands: `scripts/step9c_baseline.py` already
holds measured per-file walls (#1682's own 751.7 s figure came from it). An asserted
constant is the failure mode #1646 already paid for once.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py` (`SLOW_TESTS`, `recommended_timeout_s`,
  the `--map-files` sizing branch).
- Check whether the Step 9c diff path needs the same dispersion treatment or is already
  covered by its higher `TIMEOUT_FLOOR_S` (900) — state the answer either way.
- Pin the new behaviour with a test in the existing step9c selector test family.
- **Advisory — open sibling on the same file:** `#865` (`on_hold`, `wf-fix`,
  `wf-fix-fp:d2551dac39f5`) targets `scripts/select_step9c_tests.py` for a DIFFERENT bug
  (the selector diffs the main checkout and is blind to worktree branches). Verified
  distinct: different fingerprint, different function (`compute_touched` /
  `_resolve_repo_root` vs `recommended_timeout_s`). Not a duplicate — but read #865
  before editing so the two fixes do not collide.

## Constraints / invariants

- Workflow-surface only.
- Do NOT weaken the gate to make it pass: raising a timeout bound is legitimate; skipping
  or de-selecting tests is not.
- Whatever constant lands must be justified from a MEASURED wall (cite the baseline
  ledger row), never asserted.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes; the
  step9c selector pin tests stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: 4f67014630ef

Parked candidate (verbatim), from task #1682 `events.jsonl` @ 2026-07-25T17:42:11Z —
retained for the record; its `bug_observed` / `proposed_change` mechanism is superseded by
the premise correction above:

<!-- workflow-fix-candidate v1 -->
target_file: scripts/select_step9c_tests.py
bug_observed: the --map-files recommended-timeout-s (780s here) under-sizes the Step 10d TG legs when tests/test_workflow_lint.py is in the map — the gated leg was timeout-killed AFTER reaching [100%] with 0 FAILED (baseline measured 751.7s, 96% of the bound), forcing a crash-verdict gate re-run with a hand-doubled bound (recorded epm:progress v7 on #1682).
why_workflow_gap: the Step 9c selector applies a +2400s surcharge when test_workflow_lint.py is selected, but the --map-files sizing path applies no such surcharge, so a green run gets killed at the bound (the #1634/#1646 class recurring on the map path).
proposed_change: apply the test_workflow_lint.py surcharge (or a >=2x dispersion factor) to the --map-files recommended-timeout-s sizing, mirroring recommended_timeout_s().
diff_sketch: |
  in the --map-files sizing branch:
  + if any mapped test file is tests/test_workflow_lint.py: timeout += WORKFLOW_LINT_SURCHARGE_S
confidence: high
related_task: #1682
<!-- /workflow-fix-candidate -->
