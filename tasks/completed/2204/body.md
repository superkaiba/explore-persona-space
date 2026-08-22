---
title: 'verify_plan: WARN when a plan registers a test-retest kappa demotion gate
  while pinning judge temperature 0'
kind: infra
tags:
- workflow-fix
- verify-plan
created_at: '2026-08-08T17:11:46Z'
has_clean_result: false
origin_prompt: 'Statistics critic on #2202 plan v1: temp-0 pin makes the registered
  test-retest kappa<0.6 demotion gate unfireable by construction; candidate WARN for
  verify_plan.py alongside the c27 gate-arithmetic family.'
workflow: v1
---
# verify_plan: WARN when a plan registers a test-retest κ demotion gate while pinning judge temperature 0

## Goal

Add a WARN-level check to `scripts/verify_plan.py` (alongside the c27 gate-arithmetic family) that fires when a plan text (a) registers a test-retest reliability gate — `test-retest` (or `test retest`/`retest`) co-occurring with `κ`/`kappa` and a demotion/threshold clause — AND (b) pins `temperature 0` (or `temperature=0` / `temp 0`) in the same judged-DV / judge-wave section.

## Why (incident)

#2202 plan v1 (2026-08-08, Statistics critic blocker): the plan pinned its Sonnet label wave at temperature 0 while registering a "200-item test-retest → κ per mode; κ<0.6 demoted to report-only" gate sourced to the #1738 convention. The parent instrument ran at API-default temperature (temperature 1.0, `scripts/issue1738_characterize.py:326`); its meaningful κ range (0.786–0.982) exists because sampled outputs vary across draws. At temperature 0 a byte-identical retest returns near-identical output — κ≈1 for every mode, the demotion gate can never fire, and the threshold is uncalibrated for the temp-0 surface. The gate was the SOLE instrument-validity screen for novel data-derived labels carrying BH-tested rate headlines: a gate that can only pass is a false positive by construction. This is the threshold-transplanted-across-read-surfaces failure family and is likely to recur in any judged-labeling plan.

A second, related trap worth a companion clause in the same check (or the WARN detail text): a rubric-keyed judge CACHE serving the first-pass verdict back to the retest row makes κ≡1 exactly at ANY temperature — the WARN detail should remind that retest rows need a distinct custom-id prefix (e.g. the #1738 `rt_` convention, `scripts/issue1738_characterize.py:303`) or a fresh `cache_dir`.

## Acceptance

- New check id (next free `c<NN>` slug, e.g. `c48_retest_kappa_temp0`), WARN-only, with a standalone N/A escape phrase (e.g. `N/A — no test-retest gate`) recognized by `_standalone_na_declared`.
- Detection is regex/section-scoped over the plan text (~5–15 lines), no semantic parsing; false-positive tolerant (WARN, never FAIL).
- Unit tests in `tests/test_verify_plan.py`: fires on the #2202 v1 shape (retest+κ gate + temp 0 in same section); silent when temperature is API-default/unpinned; silent when the N/A phrase is declared; silent when retest and temp-0 appear in unrelated sections (best-effort scoping).
- `workflow_lint.py --check-lessons-index` untouched (no new rule file needed); verify_plan docstring check-list updated.

## Provenance

Surfaced by the Statistics & Measurement critic on #2202 plan v1 (session cf372c0b, 2026-08-08) as an explicit workflow-fix prose follow-up; filed by the #2202 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups auto-file + spawn).
