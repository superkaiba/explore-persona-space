---
title: 'Shared-tmp follow-up capture: with_suffix class + literal residue + atomic-DIRECTORY
  recipe + lint-arm widening + orphan-sweep hardening + unreadable-file bypass'
kind: infra
tags: []
created_at: '2026-08-25T15:39:18Z'
has_clean_result: false
parent_id: 2336
origin_prompt: 'Filed by the #2336 close-out per plan v4 §4 step 7 (follow-up capture,
  not dispatch); scopes accumulated across the batch 2-6 review rounds incl. the r10
  reconciler routing of live-atomic-temp-orphan-sweep-race.'
workflow: v1
---
# Shared-tmp follow-up capture: with_suffix class + literal residue + atomic-DIRECTORY recipe + lint-arm widening + orphan-sweep hardening

## Goal

Capture (NOT dispatch) the six residual scopes the #2336 migration deliberately routed out, per #2336 plan v4 §4 step 7. #2336 migrated every SHARED_TMP_LEGACY_ALLOWLIST member (119-file surface, allowlist now EMPTY) onto `explore_persona_space.atomic_io.atomic_replace`; these scopes are the adjacent hazard classes and hardening work that were out of its Goal.

## Scopes

**1. The `with_suffix`/`.suffix`-derived deterministic temp-name class.** Same hazard as the migrated six-arm surface, spelled via `path.with_suffix(path.suffix + ".tmp")` et al. Plan-time measurement (2026-08-23): ~148 lines / ~132 files. Filing-time regeneration (2026-08-25, broad pattern `grep -rn --include='*.py' -E '\.with_suffix\((.{0,60})\.tmp|\.suffix\s*\+\s*["'"'"']\.tmp' scripts/ src/`): 386 lines / 255 files — the class GROWS with fleet copy-paste, which is the argument for pairing the migration with a lint arm (scope 4). Regenerate the list at execution time; the broad pattern over-matches and needs the per-file audit discipline from #2336 §4.

**2. Literal-only residue files, file-by-file (the #2336 §4 step-7 list + review-round finds).** The 13 plan-named files (19 lines; ⚑ = HF_ROUTING_FROZEN_SNAPSHOT member, migration inherits the must-ask, #1547): scripts/issue1092_transfer_probe.py:278 ⚑, scripts/issue1112_rankem_dispatch.py:1042,1196, scripts/issue1333_dispatch.py:2424 ⚑, scripts/issue1336_dedup_sensitivity.py:141 ⚑, scripts/issue1336_diagnose_g1.py:304 ⚑, scripts/issue1336_gen_answers.py:287 ⚑, scripts/issue1738_sae_fullwidth.py:304,305,337,338 (memmap workfiles, the §4(h) ExitStack shape), scripts/issue1768_model_text_2x2.py:475,574, scripts/issue1776_jacobian.py:485, scripts/issue1900_figs.py:515, scripts/issue2091_analysis.py:2043,2049, scripts/run_experiment_389.py:1325, scripts/run_experiment_444.py:1050 ⚑. PLUS the review-round finds: scripts/issue2476_perrow_views.py:367, scripts/issue2222_reduce.py:674 (r10 Claude review), scripts/issue1739_features.py:119, scripts/issue1739_r2v2_run.py:295 (batch-5 non-roster residuals), and the 2 deferred with_suffix-class sites issue2222_judge.py:140 + issue541_geometry_extract.py:130 and 3 issue823_ladder_ext_gen.py with_suffix sites (batch-4/5 dispositions; overlap scope 1). PLUS a literal-shape lint ARM (needs `/`-division context engineering to stay low-FP) to close FN class (b) as a ratchet.

**3. The atomic-DIRECTORY publish/reclaim recipe** for the 3 §4(g) waived sites (issue1112_dispatch.py:1768 ⚑, issue1090_fu6.py:1222, issue1586_dispatch.py:3476): process-unique dir name + recursive failure cleanup + failure-residue tests + a directory-rename contract. New machinery the #2336 round-1 orchestrator directed out of that task.

**4. Lint arm-coverage widening** for the #2336 §10 false-negative classes (c)-(i) (wrapped literals, computed suffixes, os.path.join derivations, multi-line f-strings, ...) as measured value warrants — each arm needs fixture rows in tests/test_workflow_lint_shared_tmp.py per the established predicate-table pattern.

**5. Orphan-sweep hardening (r10 reconciler routing, concern `live-atomic-temp-orphan-sweep-race` on #2336).** The widened `*.tmp` orphan sweeps at scripts/issue2476_turnavg_sae.py:1318/:1462/:3613 unconditionally unlink every `*.tmp` match; under concurrent same-out-root invocations this can reap a live in-flight atomic_replace temp (`<name>.<pid>.<uuid8>.tmp`) and crash the owner's publish. Pre-existing hazard class (the old deterministic-shape sweep had the identical race plus collision corruption), zero realized concurrency today — hardening, not a regression fix. Options adjudicated by the r10 reconciler: name-scoped globs (`<name>.*.tmp` per known output stem), age-gated reap (mtime older than a conservative floor), or an out-root ownership lock. The same pattern generalizes to any future sweep beside atomic_replace writers.

**6. Lint unreadable-file bypass (open concern `shared-tmp-unreadable-file-bypass` on #2336).** The shared-tmp check silently skips files it cannot read; an unreadable (permissions/encoding) file bypasses the ratchet. Make the skip loud (WARN naming the file) or fail-closed.

## Provenance

Filed by the #2336 close-out per plan v4 §4 step 7 ("follow-up capture (not dispatch)"). Parent: #2336. Evidence trail: #2336 events.jsonl (r10 reconciler epm:review-reconcile v4; r9/r10 code-review markers; the batch 2-6 landed markers). Capture only — no session spawned; dispatch is the user/PM's call.
