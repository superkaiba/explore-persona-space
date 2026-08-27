<!-- epm:experiment-implementation v4 -->
## Implementation Report — round 3

**Status:** READY-FOR-REVIEW

### (a) What was done
Union fix round after FAIL+FAIL at code-review round 2 (Claude `epm:code-review v2` + Codex `epm:code-review-codex v2`). All SHAs pasted verbatim from git output.

- `scripts/issue2617_svmp_run.py`: **Blocker A** — new `_phase_input_gate(cfg, pending)` in `main()` runs model-free per-pending-phase prerequisite predicates (margin needs `judge_scores.json` OR judge scheduled earlier in pending; capture needs gen anchors OR gen earlier; finalize needs all 4 upstream sentinels), raising `RuntimeError("[input-gate] ...")` BEFORE `load_model_and_tokenizer`; margin gated only when `not cfg.tiny` (tiny uses canned pools, no judge dependency). **Blocker B** — `_margin_fp` extended with `anchors_sha` (`_anchors_sha`: sha16 over `[context_id, draw, text]` rows of the anchors JSONL; "absent" when missing; None under tiny) and `allow_short_pools`; new `_pools_content_sha` (refusal/helpful/meta, repro excluded) persisted into `margins.json` + `svmp_margin_done.json` and validated in `_margin_complete` against the recomputed value (old checkpoints lacking `pools_sha` are excluded). **Claude minors** — loop-time `PHASE_COMPLETE[phase](cfg)` re-check with an explicit invalidation print; `_finalize_fp` folds `_upstream_fps` (the 4 sentinels' recorded regime fps) so an upstream re-run invalidates the terminal sentinel. **phase-sentinels-not-durable residual** — `phase_finalize` stages `svmp_done.json` to `.finalize_stage/`, uploads for durability FIRST (`upload_dir_sharded`, `resume_skip=False`), records `durability_upload` in the payload, THEN writes the local terminal sentinel LAST. **judge-timeout-fallback-missing (runbook half)** — `phase_judge` docstring documents the exact kill-time salvage upload one-liner for the partial `judge_cache`.
- `src/explore_persona_space/eval/batch_judge.py`: **judge-timeout-fallback-missing (mechanism half)** — new module-level `_make_cache_write_through(cache, uncached_items, rubric_key)` factory; `judge_completions_batch` passes its return as `on_item_result` to `dispatch_judge_items`, so the SYNC path persists each item's verdict into `JudgeCache` the moment it lands (transport / truncation / api-refusal error dicts put-skipped, matching the terminal-loop filters). Terminal cache loop kept as the idempotent batch-path/backstop pass.
- `scripts/issue2617_svmp_reads.py`: **overflow-staging-disconnected** — overflow fallback now uses `repo_type="model"` (`DEFAULT_OVERFLOW_REPO` is a MODEL repo; the old `repo_type="dataset"` fallback could never resolve); new `_load_reroute_pointers(revision)` stages the durability `svmp_done.json` and unions `rerouted_paths` across all `upload_*` records; `_stage_with_overflow` consults pointers FIRST (pointer hit → direct overflow stage, never a doomed canonical read). **Claude minor (ii)** — `stage_ridge_payloads_svmp` returns the resolved revision, recorded as `summary["staging"]["ridge_revision"]` (null under tiny by design).
- `tests/test_issue2617_round3_fixes.py` (NEW, 19 tests): fp-invalidation (anchors content, allow_short_pools, pools content sha, repro-timestamp invariance, finalize upstream-fp folding), input-gate legs (missing judge_scores / anchors / sentinels; tiny margin exemption), `main()` spy tests (source-order pin: gate precedes load; margin/capture re-entry raises with ZERO model loads), loop-time re-check pin, finalize ordering pin, judge-cache write-through production-body test, overflow model-repo + pointer-first + pointer-union tests, ridge-revision return test.
- `tests/test_issue2617_round2_fixes.py`: margin-predicate test's synthesized sentinel updated to carry `pools_sha` (the new completion predicate validates it).
- Diff (this round's commit): +780 / −74 across 5 files. Cumulative branch vs `origin/main` (`git diff --stat origin/main...HEAD`): 7 files changed, 5232 insertions(+).
- Plan adherence: round scoped to the code-review v2 union punch list — every item DONE (see (e) + Response section); no plan-section deviations introduced.
- Commits: `7754a106cbaa7f106ddd8033db11509ecd4713ae` issue-2617 r3: model-free input gates, anchor/pool-content fingerprints, judge cache write-through, overflow pointer staging
- Branch: `issue-2617` pushed (bare push, rc=0). Prior tip `aaaa526b000bab33db48c8c282c553b0c88b1466`.

### (b) Considered but not done
- **Write-through on the Batch path**: `on_item_result` fires per item on the SYNC path only (`_judge_items_sync`); the Batch path's results land all-at-once at collection, where the existing terminal cache loop already persists them — a per-row batch callback would add no durability (the batch object itself is the durable store until collection). Documented in the factory docstring.
- **Extending `_margin_fp` with the full anchors sha under `--tiny`**: deliberately `None` under tiny (canned openers, no anchors dependency) — hashing a nonexistent file as "absent" under tiny would couple tiny re-entry to gen state it does not consume.
- **Backfilling `pools_sha` into pre-r3 checkpoints**: old margin checkpoints without `pools_sha` are treated as incomplete (re-run) rather than migrated — no production run has executed yet, so there is no banked state worth a migration path.
- **A generic reroute-pointer store**: pointers are read from the durability `svmp_done.json` only (the single sentinel `phase_finalize` uploads) rather than a new sidecar index — one fewer artifact class, and the sentinel is already the durability root.
- Nothing else material — the round tracked the punch list.

### (c) How to verify
- **Lint:** `uv run ruff check . && uv run ruff format --check .` — PASS (run this round pre-commit). **Ruff-policy pin (#1699):** `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset tests/test_no_direct_task_path_construction.py tests/test_no_pod_side_task_py_shellout.py tests/test_no_dollar_budget_caps.py -q` → 22 passed in 131.87s (repo-wide invariants union included per checklist 2c).
- **pin-sweep:** batch_judge.py + issue2617 scripts/tests fragments → 28 hit files: tests/test_api_dispatch.py, tests/test_batch_create_404_grace.py, tests/test_batch_judge_agg_non_dict_parse.py, tests/test_batch_judge_sharded_submit.py, tests/test_batch_judge_transport_cache.py, tests/test_batch_stuck_escape.py, tests/test_fleet_helper.py, tests/test_graded_judge.py, tests/test_i528_phase4_judge_pending_v2.py, tests/test_issue1074_base_negatives_regen.py, tests/test_issue1090_fu6.py, tests/test_issue1415_analysis.py, tests/test_issue1739_bareq_score.py, tests/test_issue1739_pvsynth_arms.py, tests/test_issue1739_wcrung_arms.py, tests/test_issue2092_route_parity.py, tests/test_issue2151_api_refusal_class.py, tests/test_issue2222_judge_mock.py, tests/test_issue2254_firstk_validators.py, tests/test_issue2617_judge_rubric.py, tests/test_issue2617_round2_fixes.py, tests/test_issue2617_round3_fixes.py, tests/test_issue545_claude_betley_aggregation.py, tests/test_issue663_batch_hardening.py, tests/test_issue779_stage1.py, tests/test_judge_dispatch.py, tests/test_judge_pilot_gate.py, tests/test_shared_vm_thread_caps.py; sweep_scope: selector-universe. ALL 28 run locally: **706 passed, rc=0** (132s). Selector `--json` `slow_tests_selected` = [tests/test_workflow_lint.py] — not in the hit set, nothing deferred.
- **pin-sweep:** `pools_sha` changed-literal grep → 2 grep-only hit files (tests/test_issue1112_margin_persist.py, tests/test_issue2221_pipeline.py — sibling issues' own pools_sha concepts, verified untouched by this diff): run locally, **145 passed, rc=0**; sweep_scope: repo-wide (grep-only supplement).
- **Regression tests for the blocker fixes** (permanent invariants, fail-pre-fix/pass-post-fix, all in `tests/test_issue2617_round3_fixes.py`): `test_main_margin_reentry_blocks_before_model_load` + `test_main_capture_reentry_blocks_before_model_load` (synthesized partial roots trip the `[input-gate]` RuntimeError with the loader spy recording ZERO calls); `test_main_input_gate_precedes_model_load` (source-order pin); `test_margin_fp_tracks_anchor_content` / `test_margin_fp_tracks_allow_short_pools` / `test_margin_complete_validates_pools_content_sha` (each drift axis flips the predicate to incomplete); `test_finalize_durability_upload_precedes_local_write` (ordering pin); `test_stage_with_overflow_fallback_uses_model_repo` + `test_stage_with_overflow_consults_reroute_pointers_first` (wrong repo_type / skipped-pointer regressions trip assertions).
- **Production-body test for the seam-stubbed write-through:** `test_judge_cache_write_through_persists_mid_wave` executes the REAL `judge_completions_batch` → `dispatch_judge_items` → `_judge_items_sync` bodies with a fake ONLY at the anthropic-client boundary (`_FakeMessages.create(**params)` asserts `{"model","messages","max_tokens"} <= set(params)` — signature-shape conformant, never a bare Mock) and observes `cache_dir.rglob("*.json")` non-empty BEFORE the wave's second call — the mid-wave persistence claim itself.
- **Bug-class self-sweep:** overflow repo_type class — `grep -n 'DEFAULT_OVERFLOW_REPO' scripts/issue2617_svmp_reads.py scripts/issue2617_svmp_run.py` shows every overflow op now passes `repo_type="model"`; no un-fixed sibling of the class remains in the issue scripts.
- **End-to-end commands** (happy path + 2 error paths): (1) `uv run python scripts/issue2617_svmp_run.py --tiny --phase all --out-root /tmp/x` → rc=0, 5 sentinels + margins.json with a `pools_sha`; (2) non-tiny `--phase margin` on an empty root → rc=1 `[input-gate]` naming judge_scores.json, no model load; (3) doctor a margin sentinel's `pools_sha` → margin re-runs instead of skipping (completion predicate refuses the mismatch).
- **Pod-side dispatcher / poller line:** unchanged from r2 — this round touched no sentinel-writer/poller contract surface beyond ordering `svmp_done.json`'s durability upload before the local write; the `[phase=done]` emission + sentinel schema are byte-identical.
- **What success looks like:** a killed/partial production run can NEVER cold-load the 7B model just to crash on a missing input (the gate names the missing file in seconds), a completed run's judge verdicts survive a mid-wave kill in `judge_cache/`, and every resume decision is content-keyed (anchors, pools, judge scores) rather than filename-keyed.

### (d) Needs human eyeball
- The write-through fires on the SYNC judge path only; the production plan routes large waves through the Batch API, where mid-wave durability is the batch object itself. If the production wave is forced sync (timeout fallback), the write-through is the load-bearing salvage — worth one glance at the `phase_judge` docstring runbook.
- `_load_reroute_pointers` returns an empty set (with a warning) when the durability sentinel does not exist yet — correct for pre-finalize reads, but means overflow-rerouted mid-run artifacts are reachable only via the EntryNotFound fallback until finalize has run once.
- Nothing touched authentication/secrets; upload surfaces reuse the existing `upload_dir_sharded` / `stage_hub_file` helpers unchanged.

### (e) Concerns addressed
- `overflow-staging-disconnected` — ADDRESSED (recorded via `task.py address-concern`, round 3): model repo_type + pointer-first staging + pointer-union loader; 3 tests.
- `judge-timeout-fallback-missing` — ADDRESSED (recorded, round 3): per-item JudgeCache write-through on the sync path + documented kill-time salvage upload; production-body test.
- `phase-sentinels-not-durable` (residual) — ADDRESSED (recorded, round 3): durability upload of `svmp_done.json` reordered BEFORE the local terminal write; ordering test.

## Smoke run
All smoke outputs wrote to /tmp scratch roots (`/tmp/issue2617-r3-*`) — never the committed `eval_results/` paths; `git status --porcelain -- eval_results/ figures/` empty in the worktree. Every VM launch carried the thread-cap prefix (`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2`) and `timeout --kill-after=30s` bounds.

### import-check
`uv run python scripts/issue2617_svmp_run.py --import-check` rc=0 (argcheck whole-module + argcheck-bind); same for `issue2617_svmp_reads.py` and `issue2617_svmp_figures.py`, both rc=0 — all at the r3 code.

### bank-check
`uv run python scripts/issue2617_svmp_run.py --bank-check` rc=0 — real pair bank loaded, per-category counts + budget filter pass (bank content referenced structurally only; no stems in logs).

### input-gate fix-engaged demo (Blocker A)
Non-tiny `--phase margin` on fresh root `/tmp/issue2617-r3-gate-demo`: rc=1 in seconds, log carries `[input-gate] pending-phase inputs missing (model-free, pre-load):` naming `judge_scores.json — run --phase judge first`; ZERO model-load lines (`/tmp/i2617_gate_demo.log`). Same for `--phase capture` / missing anchors (`/tmp/i2617_gate_demo2.log`). This is the fix-engaged signal for Blocker A: the raise fires before `load_model_and_tokenizer` on the REAL entrypoint.

### tiny e2e (all phases)
`--tiny --phase all --out-root /tmp/issue2617-r3-tiny`: rc=0; artifacts: anchors/, va_store/, judge/, margin/margins.json (carries `"pools_sha": "3b91dd152d334c8e"` — Blocker B persisted), manifests/, all 5 sentinels incl. `svmp_done.json` (`/tmp/i2617_tiny_all.log`).

### completed-scratch re-entry
Re-run on the completed root: rc=0, `[preflight] phases=[...] skipped=['gen','capture','judge','margin','finalize'] pending=[] (0.00s, model-free)`, ZERO model-load lines (`/tmp/i2617_tiny_reentry.log`).

### loop-time invalidation demo (Claude minor i)
Re-entry with `--allow-short-pools` flips margin pending (the new fp key); after margin re-ran, the loop printed `[loop] finalize: preflight skip invalidated by an upstream re-run this invocation — running` and finalize re-executed (`/tmp/i2617_tiny_loopdemo.log`) — the loop-time `PHASE_COMPLETE` re-check + `_finalize_fp` upstream folding, both engaged live.

### reads --local --tiny
rc=0; `summary["staging"]["ridge_revision"]` present (null under tiny by design — the non-tiny path records the resolved revision) (`/tmp/i2617_reads_tiny.log`; out roots /tmp/issue2617-r3-reads{,-stage}).

### figures
rc=0; PNG read-back confirmed non-empty axes + real plotted series at sane ranges (`/tmp/i2617_figs.log`; out root /tmp/issue2617-r3-figs).

### judge cache write-through
Covered by the production-body pytest (see (c)) — real dispatch bodies, fake only at the client boundary, cache observed populated mid-wave before call 2. No live API spend this round.

### Response to code-review v2
- **Blocker A (both reviewers) — model-free prerequisite gates before model load:** ADDRESSED. `_phase_input_gate` + spy tests + source-order pin + live rc=1 demos on the real entrypoint (zero loads).
- **Blocker B (both) — margin fingerprint blind to actual opener pools:** ADDRESSED. `_margin_fp` keys anchors content sha + `allow_short_pools`; `pools_sha` persisted in margins.json + sentinel and validated in `_margin_complete`; old rows excluded; 4 drift-axis tests.
- **Blocker C (both) — smoke-arch marker arm-registry grammar:** ADDRESSED. Re-posted as `epm:smoke-architecture-check v3` with the structured `arm-registry: source=PHASES file=scripts/issue2617_svmp_run.py n=5 members=capture,finalize,gen,judge,margin` line (command prose moved to `notes:`); draft-validated via `task_workflow.smoke_arch_registry_check` (ok=True) and confirmed via `task.py check-smoke-arch-registry 2617 --repo-root <WT>` (OK, rc=0). The resume-matrix mid-wave judge-cache leg upgraded FALLBACK→REAL (the write-through).
- **Codex: overflow-staging-disconnected:** ADDRESSED — model repo_type + pointer-first staging (see (a)/(e)).
- **Codex: judge-timeout-fallback-missing:** ADDRESSED — incremental write-through + documented kill-time upload command.
- **Codex: phase-sentinels-not-durable residual:** ADDRESSED — upload-then-local ordering in `phase_finalize`.
- **Claude minor (i) loop-time re-check + finalize fp folding:** ADDRESSED (live demo above).
- **Claude minor (ii) ridge revision recorded:** ADDRESSED (`summary.staging.ridge_revision`).
- **Claude minor (iii) marker head-sentinel version hygiene:** ADDRESSED — this marker posted at max+1 (v4) with the head sentinel matching; the smoke-arch marker posted version-omitted (CLI-derived v3).
<!-- /epm:experiment-implementation -->
