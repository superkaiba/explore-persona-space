

# ===== [g1 b42275d162] =====

# Code Review: #2587 split-review r2 — group g1 (commit b42275d162, unit R2b: analysis/fits round-1 fixes)

**Verdict:** FAIL
**Tier:** leaf (two analysis entrypoints + their tests; no in-round importer) — reviewed at trunk depth given the registered-lattice content.
**Scope:** commit b42275d162 only (`git show`, +826/−86 across 4 files); round gates 0.5/0.55/0.6/0.8/0.9 SKIPPED per brief (CONTRACT-BEARING: no). Plan v3 confirmed via `readlink` on the main-checkout symlink.
**Tests actually run:** yes — commit-state ISOLATED run (files extracted from `git show b42275d162:` into /tmp with symlinked script deps; the four files drifted at HEAD via R2d1/R2d2, so a worktree-HEAD run would not be commit-state): `80 passed in 20.06s` (40 fits + 40 analysis), matching the claim. Ruff check + `ruff format --check` clean with the repo config. `workflow_lint.py --check-shared-tmp-name` → PASS tree-wide (Fix 6 verified; `atomic_io.atomic_replace` tmp = `<name>.<pid>.<uuid8>.tmp` — process-unique, never matches the `*.pt` upload globs; `save_pt_atomic(path, obj)` call binds the real signature).

## Blockers

1. **[Critical] The missing-fire-defaults-true fix falsely kills the `query_form` axis on BOTH sides — proven live against the committed artifacts.** `scripts/issue2587_analysis.py:1240` (`UNCHECKED_CLASSES` omits `query_form`), `:1262` (a query_form endpoint vid → `(False, True)` = NOT fired + "missing"), `:1743-1747` (`axis_row_missing` keys on `GRIDLESS_CLASSES`, which also omits `query_form`). The production banks carry 36 `query_form` pairs per side with vids {E, imp, stmt} (committed `eval_results/issue_2564/bank_manifest.json`; the 9B bank inherits them — `bank2587.EXPECTED_PAIR_COUNTS["query_form"]=36`); neither fire artifact carries `query_form` value rows or an axis row (committed parent `manipulation_check.json` = instruction axes × v1..v5p only; `issue2587_judge.py:913-947` emits axis rows only for INSTRUCTION_AXES + answer_language + query_content_oneword). Live probe on the REAL committed inputs: `pair_fired_mask` → fired-both = **0/36**, missing-count = **36**; `axis_row_missing=True` → `compliance_limited=True` — while the parent's committed `minpair_delta.json` (the port-parity reference on identical inputs) records `compliance_limited=false, n_headline_pairs_fired70=36`. Failure scenario: the production analysis run nulls the registered per-axis headline read for `query_form` on both sides, reports 36 false "missing fire row" drops per side in the new `n_missing_fire_rows_*` fields (`:2531`), and mislabels the crossmodel `query_form` row as a compliance fallback (`symmetric_headline=false` via the `head_ok and sym.any()` gate at `:2515-2517`) — contradicting the commit's own stated intent ("query classes … carry no manipulation check and stay unfiltered"), the parent convention, and plan v3 line 115's fire-gate scope (fire-gating quantifies over manipulated values; query classes have nothing manipulated). Fix: exempt the full query-class set at BOTH arms — value level (e.g. key the exemption on the pair's `cell == "query"` or a complete query-class tuple including `query_form`) and axis level (same criterion instead of `GRIDLESS_CLASSES`). Note the test suite is structurally blind here: `tests/test_issue2587_analysis.py::_parent_pairs_and_contexts` builds no `query_form` pairs — add one to the fixture so the exemption is pinned.

## Round-1 concern dispositions (this group's two named items)

- **matched7b-resume-contract (BLOCKER) — FIXED, verified.** `scripts/issue2587_fits.py:851` (`_matched7b_completion_gaps`): resume skip now requires complete record + regime match + REQUESTED upload contract (mode + `preds7b_prefix`, key names verified identical to the fresh-path record at `:1162-1183`) + a valid sentinel (done + regime match); any gap routes to `_matched7b_repair` (`:878`): refuses without the 3 persisted arm files, re-uploads with `resume_skip=False` when the requested hf contract is unmet (refusing a missing `--preds7b-prefix`), updates the record (`repaired: true`), rewrites the sentinel via the single shared derivation (`_matched7b_sentinel_path`, used by writer + predicate + repair). Regime mismatch still halts BEFORE any skip/repair (reordered correctly). Crash-window analysis: crash between upload and record-update re-repairs idempotently; between record and sentinel → sentinel-only rewrite, no re-upload. `regime_key` correctly excludes upload (completion-predicate concern, not a regime). Six new tests exercise the production `run_matched7b` body with `create_autospec` only at the upload boundary (changed-upload repair, record→sentinel crash, genuine-complete skip, repair-impossible halt, regime-mismatch halt, gaps predicate matrix).
- **missing-fire-defaults-true (CONCERN) — PARTIALLY fixed; the residual is Blocker 1.** The core mechanism is correct and pinned: a fire-CHECKED value with no row is NOT fired, dropped from both endpoint masks, counted per axis (`n_missing_fire_rows`) and per crossmodel row (`n_missing_fire_rows_9b/_7b`); `fire_missing`'s first-threshold shortcut is sound (`load_fire` populates all three thresholds per row, so membership is threshold-independent); the E/bare install-endpoint exemption is safe ("E" appears only as the unmanipulated install a-side in both banks). But the exemption set is incomplete (query_form), so the fix as committed introduces a new false-kill on an adjacent registered axis.

## Other fixes verified (no findings)

- **Fix 2 (`_ref7b_stat` fail-loud, analysis:2443-2479):** verified live over the REAL parent artifact — all 66 (11 axes × 6 stats) extractions: the only raising combos (crossfam/identity on query axes, 5 KeyErrors) are exactly those the caller structurally skips before reaching `_ref7b_stat` (`mask is None → continue` at `:2651/:2658`; sole call site `:2673`), matching the docstring's caller-scoping claim. Unknown stat → ValueError, pinned.
- **Fix 3 (metadata):** `primary_h2_7b_arm` rides side_meta, h1.json, every perpair row, and both perdraw npz sites via `_savez_with_meta` (grep: no remaining bare `np.savez` outside the helper; no in-round consumer iterates npz keys, so the added key is safe). Pinned by 3 tests.
- **Fix 4 (`_resolve_ref7b_default_commit`, analysis:553-587):** UNRECORDED sentinel gone; derives `git log -1 --format=%H -- <rel>` gated on a clean `status --porcelain` for the path, halts (SystemExit naming `--ref7b-parent-commit`) on outside-repo / gitless / dirty / untracked / no-commit; explicit flag wins. Pinned by 2 tests incl. a real tmp git repo.
- **g5 m2 (edge-selected refusal):** finalize halts on any persisted checkpoint with non-null `lambda_grid_edge` (`fits:703-710`; schema verified against the checkpoint writer `:636`); matched7b refuses persisting an edge-selected arm (`:1051-1057`). With `extend=True` the fit structurally cannot return a non-null edge (returns only on `edge is None`, else raises past MAX_GRID_EXTENSIONS), so both refusals scope to `--no-edge-extension` diagnostic fits as intended.
- **g7 m4:** compositional-asymmetry disclosure added to `calibration_ratio_to_global`'s stats_def (text-only).

## Concerns (non-blocking)

1. `scripts/issue2587_fits.py:862-865` vs `:892-894` — the upload-contract predicate is duplicated verbatim between `_matched7b_completion_gaps` and `_matched7b_repair`; a future edit to one silently desyncs the other (repair could re-upload when no gap was declared, or skip an upload the gaps list demanded). Suggest repair consume the computed gaps list.
2. `scripts/issue2587_fits.py:866-871` — a corrupt/unreadable sentinel is silently coerced to `{}` and routed to repair (rewrite). Fail-safe direction and the record stays authoritative, but the corrupt artifact is overwritten rather than quarantined; a one-line log of the parse failure would keep the forensic trail.

**Recommendation:** FAIL — re-roll confined to the query-class exemption in `pair_fired_mask` + the `axis_row_missing` guard (one commit; add a `query_form` pair to the test fixture bank so the exemption is pinned). Everything else in this commit is verified sound.


# ===== [g2 6b614a4747] =====

# Split-review r2 g2 — commit 6b614a4747 (unit R2a: pod workload launcher + P1 enforcement + fork item 8)

## Code-Reviewer Verdict — CONCERNS

**Tier:** trunk (the launcher orchestrates every production pod phase across four scripts; `issue2587_map_gen_capture.py` has multiple callers — launcher, tests, its own fits-smoke subprocess).

**Scope:** commit 6b614a4747 only (`git show 6b614a4747`, 68.8 KB, read in full). CONTRACT-BEARING: no — Steps 0.5/0.55/0.6/0.8/0.9 skipped per brief.

### Blockers

None.

### Concerns

1. **HF-prefix override threads into producers but not consumers** — `scripts/issue2587_pod_workload.sh:74` (`HF_PREFIX="${EPM_I2587_HF_PREFIX:-issue2587_q35_map/qwen35_9b}"`) vs the P4/P8 fits invocations (`scripts/issue2587_pod_workload.sh:~400-470`), which pass no `--store-prefix`; `issue2587_fits.py` then reads its internal default `STORE_PREFIX_9B = "issue2587_q35_map/qwen35_9b"` (fits.py:106). Failure scenario: a run launched with a non-default `EPM_I2587_HF_PREFIX` (e.g. a smoke twin under `issue2587_minpair/smoke/` per plan §10) uploads P2/P3 captures to the overridden prefix while P4 fits consume the DEFAULT prefix — a missing-store fail-loud in the clean case, but a silent stale-store read if a prior production run already populated the default prefix. Defaults match exactly today (verified byte-equal), so the plan's own dispatch is unaffected; either thread `--store-prefix "$HF_PREFIX"` into the P4/P8 legs or document the env knob as dryrun/test-only.

2. **`--gate compose_p1` sits behind the HF_TOKEN assert** — `issue2587_map_gen_capture.py:3438` (token assert) precedes the gate dispatch at `:3453`, yet every compose_p1 check is local (interpreter/pins/banned-dists/driver/run_meta/manifests). Failure scenario: on a pod whose `.env` failed to stage, the FULL-P1 verdict composer dies with "HF_TOKEN missing" instead of a compat verdict — misattributed failure class, though still fail-loud. The two genuinely local modes (`--fits-smoke`, `--p1-apply-probe`) are correctly dispatched before the assert; compose_p1 could join them.

3. **A failed bg shard orphans its live sibling** — `scripts/issue2587_pod_workload.sh` `wait_bg` (exits the launcher on the first non-zero pid) after paired `launch_bg` calls in P2/P3/P4/P5/P6. Failure scenario: shard0 crashes early, the launcher exits, and shard1's vLLM process keeps holding GPU 1 until the relaunch's kill-before-relaunch discipline reaps it. Matches the inherited #2330 pattern and the crash-fix-rounds rule covers relaunch; noting for the operator runbook.

4. **Nit:** `run_p1_apply_probe` asserts `n_rows >= 1` (`issue2587_map_gen_capture.py:3172`) but `X.std(dim=0)` at `:3206` is NaN at n_rows==1, so a 1-row store would fail on the downstream finiteness assert with a misleading message. The launcher's tiny cell (3 carriers × K=2) can never produce n_rows < 2; unreachable in practice.

### Round-1 concern disposition (this group)

- **missing-pod-workload (BLOCKER) — FIXED.** `scripts/issue2587_pod_workload.sh` (new, 565 lines) implements the plan-§4.7 pod DAG end to end: bootstrap (issue-2564 object fetch, §4.1 venv build via `issue2587_common.build_model_venv`, driver gate + cuda-compat remediation) → P0b (template_pin → length_scan → hook_probe, `split_ids_done.json` asserted) → P1 smoke (500-row shard gen+capture `--no-upload` → fits-smoke → tiny battery cell `--upload none` → apply probe → compose_p1) → P2/P3 (6 splits × 2 CVD-pinned shards; SPLITS array verified == `SPLIT_TO_MANIFEST` keys) → P4 layer-sharded fits + finalize → P5/P6 battery gen/capture + repo-venv embed (plan's P7, folded in) → P8 matched7b → results_push. Header dispatch command matches plan §9 line 347 verbatim. Pod-side reporting conforms: no `task.py` shellout; `epm:results` sentinel carries all `poll_pipeline._SENTINEL_REQUIRED_KEYS` (`sentinel_schema_version`/`kind`/`version`, poll_pipeline.py:1448) at the drain-glob path `/workspace/logs/issue-2587-epm_results-<epoch>.json` (top-level, `issue-<N>-*.json` shape); results push uses explicit-pathspec commit + the sanctioned `git push > log 2>&1; rc` composition + rev-list==0 + per-file ls-tree remote verification (#1205); HF mirror is ONE `upload_folder` commit. CVD is launcher-pinned per shard (drivers write it nowhere — r1 union item 7). Per-leg out-roots split P1 smoke/battery from production (#1333).
- **compat-gate-not-enforced (BLOCKER) — FIXED.** `--gate compose_p1` verifies the FULL §4.7 P1 set in the MODEL interpreter: interpreter identity (`model_python()` resolved), realized `MODEL_VENV_PINS`+`EXTRA_PINS`, banned-dist absence (version AND find_spec), `assert_driver_compat()`, all six `P1_COMPOSE_REQUIRED` run_meta PASS records, tiny-battery manifests. All six records have REAL producers with `passed: True` (mapgen: template_pin:1531, length_scan:1640/1671, hook_probe:1974, smoke_shard:2458 on the exact `--no-upload` shape the launcher uses, fits_smoke:3135, apply_probe:3237) — the test fixture's consumer-authored run_meta is corroborated by producer code, not load-bearing. Report written ALWAYS, sentinel ONLY on all-PASS, rc 5 kills the launcher via `run_logged`; `require_p1` re-asserts `status == PASS` before ALL SIX production waves (static + dry-run tested; sentinel path threaded consistently `$OUT_ROOT/compat_smoke_done.json`). Producer parity for the apply probe verified against `issue2587_battery_run.py`: store paths `capture/{va2587,vc2587}/<cell>.pt`, keys `layers/hidden/va_tail_incl/rows`/`vc` (battery.py:1192-1226), manifests `{anchors,capture}_<cell>.done.json` with `n_rows` (:731/:1253), `--upload none` keeps local bytes (delete only on the hf branch :1229-1240), CAPTURE_LAYERS = all 32 ⊇ layer 22; `apply_map` ridge-branch payload keys/dtype contract match (`issue779_ffc_n1m_fits.py:901-916`, returns (n, D) float64).
- **phase-not-idempotent (Codex tag) — ADDRESSED for the launcher DAG.** Relaunch is safe: phases are strictly sequential and fail-loud, so no stale sentinel can shield a failing gate (compose_p1 always re-runs and rewrites `compat_smoke_done.json` BEFORE the first `require_p1`); `length_scan` re-run post-drop takes the PASS branch (and fork item 8 closes the crash window — split_ids mutation now precedes the `passed:true` record, with a fails-pre-fix test forcing the record write to raise and asserting the drop already landed); expensive waves rely on driver-level resume (battery gen `_gen_done` manifest skip battery.py:674, capture manifest skip :1128, P2 chunk-grain fingerprint resume per plan §checkpoint-cadence, fits cache-dir); `write_sentinel` overwrites atomically; `results_push` has an explicit no-change branch. Residual recompute on relaunch (P0b gates + P1 smoke, ~0.5 h) is bounded and by design.

### Plan adherence

Matches plan v3 (§4.7 DAG, §9 sizing/dispatch, §10 workload command + harvest set). Deviations found: none substantive. `RESULT_JSONS` matches the §10 pre-teardown harvest set at this commit (extended by the later R2d1-2/3 commit — other group's scope). `split_ids.json` absent from the tree is by design: `gate_length_scan` bootstraps it in place (mapgen:1589) and `_maybe_write_p1_sentinel` early-returns on missing records, so the template_pin-first ordering is safe.

### Tests

`uv run pytest tests/test_issue2587_map_gen_capture.py -q` → **46 passed in 7.22s** (at worktree HEAD; `issue2587_map_gen_capture.py` is byte-identical to this commit's blob — only the launcher/tests were extended additively by 49a6c55eb8). New coverage is real, not import-only: fails-pre-fix drop-ordering test, apply-probe happy/reject paths on torch fixtures, compose_p1 PASS/FAIL×4 with report+sentinel assertions, launcher `bash -n` + static contract + a real dry-run execution asserting phase order, per-wave require_p1 placement, derived (never retyped) §4.1 env pins, all-six-splits coverage, and universal log redirection.

### Recommendation

Both of this group's round-1 blockers are fixed and test-pinned. Concerns 1-2 are cheap hardening (thread `--store-prefix "$HF_PREFIX"`; move compose_p1 ahead of the token assert) — neither blocks the round.


# ===== [g3 4ccc5e068d] =====

<!-- split-review r2 g3 — commit 4ccc5e068d09a83e9782f26607eb479d274b33b1 -->
## Code-Reviewer Verdict — CONCERNS

Scope: issue #2587 round-2 group g3, commit `4ccc5e068d` (R2c: `scripts/issue2587_figures.py` +2010/−118, `tests/test_issue2587_figures.py`). Diff sized 109,393 B (< 300 KB budget); full file read (1,933 lines) + tests. No other round commit touches these files (verified `git log b043c0ccea..HEAD -- <files>`). Plan v3 symlink asserted (`readlink` → `v3.md`). CONTRACT-BEARING: no — round-level gates 0.5/0.55/0.6/0.8/0.9 skipped per brief.

### Round-1 concern `plan-s6-figures-deliverable-gap` — VERIFIED FIXED
- Plan §6 "Figures." + §13 enumerate 2 heroes + 17 exploratory-dump items + the manipulation-check table (20 named items). All 20 map to registry entries; the 4 extras (`matched_n_table` §4.6, fit-side `knn_per_layer` + `reliability_ceiling` §6 DV rows, `crossmodel_delta_forest` convention-10 support) are legitimate over-production. Mapping is documented per-entry in the module docstring (lines 13–49) with the §6 name quoted.
- Every renderer is a REAL implementation (no stubs): each consumes validated inputs and writes via `savefig_paper` / md+json writers; the r1 "inputs the registry doesn't load" gap is closed (`perpair` jsonl, `manip9b/manip7b`, `bank`, `leakdir` loaders added, `_INPUT_SPECS` lines 1808–1824).
- Consumer↔producer schema trace (all verified against same-branch producer CODE or committed artifacts, not fixtures alone): battery done-manifests (`issue2587_battery_run.py:739–748` — `cap_hit_frac`/`cap_hit_frac_regen`/`think_leak{n,n_leaked,frac}`), map-side `cap_hit_<split>.json` (`issue2587_map_gen_capture.py:2567–2578, 2641`), `bank2587.run_token_gates` keys (`bank2587.py:596–601`), delta-doc sides/meta/direction-ci95/null-q2_5/calibration/reliability-r10/`layer_twins`/`pooling_twin_span`/`per_vp_cos`/`retrieval.global`/`edit_dose_ols` (`issue2587_analysis.py:1963–2323, 3010–3028, 3270–3281`), h1 keys (`:2525–2535`), crossmodel `s_9b/s_7b/s_7b_ref_parent/delta_ci95` (`:2822–2826`), fits sweep/matched7b/anchor keys (`issue2587_fits.py:630–644, 784–794, 1189–1205`; `issue2330_matched_fits.py:341–395`), 9B judge `axis_rows` incl. `not_in_slice` special rows (`issue2587_judge.py:572–585, 913–940`), the COMMITTED parent artifact `eval_results/issue_2564/manipulation_check.json` (main checkout; axis_rows keys `axis,floor,floor_met,n_fired_base,…,width` — exactly what `_cell` reads; 0 special rows; `value_rows` present), and the committed #2330 refs (`matched_fits_q{35,25}_n10k.json` — `layers` + `per_layer[str].ridge.test_r2`).
- Drift guard: `test_registry_covers_plan_deliverables` pins the exact 24-name set (tests/test_issue2587_figures.py:1075–1104).

### r1 Minor 1 (blank crossmodel render) — VERIFIED FIXED
`_require_stat_axes` (scripts/issue2587_figures.py:293–301) wired into `fig_crossmodel_axis_profile` (:697), `fig_crossmodel_delta_forest` (:852, both stats), `fig_matched_vs_parent_scatter` (:902 + zero-finite raise :941–946); `_delta_sides` (:304–319) rejects side-less/empty-axes delta docs; per-figure `n_finite == 0` / empty-class / no-pilot / no-twin raises at :1016, :1143, :1185, :1287, :1329, :1411, :1515; `_load_jsonl` zero-rows raise (:242), `_load_leak_dir` empty-dir raise (:258). Tested (`test_crossmodel_empty_axes_fail_loud`, `test_install_swap_violins_empty_class_fails_loud`, `test_leakdir_empty_fails_loud`, `test_pilot_panels_require_pilot_axes`, `test_token_count_equality_requires_gates`).

### r1 Minor 2 (`delta?` optionality) — ADDRESSED
Under `--figs all`, `delta` appears BARE in `crossmodel_axis_profile`/`edit_dose_scatter` requirements, so `main()`'s strictest-wins optionality (:1905–1910) makes it REQUIRED on the production path — the H1 rows cannot be silently absent from a full run. Optional only on a hand-scoped `--figs matched_n_table` run (tested, `test_cli_optional_delta_absent`).

### Figure conventions (brief-named)
- ONE display-name map: `DISPLAY` (:126–169) + `axis_label` (:177); every rendered legend/title/table header I traced routes through it; md slug-freedom test-asserted.
- No caption blocks inside plots: zero `fig.text`/`figtext` in the file; suptitles/titles/legends only.
- Hero-1 layer contract (plan §6): map-arm point + 95% CI whisker (clamped offsets via `_err_offsets`, gotchas xerr rule), iddelta ×, split-half ceiling |, null 95% band, 7B parent-ref hollow-D, calibration panel with CI + baseline, ceiling-adjusted separation panel — present (:731–817).

### Tests + lint (run by me)
`uv run pytest tests/test_issue2587_figures.py -q` → **24 passed** in 136 s (rc=0; thread-capped). Real-render bodies (no mocks), CLI e2e via `main(argv)`, inverted-CI clamp through the real errorbar call, fail-loud paths, registry pins. `ruff check` + `ruff format --check` on both files: clean.

### Concerns (all minor, non-blocking)
1. **One-color-one-meaning self-violation in `fig_floors_per_layer`** — scripts/issue2587_figures.py:588 (`colors = paper_palette(len(floor_names) + 1)`): neurips role hexes ARE Wong palette indexes (paper_plots.py:260–267 — baseline=_PALETTE[1], control=[2], accent=[5]), so the 5 floors get Wong orange/green/…/vermillion — orange = the 7B color set-wide. Failure scenario: hero-2 (orange 7B points) and floors-per-layer (orange dashed identity+bias floor) share the same depth-vs-R² axes shape; a reader skimming the set reads the orange floor dashes as 7B data. Also contradicts the module's own declared contract (":54–57 … neutral = reference lines / floors"). Fix: non-role palette indexes (3,4,6,7) + a grey, or linestyle-differentiated neutrals; keep colors[0] (primary) for the ridge map.
2. **Hero-1 panel 3 has no emptiness guard** — :697 guards `direction_cos` only; :713 tolerant-gets `obs_separation_snr`. Failure scenario: a crossmodel doc missing/empty on `obs_separation_snr` renders hero-1's third panel BLANK while the figure passes the PNG-size floor (the delta forest already guards both stats at :852). One-line fix: add `"obs_separation_snr"` to the :697 call (or an explicit skip + raise).
3. **`DISPLAY.get(x, x)` fallbacks can leak a future internal slug onto a rendered surface** — :540, :602, :936. All CURRENTLY realized keys are mapped (verified), so this is latent only; a KeyError-hard `DISPLAY[...]` would pin the discipline. Cosmetic.

### Plan adherence / scope
Diff implements exactly the R2c unit (registry 9→24 + hero-1 layer contract + r1 minors); no smoke-conditional branches added (grep clean — Step 0.71 N/A); no new third-party imports; dotenv-before-numpy/matplotlib ordering intact (:76–91); argcheck `--import-check` convention present (:1890–1895); per-fig progress lines + `[phase=done]` terminal (:1924–1925); no secrets, no network, no task-state mutation.

**Recommendation:** accept for the round; fold concerns 1–2 into the next touch of `issue2587_figures.py` (both are ≤5-line fixes + one 3-line pytest each); concern 3 optional.

Blockers: 0. Concerns: 3 (minor).
<!-- /split-review r2 g3 -->


# ===== [g4 fa2f530221] =====

<!-- split-review r2 g4 fa2f530221 -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `fa2f5302215509865854bb9e66e739a467c608d7` only (R2d1 1/3: NEW `smoke_registry.py`; 7-entry `SMOKE_BLIND_SPOTS` + `production_gate()` in `issue2587_analysis.py`; 2-entry registry + path-normalized eval_results guards in `issue2587_judge.py`; battery_run sentinel-fp/completion-predicate + parity admission floors; bank2587 docstring; 3 test files). `CONTRACT-BEARING: no` ⇒ round gates 0.5/0.55/0.6/0.8/0.9 SKIPPED per brief. Plan v3 asserted (`readlink tasks/running/2587/plans/plan.md` → `v3.md` at main checkout). No later round commit touches these files (`git log fa2f530221..HEAD -- <files>` empty), so HEAD-state tests exercise exactly this commit's blobs.

**Tier:** trunk (`src/explore_persona_space/experiments/issue2587/` touched; whole diff reviewed at trunk depth — every line read).

### Round-1 concern `analysis-smoke-blindspots` (BLOCKER) — VERIFIED FIXED

1. **All r1-cited sites route through `production_gate()`.** The five r1-cited downgrades are converted: `expected_contexts`/`expected_pairs` (old `:1127-1129` → gates at blob `:1301/:1307`), `axis_completeness` (old `:1617-1620` → `:1822`), `carrier_count_12` (old `:2894-2895` → `:3145`), `bootstrap_null_b` (old `:609-610` → registered param-narrowed + explicit `[smoke-blind-spot]` warning at `build_config`, fires only when actually narrowed). The commit additionally converted TWO `... or cfg.smoke` downgrades r1 did not cite: `bank9_cardinality` (`:3103`) and `parent_axes_count` (`:2650`) — coverage exceeds the r1 list.
2. **No smoke-conditional site remains outside the registry.** Independent sweep of the blob at this commit: zero residual `if not smoke` / `if not cfg.smoke` / `or cfg.smoke` / `or smoke` forms in `issue2587_analysis.py`; the surviving smoke reads are UPGRADES (smoke out-root rebind `:718/:727`, REQUIRED `--manip-*` under smoke `:729-731`) or the registered B narrowing (`:749-751`). Judge sweep: the only downgrades are the registered `call_arithmetic_1464` skip (`:815`) and the registered slice/cap narrowing (`:800-803` + `SMOKE_JUDGE_ITEMS`); the `:780-796` out/work rebinds + eval_results refusals are upgrades, correctly unlisted.
3. **Drift protection (analysis side) is mechanical:** `production_gate` RAISES on an unregistered site in BOTH modes (test-pinned incl. the param-narrowed-is-not-a-skip-licence case), and `test_every_assert_skipped_site_gated_in_source_and_vice_versa` pins source↔registry set equality by regex scan — a new skip cannot land without a registry entry, a stale entry cannot outlive its site.
4. **Enumeration is durable + extractable:** both `--list-smoke-blind-spots` CLIs run live (7 and 2 entries, verified by execution); smoke artifacts stamp `meta.smoke_blind_spots` (judge side e2e-test-pinned); the v2 implementation marker's `## Smoke run → ### analysis` block records the real-subprocess `--smoke` run with "all six registered assert-skipped sites fired in the child log". Registry prose cardinalities cross-checked: 1,080/2,874 == `B87.N_CONTEXTS/N_PAIRS` (bank2587.py:294-295); 1,392+72=1,464 == judge `EXPECTED_*` constants.
5. **Semantics preserved:** each conversion is production-equivalent to the inline assert it replaced (`assert cond, (site, detail)`); under smoke the skip now logs instead of silently passing.

**Step 0.71:** PASS — 9 smoke-conditional downgrades in the diff (7 analysis + 2 judge), all enumerated in the in-code registries + marker Smoke-run block; no unenumerated (a)-substitution or (b)-downgrade branch remains.

### r1 minors — verified

- **[g3] upload out of cell-grain fp:** `_regime_fp` drops `upload`; gen/capture sentinels re-add it in `extra`; `test_regime_fp_sensitivity` pins both directions (cell fp invariant, sentinel fp sensitive) and `test_phase_gen_upload_flip_none_to_hf_uploads_without_regenerating` pins the flip e2e (0 generate calls, 1 upload call, manifests byte-untouched, sentinel rewritten at hf fp). Traced `phase_gen`: pending empty + sentinel mismatch → skips pilot + cells, runs the upload leg, rewrites sentinel — correct.
- **[g4 C1] parity admission floors:** `n_anchors` int-typed (bool excluded) ≥ 10, `cos_min_bar` numeric ≥ `PARITY_COS_MIN`, absent-field refusal — all 4 refusal branches + 2 admit branches test-pinned. Verified the REAL probe report writes both keys (`run_engine_parity_probe` `:1754-1755`) and did so PRE-fix too (parent blob `:1716-1717`), so no legacy real report is stranded; a weakened `--parity-n-anchors`/`--parity-cos-min` probe run is refused at consumption as intended.
- **[g6 M1] path-normalized eval_results guards:** `_inside_eval_results` uses resolved `.parts` membership — absolute, relative, and `../`-spelled forms all covered (test-pinned incl. the absolute bypass that motivated it); applied to BOTH `--smoke` and the new `--dry-run` explicit-out refusal. Deliberately conservative (any `eval_results` component refuses).
- **[g1] bank2587 docstring:** names the dropped parent default-system-injection render probe with the #2329 rationale + the disclosed residual — documentation-only, plan-adherent.

### Tests + lint

- `uv run pytest tests/test_issue2587_{analysis,judge,battery_run}.py` → **115 passed in 52s** (44+34+37, matching the marker's claims).
- `ruff check` + `ruff format --check` clean on all 8 touched files.
- Both `--list-smoke-blind-spots` CLIs executed: exit 0, valid JSON, expected site sets.

### Concerns (Minor — none blocking)

1. `src/explore_persona_space/experiments/issue2587/smoke_registry.py:9-11` vs `scripts/issue2587_judge.py:815` — the module docstring's contract ("the skip sites must route through a gate helper that REFUSES an unregistered site") is implemented only on the analysis side; the judge's `call_arithmetic_1464` skip is a raw `if not smoke:` branch with no refusing helper and no source↔registry set-equality test. Failure scenario: a future round adds a second judge smoke skip and nothing mechanically trips — the registry silently understates the blind spots (the exact drift class the registry exists to prevent). Suggested fix-at-leisure: mirror the analysis regex test (scan for `if not smoke` downgrades) or thread the skip through a shared helper.
2. `scripts/issue2587_battery_run.py:493-500` — the `_regime_fp` docstring's "an --upload none -> hf flip … never regenerate them" is realized for GEN only: on CAPTURE the same flip re-runs the full GPU wave (`_capture_cell_complete` requires `m["uploaded"]` under hf at `:1147-1148`; `_capture_cell` has no upload-only leg for existing local `.pt` stores). Same behavior as pre-fix and conservative (redo, never skip) — doc accuracy + residual cost footgun, not a correctness bug.
3. `tests/test_issue2587_analysis.py` `test_smoke_config_narrows_b_and_meta_discloses` — the name promises a meta-disclosure assertion the body never makes (it tests only B narrowing/override); the analysis-side `meta_common["smoke_blind_spots"]` stamp is exercised only by the smoke driver, not unit-pinned (judge side IS e2e-pinned). Failure scenario: the analysis meta stamp regresses and only the pod smoke log would show it.

### Observations (not findings)

- `meta.smoke_blind_spots` lists the full registry under `--smoke` even when `--b-boot/--b-null` are overridden to production values — over-disclosure in the safe direction, and the entry text ("default … CLI-overridable") stays accurate.
- `production_gate` uses `assert` (stripped under `python -O`) — identical semantics to the inline asserts it replaces; project-wide pattern, not a regression.

### Security

No secrets, no injection surfaces, no unsafe eval/exec; new file writes go through existing atomic helpers; `--list-smoke-blind-spots` prints module constants only. Clean.

**Blocker tags:** none

**Recommendation: PASS** — the round-1 blocker is fixed by a mechanism stronger than the prescribed marker-prose fix (refusing gate helper + set-equality test + CLI + artifact disclosure); 3 Minor concerns above are fix-at-leisure.
<!-- /split-review r2 g4 -->


# ===== [g5 49a6c55eb8] =====

# Split-review r2 g5 — commit 49a6c55eb8 (leak/cap-hit harvest phase)

Verdict: CONCERNS

Scope: `git show 49a6c55eb8` — scripts/issue2587_pod_workload.sh (+72/−4), tests/test_issue2587_map_gen_capture.py (+30). CONTRACT-BEARING: no (gates 0.5/0.55/0.6/0.8/0.9 skipped per brief). Plan symlink verified `v3.md` == brief's plan_version=v3. Files unchanged between 49a6c55eb8 and worktree HEAD, so worktree test runs certify the commit state.

## Round-2 concern `leak-caphit-manifests-not-in-harvest-set` — VERIFIED FIXED

- Production: new `leak_caphit_harvest` phase (workload lines 480–527) copies the battery gen done-manifests from `$BATTERY_ROOT/{,shard*/}manifests/anchors_*.done.json` and runs the CPU-only `--aggregate-cap-hit` leg per split (6×, `run_logged` + `assert_file` per output) into `eval_results/issue_2587/leak_caphit/`. The aggregate leg (issue2587_map_gen_capture.py:2600–2670) is genuinely CPU-only (HF `_remote_index` + `_hub_download` + JSON reduce; no torch/vLLM), honors `--cap-hit-out`, and writes atomically.
- Harvest → committed set: `RESULT_JSONS+=("${f#"$REPO_ROOT/"}")` (line 526) over `find "$LEAK_DIR" -maxdepth 1 -type f -name '*.json' | sort`, BEFORE `phase results_push`; results_push then asserts existence per file, `git add/commit -- <paths>`, push with rev-list==0 + per-file `ls-tree` remote verification (the #1205 contract), and the HF mirror `upload_folder` carries the same array. Basenames in the flat HF mirror are collision-free by construction (LEAK_DIR is itself flat + guarded; no overlap with the 4 static names).
- Consumer reach: `issue2587_figures.py:1873` default `--leak-caphit-dir eval_results/issue_2587` with recursive rglob (`_load_leak_dir`, lines 255–257) matches both harvested classes (`anchors_*.done.json`, `cap_hit_*.json`); `think_leak_cap_hit_table` unit extraction (`anchors_<cell>.done.json` slice, `doc["split"]` for aggregates) matches the harvested names. Fail-loud on empty dir at BOTH ends (workload exit 6 on empty glob; figures RuntimeError on zero matches).
- Namespace hygiene: the P1 tiny-battery smoke writes its manifests under `$OUT_ROOT/p1_battery` — a sibling of `$BATTERY_ROOT`, not matched by either harvest glob — so smoke manifests cannot contaminate the harvested production set.
- Collision guard is a true invariant within one run: battery shards own disjoint axes (issue2587_battery_run.py:1932 `assign[a] == args.shard_index`), so same-basename-different-content across shard dirs is a genuine fault; exit 6 does not collide with the script's other exit codes (3 = assert_file, 5 = results_push).
- Plan adherence: plan v3 §4.3 (line 88) and §4.4 (line 96) mandate "per-split cap-hit + think-leak fractions reported"; §13 (line 199) lists the think-leak + cap-hit tables. This phase supplies exactly those inputs; nothing beyond scope.
- Tests: 46/46 pass (run locally, 8.2 s). `launcher_dryrun` genuinely EXECUTES the script under bash (subprocess, DRYRUN env), so the new test pins realized control flow: phase order (leak_caphit_harvest between p8 and results_push), exactly one redirected `--aggregate-cap-hit` command per `SPLIT_TO_MANIFEST` key with the right `--cap-hit-out` name, the manifest-copy echo, and the static RESULT_JSONS extension. The redirect-exemption addition is scoped to the single `cp` echo line; the aggregate lines remain redirect-asserted (and are additionally asserted `" > "` in the new test).

## Concerns (non-blocking)

1. **Collision guard is a relaunch trap: any fresh-pod re-run after a prior successful push dies exit 6 with a misdiagnosis** — scripts/issue2587_pod_workload.sh:506–509. The guard fires on ANY pre-existing `$dest` with differing bytes, but `$dest` lives in the git-committed `eval_results/issue_2587/leak_caphit/`, which results_push pushes to the issue branch. A later full relaunch (regen/fix round on a fresh pod) clones those old copies, and the regenerated manifests differ byte-wise with certainty — the manifest embeds `"repro": _repro(...)` (issue2587_battery_run.py:453+, `git_provenance()` sha), and the branch tip ADVANCED by the prior results_push commit itself, plus stochastic `cap_hit_frac` at temp 1.0. Failure scenario: relaunch completes all GPU phases, then exits 6 at harvest with "shards must own disjoint axes" (the wrong diagnosis — it is stale prior-round output, not a shard fault), and every re-run reproduces the exit until someone manually deletes the committed copies pod-side; the new results never reach results_push. Same-pod re-run IS idempotent (gen resume-skips on `regime_fp` match → identical bytes → `cmp -s` passes; aggregates overwrite atomically) except one edge: a crash mid-`cp` (line 511, non-atomic) leaves a truncated `$dest` that trips the same guard on the recovery re-run. Suggested shape: track basenames copied WITHIN this run (assoc array) and reserve exit 6 for within-run duplicates; unconditionally overwrite a pre-existing dest from the clone (results_push then commits the supersession as an ordinary diff). Non-blocking because: fail-loud (never silent corruption), pre-push, first production run unaffected (nothing is committed under eval_results/issue_2587/ today), and it fires only on relaunch-after-full-success.
2. **Stale results_push commit message** — scripts/issue2587_pod_workload.sh:548 still enumerates only "(split_ids, compat report, layer sweep, matched7b anchor)" though the committed set now includes the leak_caphit files. Cosmetic; misleads branch archaeology only.
3. **DRYRUN output becomes checkout-dependent after results land** — lines 522–527 (`if [ -d "$LEAK_DIR" ]` + find) run in dry-run mode too; once leak_caphit/*.json are committed, a VM dry-run appends them to RESULT_JSONS and the `[dryrun] results_push:` line content changes. Current tests remain green (assertions don't key on that line's file list); noting so a future test doesn't pin it.

## Recommendation

Accept for this round; fold concern 1's guard-scoping into the next touch of the workload script (it only bites at the first regen round).


# ===== [g6 1962fd73cb] =====

# Split-review sub-scope verdict — issue 2587, round 2, group g6

Commit: 1962fd73cb5fa35d404e6a9dd6b6c37d123eaf6a (NEW `scripts/issue2587_smoke_run.py`, 534 lines)
Reviewer: code-reviewer-lean (split-review sub-scope; CONTRACT-BEARING: no — gates 0.5/0.55/0.6/0.8/0.9 skipped per brief)

## Verdict: PASS

## Round-1 concern `smoke-run-coverage` (BLOCKER): VERIFIED FIXED

1. **Analysis leg exercises the PRODUCTION entrypoint, zero fakes.** `run_analysis` (issue2587_smoke_run.py:291) launches `scripts/issue2587_analysis.py --smoke` as a real `subprocess.run` with a real rc; the driver contains no monkeypatching on this leg (it only writes fixture files + spawns). Real dims confirmed against the entrypoint's own constants (`H_9B=4096`/`H_7B=3584`/`TWIN_LAYERS_9B=(16,22,30)`/`LAYERS_7B=(14,19,26)`, issue2587_analysis.py:138-142); every CLI flag in the argv tail exists in the analysis parser (:635-667). "No network path reachable" verified on the REALIZED runs: 0 `[an] staging` lines in the child log (the only two hub call sites, issue2587_analysis.py:800/:3173, are gated behind the local overrides the driver passes). All six registered assert-skipped sites fire in the child log + `bootstrap_null_b` param-narrow, and the artifact's `meta.smoke_blind_spots` disclosure matches the script's own registry exactly (6 assert-skipped + 1 param-narrowed — the complete current registry, issue2587_analysis.py:168-240).
2. **Judge leg's single fake sits at the true boundary and cannot be bypassed.** `_judge_child` (:414) patches `batch_judge.judge_completions_batch` as a MODULE attribute before `runpy.run_path(..., run_name="__main__")`; the production caller references it by module-attribute lookup at call time (`from ...eval import batch_judge as _batch_judge`, graded_judge.py:35; call at :336), so the patch binds regardless of import order, and `issue2587_judge.py` contains no direct `judge_completions_batch` import that could route around it. The child's exit code IS the judge entrypoint's SystemExit (runpy propagates it through `main()`'s `sys.exit`). `create_autospec(real, side_effect=fake)` pins the boundary signature (all-keyword call site → `fake(**kw)` binds); the fake asserts the three plan invariants (`claude-sonnet-4-5-20250929` / `max_tokens=1024` / `threshold_base=0` — a missing `threshold_base` would KeyError, fail-loud) and writes a `save_raw` payload consumed by the REAL `judge_result_from_save_raw` reduce (graded_judge.py:352ff), so judge_graded's real body (packing, custom_id guard, passthrough) plus the real drop-class reduce both run. Anchors come from the real sha-pinned bank module (`bank2587.PIN = 8265bcd…`, bank2587.py:106) via the production `judged_specs`/`lang_specs` on the production smoke slice.
3. **Claimed artifacts and digests reproduce.** (a) Round-execution artifacts at `/tmp/issue-2587-smoke/r2d1/` match the marker v2 sha256 claims EXACTLY (`minpair_delta_2587.json` = de82b8c2…, `manipulation_check_2587.json` = 87976cdf…, verified by `sha256sum`). (b) Fresh re-run this review (`issue2587_smoke_run.py all --out-root /tmp/issue-2587-smoke/review-r2-g6`, thread-capped): rc=0 both legs; realized counts match the marker (perpair 106 rows, 3 ckpts, 7 npz; 7 value rows, 42 scored rows, exactly 2 boundary calls — both rubric families — each with the three invariants pinned in `boundary_calls.jsonl`). Sha divergence across out-roots is expected (analysis meta embeds out-root paths; judge meta embeds git state).
4. **Hygiene:** `ruff check` clean; `--import-check` rc=0 (argcheck-bind 0/0/0); post-commit delta to this file within the round is ONE lint-waiver comment (R2d2 cb613f7954), nothing behavioral; nothing production-side imports the driver; judge leg passes `--skip-upload` (issue2587_judge.py:747/:951) so the leg is fully offline.

## Blockers

None.

## Concerns (minor, non-blocking)

- **M1 — offline property is realized, not structurally asserted.** scripts/issue2587_smoke_run.py:305-309 — `resolve_rel`'s hub fall-through (issue2587_analysis.py:790-803) means a MISSING tiny-world fixture file silently attempts an HF staging fetch instead of failing as a fixture bug. Failure scenario: once production artifacts exist under the HF prefix, a fixture-path drift would let the "fully-local, zero fakes" smoke silently consume PRODUCTION artifacts from HF and still PASS. One-line hardening: assert `"[an] staging" not in combined` in `run_analysis` (today the realized runs show 0 staging lines, and pre-production a fall-through 404s loud).
- **M2 — hardcoded skip-site tuple can lag the registry.** scripts/issue2587_smoke_run.py:70-77 + the subset check at :316 (`set(ANALYSIS_SKIP_SITES) <= set(bs)`): a FUTURE site registered in `SMOKE_BLIND_SPOTS` would fire unasserted. Deriving the tuple from `AN.SMOKE_BLIND_SPOTS` (kind == "assert-skipped") would be self-updating. Currently exact (verified against the registry).
- **M3 — boundary fake re-derives the custom_id grammar.** scripts/issue2587_smoke_run.py:441-448 hardcodes `__00000__{ci:02d}` (question idx pinned to 0). Exact for `judge_graded`'s single-question-per-item packing, and reduce-side grammar changes ARE caught (the real reduce consumes the fake's keys), but a PRODUCER-side grammar change in `batch_judge._enumerate_and_check_cache` (batch_judge.py:666 — inline f-string, no shared helper to import) would not be. Inherent to faking at this seam; noted for the record.
- **M4 — commit-message prose nit.** The message says the parent asserts "42 scored rows"; the code asserts `n_scored > 0` (scripts/issue2587_smoke_run.py:404) — 42 is the realized count (reproduced in my rerun and recorded in the marker), not an assertion. No code change needed.

## Test/verification runs (this review)

- `issue2587_smoke_run.py all` (fresh out-root) → rc=0, both legs PASS, counts match marker.
- `sha256sum` on the round-execution artifacts → both digests match marker v2 verbatim.
- Child-log grep → all 6 assert-skipped sites + `bootstrap_null_b` fired; 0 hub-staging lines.
- `ruff check scripts/issue2587_smoke_run.py` → clean; `--import-check` → rc=0.


# ===== [g7 cb613f7954] =====

# Split-review r2 g7 — commit cb613f7954 (R2d2) + round-level contract gates

**Verdict: PASS**

**Tier:** leaf (two `scripts/issue2587_*` entrypoint edits; reviewed at contract-bearing depth).
**Diff read:** full commit body (`git show cb613f7954`, 2,211 bytes; both hunks read with surrounding context).
**Prior-concerns ledger:** empty (`list-concerns 2587 --open-only` → 0 open).
**Main-side divergence:** brief `diverged_on_main: none` (probe r2 count=0 at main=93ffc35f530e) — nothing to re-derive.

## Commit review (g7 scope)

1. `scripts/issue2587_fits.py:870` — `UnicodeDecodeError` added to the sentinel-read guard tuple. CORRECT: `sentinel.read_text(encoding="utf-8")` raises `UnicodeDecodeError` (a `ValueError` subclass, NOT under `OSError`/`json.JSONDecodeError`), so a byte-corrupt sentinel previously crashed outside the caught names; it now routes to `sdoc = {}` → `sentinel_ok=False` → the `_matched7b_repair` idempotent-repair path that REWRITES the sentinel (fail-forward repair, not a silent swallow — consistent with the r1 `matched7b-resume-contract` design read at fits.py:851–876). Guard placement is right: the decode happens in `read_text`, inside the try. Sibling sweep discharged mechanically: `workflow_lint.py --check-json-guard-unicode` re-run by this reviewer → PASS (tree-wide).
2. `scripts/issue2587_smoke_run.py:176–177` — `# PROD_IMPORT_LINT_EXEMPT: <reason>` waiver comment above `import test_issue2587_analysis as TW`. CORRECT and recognized: verified against the recognizer (`workflow_lint.py:9689 _prod_import_lint_waiver_present` — same physical line OR immediately-preceding non-blank line; reason ≥ 10 chars; exempts that site only). The comment is the immediately preceding non-blank line; reason is 60+ chars. Waiver is legitimate: the module is an in-repo `tests/` module (not a third-party dist — lockfile-unresolvable is the check's known false-positive class here), and runtime resolution is guaranteed by the module-top `sys.path.insert` of `REPO_ROOT/tests` (smoke_run.py:56–59); the scripts→tests coupling is declared in the module docstring. `--check-prod-import-lockfile` re-run by this reviewer → PASS rc=0.
3. Commit-message claims certified at tip: both targeted lint checks PASS (re-run), `tests/test_issue2587_fits.py` 40 passed (re-run, 22.0s), `ruff check` on both files clean (re-run). Step 0.71: N/A — the diff adds/edits no smoke-conditional branch (a comment line + an exception-tuple broadening only).

## Round-level contract gates (CONTRACT-BEARING)

- **Gate 0.5 (implementer marker):** PASS. `epm:experiment-implementation v2` present in canonical events.jsonl (ts 2026-08-26T14:11:48Z, head sentinel `v2`); all four H3 sections (a)–(d) present and in order; (c) carries copy-pasteable commands (`git diff b043c0ccea..cb613f7954`, per-commit `git show <sha>`) + observable success signals (259/0 suites at 1962fd73, fits 40/40 at tip). Cosmetic only (orchestrator-composed after implementer death — marker-shape class, fixable by orchestrator, per brief): `## Smoke run` interleaved between (c) and (d); (c) commands are inline-backticked rather than fenced blocks.
- **Gate 0.55 (smoke-architecture marker):** PASS. v1+v2 rows present with parseable `verdict: PASS_PARTIAL arms_stubbed=<12 arms>`; `task.py check-smoke-arch-registry 2587 --repo-root <worktree>` → OK (registry N/A form adjudicated against per-driver registries recomputed at tip 1962fd73cb); v2 per-arm FALLBACK row set (12: analysis.main, battery_run.{gen,capture,embed}, fits.{fits,finalize,matched7b}, judge.main, map_gen_capture.{coresident,phase_split_gen,phase_split_capture}, pod_workload.main) set-equals `arms_stubbed` (resume-matrix / production-outroot FALLBACK sub-block rows correctly excluded from the per-arm set).
- **Gate 0.6 (end-to-end smoke):** PASS across v1+v2. v1: figures (command + rc=0 + 26-file digest + all-8-PNG read-back), fits CPU smoke-chunk leg (command + rc=0 + per_layer digest), battery + map_gen_capture labeled GPU-bound carve-outs with all three substitute items, judge API-bound carve-out. v2 SUPERSEDES the two weakest v1 entries with production-entrypoint legs: analysis (`issue2587_analysis.py --smoke` real subprocess, rc=0, sha256-digested artifacts, all six registered blind-spot sites fired) and judge (`issue2587_judge.py --smoke --skip-upload`, boundary-only autospec fake, rc=0, sha256 digest, 42 scored rows). Pod-only legs (pod_workload waves, HF uploads, results push) disclosed in (b) + rowed FALLBACK in the smoke-arch marker + covered by the R2a launcher dry-run structural suite (map_gen_capture 45/45 incl. dry-run suite, compose_p1 pos+5 neg, apply-probe pos+4 neg — v33 durable record). Composition note (cosmetic): the new `pod_workload.sh` entrypoint has no labeled `### — Carve-out` sub-section inside the v2 `## Smoke run`; its carve-out evidence lives in (b) + v33 + the smoke-arch row.
- **Gate 0.8 (prior concerns closure):** PASS. Open ledger empty; all 8 round-1 concern ids carry `epm:concern-addressed` rows (missing-pod-workload, compat-gate-not-enforced, matched7b-resume-contract, missing-fire-defaults-true, plan-s6-figures-deliverable-gap, analysis-smoke-blindspots, smoke-run-coverage, leak-caphit-manifests-not-in-harvest-set). All 5 BLOCKERs spot-verified against the actual tree/diffs: (1) missing-pod-workload — `scripts/issue2587_pod_workload.sh` exists with the full §10 phase chain (6b614a4747); (2) compat-gate-not-enforced — `require_p1` re-asserts the P1 compat sentinel before EVERY production wave (pod_workload.sh:356/379/402/423/440/464) + `--gate compose_p1` / `--p1-apply-probe` legs; (3) matched7b-resume-contract — `_matched7b_completion_gaps` keys the skip on complete+regime_key+requested-upload-contract+sentinel, else `_matched7b_repair` (fits.py:851–905, read directly); (4) analysis-smoke-blindspots — `smoke_registry.py` present, `SMOKE_BLIND_SPOTS`+`production_gate()` wired in analysis.py (lines 163–259, `--list-smoke-blind-spots` at 663), sites attested FIRING through the production entrypoint in the v2 smoke leg; (5) smoke-run-coverage — `scripts/issue2587_smoke_run.py` present, both legs executed rc=0 with sha256 digests.
- **Gate 0.9 (git provenance):** PASS. All 7 round payload commits (b42275d162 → cb613f7954) present on `origin/issue-2587`; the two post-round spec-sync commits 707c825e1b + 972db230c6 verified blob-identical to fetched origin/main on all 53 changed paths (0 DIFFERS / 0 ABSENT).
- **Smoke-output hygiene:** PASS. Round diff b043c0ccea..cb613f7954 touches zero `eval_results/` / `figures/` paths; smoke legs write under `/tmp/issue-2587-smoke/`; worktree `git status --porcelain -- eval_results/ figures/` empty (also after this reviewer's own test runs).

## Concerns (non-blocking; NOT persisted — brief forbids task-state mutation; orchestrator may persist if it wants them binding)

- **[gate 0.6/(c) residual] no-flags-gate-result-unrecorded** — v2 marker (c) states the tip-level full no-flags `workflow_lint` re-run was INCONCLUSIVE (570s timeout), relaunched with a 40-min bound, "result will be recorded in a follow-up `epm:progress` note"; no such note exists (v37–v41 are watcher/probe/dispatch rows only — the predecessor session died). Failure scenario: a non-targeted check in the no-flags bundle fails on round payload at tip and is first discovered at the Step 10d landing gate. Mitigation already in hand: the predecessor's COMPLETED pre-R2d2 no-flags run surfaced exactly the 2 findings this commit fixes; this reviewer independently re-ran both targeted checks (PASS/PASS), fits 40/40, ruff clean; Step 10d re-runs the full no-flags instrument before landing. Low residual risk — CONCERN, not blocker.

**Blockers:** 0. **Concerns:** 1 (above). **Recommendation:** PASS — the R2d2 commit is a correct, minimal, independently re-verified pair of lint-conformance fixes, and every round-level contract gate passes.


# ===== [g8 707c825e1b..972db230c6] =====

## Split-review sub-verdict — issue 2587, round 2, group g8

**Verdict: PASS**

**Scope:** identity verification of the two orchestrator-authored spec-freshness sync commits 707c825e1b + 972db230c6 (NOT implementer payload). CONTRACT-BEARING: no — gates 0.5/0.55/0.6/0.8/0.9 skipped per brief. Diff sized first: 270,280 bytes; body not read (identity probes only, per brief).

**Tier note:** touched paths include `.claude/**` and shared `scripts/` (trunk patterns), but review mode here is identity-verification per the SPLIT-REVIEW brief, not content review.

### Findings

1. **Byte-identity to fetched origin/main — PASS (53/53 paths).**
   `git diff --name-only origin/main 972db230c6 -- <all 53 paths>` is EMPTY, and the working-tree form `git diff --name-only origin/main -- <paths>` is also EMPTY (972db230c6 is the current branch tip, so the two probes coincide). Every path the group touched is byte-identical to origin/main at its current tip 083f92e7aa. Failure scenario averted: a partial/hand-edited import would have shown a residual per-path diff — none exists.
   Note: origin/main has ADVANCED past the brief's divergence-probe pin (93ffc35f530e → 083f92e7aa) and the 53 paths are identical even against the NEWER main — the identity claim holds against both refs.

2. **No issue-2587 payload files touched — PASS (0 hits).**
   `grep -cE 'issue2587|issue_2587'` over the 53-path name-status list = 0; no `scripts/issue2587_*`, no `tests/test_issue2587_*`, no `src/explore_persona_space/experiments/issue2587/**` (no `src/` paths at all). Status distribution 23 A / 30 M, zero deletions. Failure scenario averted: a sync commit sweeping round payload would hide implementer changes from the payload groups' review — none swept.

3. **Commit subjects carry the sync anchor phrase — PASS (2/2).**
   - 707c825e1b: `issue-2587: sync workflow-surface specs from origin/main (spec-freshness)` — exact canonical Step 5a form (steps/09-step-5.md:496).
   - 972db230c6: `issue-2587: sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files + .gitleaksignore)` — the sibling-files form (steps/09-step-5.md:596) extended with `+ .gitleaksignore`; the bare `spec-freshness` token the Step-5 subject-scoped exclusion keys on (steps/09-step-5.md:427) is present. Failure scenario averted: a missing anchor would make a later branch-dirtiness probe misread these imports as feature edits — both anchors present.

Ancestry: round parent b043c0ccea is an ancestor of 707c825e1b (`git merge-base --is-ancestor` rc=0); 707c825e1b's parent is cb613f7954 (R2d2), i.e. the sync commits are the last two of the round.

**Recommendation:** PASS — group g8 is a clean import of origin/main bytes, nothing else.
