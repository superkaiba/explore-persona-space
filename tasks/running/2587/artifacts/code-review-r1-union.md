split_review: groups=8 commits=8 trigger=commits|bytes contract_bearing=g8

<!-- epm:code-review v1 -->
# Code review round 1 — composed split-review verdict

**Verdict: FAIL** (composed mechanically: FAIL if any sub-verdict FAILs. The orchestrator never re-grades a sub-verdict.)

Round: 8 commits, 19 files (all additions), +16,453 lines, 710,767 diff bytes at merge base 6eb0d82b496414fdd7ae7d1fc0de41e889ae4636.
Split-review fired on BOTH triggers: T1 commits=8 > 4, and T2 bytes=710,767 > 100,000 with commits >= 2.
Claude side: 8 parallel per-commit `code-reviewer-lean` sub-reviews (G = min(m,8) = 8, one commit each).
Codex twin: deliberately UNSPLIT on the whole round — the designed catching arm for cross-commit interaction bugs.

## Per-group verdict table

| group | commit | scope | verdict |
|---|---|---|---|
| g1 | `03b8b8fe5a` | bank2587 + issue2587_common | PASS |
| g2 | `ec2cb6ce47` | map_gen_capture (#2330 fork) | PASS |
| g3 | `f3227b2234` | battery_run P5 generation | PASS |
| g4 | `0dfe39011b` | battery capture P6 + embed | PASS |
| g5 | `c47a490211` | fits P4 + both §4.5 7B arms | PASS |
| g6 | `3635489766` | judge P9 | PASS |
| g7 | `891d266f7c` | analysis P10 + lattices | PASS |
| g8 | `b043c0ccea` | figures + prefix fix + ROUND GATES | **FAIL** |

Contract-bearing group: **g8** — the only group that ran round-level gates 0.5/0.55/0.6/0.8/0.9. The other seven explicitly skipped them per their sub-briefs, which is why a round-level omission could pass seven per-commit reviews and still be caught.

No mechanical strip applies: g8's blocker tags include `smoke-blind-spot-unenumerated` and `substantive`, so this is not a mechanical-contract-only FAIL.

## Unioned findings, verbatim per group

### [g1 03b8b8fe5a] — sub-verdict PASS

<!-- epm:code-review v1 group=1/8 commit=03b8b8fe5a0319f4d37a53269b09ead1f0f6a42a -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `03b8b8fe5a0319f4d37a53269b09ead1f0f6a42a` only (unit 1: `bank2587.py` + pinned values JSON + `issue2587_common.py` + 2 test files; 1,769 insertions). Round gates 0.5/0.55/0.6/0.8/0.9 skipped per brief (`CONTRACT-BEARING: no`; group 8 runs them).
**Tier:** trunk (touches `src/explore_persona_space/`).
**Plan:** v3 confirmed (`readlink plans/plan.md` = `v3.md`); reviewed against §4.1/§4.2/§4.4.

### Brief claims — all six verified by live probe

1. **By-import reuse (§4.1): VERIFIED.** `build_model_venv is issue2378_dispatch._build_model_venv`, `assert_driver_compat is _assert_driver_compat` (identity-tested, signatures `(logs_dir)` / `(compat_dir=None)` confirmed at source). All 7 pin constants re-exported from `issue2378_common` by attribute reference; ground-truth values match the plan (`vllm==0.27.1 / transformers==5.15.1 / torch==2.13.0`, `LAUNCH_ENV_PINS={"VLLM_USE_FLASHINFER_SAMPLER":"0"}`, `ENGINE_KWARG_PINS={"gdn_prefill_backend":"triton"}`, floor 580, compat dir). No divergent retyped copy: the only local literals are `THINK_SCAN_MAX_FRAC=0.01` (a #2330 convention, verified == `issue2330_qwen35_generate_capture.py:214`, not a #2378 pin) and the plan-named `VLLM_WORKER_MULTIPROC_METHOD=spawn` addition; test-file literals are deliberate plan pins.
2. **Transcription + pins: VERIFIED.** `sha256(git show 6894503746:scripts/issue2564_langow_pilot_run.py)` = `4cc78ad8…dda74` == `LANGOW_SHA256`. Both `PINNED_SHA256` entries reproduce from `git show 8265bcd:…` (`d581a2c5…`, `29e6bebc…`). Committed `bank2564_values_pinned.json` is byte-identical to the pinned blob (same sha256). The pin is ASSERTED at runtime: `_pinned_bytes` raises `RuntimeError` on digest mismatch on EVERY load path (committed copy and git-show alike), and tests re-derive all three digests from git ground truth (`test_pinned_blob_sha256_ground_truth`, `test_langow_source_sha256_pin`, `test_committed_values_copy_matches_pin`, tamper test `test_pinned_bytes_sha_drift_fails`). `LANG_VALUES` (3 strings, order included) and `ONEWORD_PAIRS` (24 tuples) compared verbatim against the pinned source — identical; the `_git_show`/`_import_pinned` machinery is a faithful transcription with three disclosed adaptations (lazy `_bk()` instead of import-time extraction, unique module name `bank2564_pinned_2587`, module-local `_repo_root()`); pilot pair/context constructions are line-faithful, the added parent-schema `"cell"` key on pilot pairs is disclosed in the docstring.
3. **Totals: VERIFIED bottom-up.** Pinned parent constants: 984 contexts / 2,778 pairs, per-class {install 468, swap 864, famswap 864, instruction_paraphrase 468, query_content 66, query_form 36, query_paraphrase 12}. Pilot: 12×(1+3)+48=96 contexts, 36+36+24=96 pairs. Merged 1,080 / 2,874 with the exact per-class table; `gate_merged_complete` also asserts pair-id uniqueness, and the parent/pilot context-id collision check is explicit.
4. **Gate severity split: EXACT match, no silent upgrade/downgrade.** (i) fail-loud (`Bank2587GateError`, parent portion via pinned `gate_grid_complete`); (ii) fail-loud both portions; (iii) recorded + reported bools only — proven non-raising by the fixture whose fake tokenizer yields unequal within-axis counts and still PASSes; (iv) recorded only — `CharTok` many-token names do not raise, q25 pinned ids carried as reference; (v) asserted ≥ 1 via the pinned `attach_changed_tokens` (confirmed at the 8265bcd blob: `assert p["changed_tokens"] >= 1, (pair_id, "pair sides render identically")`), computed over FULL rendered-prompt ids; (vi) fail-loud (merged no-"assistant"-in-system over parent+pilot; parent `gate_form_triplets`); (vii) fail-loud (one `<|im_start|>assistant` header + closed-empty-think). Trip tests exercise i, ii, v, vi (both halves), vii (open / non-empty / double-header) — all 37 tests pass live (2.5 s).
5. **Think assert: single implementation, correct form.** `assert_closed_empty_think` lives once in `bank2587.py`; `issue2587_common` re-exports by identity (identity-tested). Form matches the plan-cited #2333 precedent lines exactly (verified against `issue2333_run.py:346-351`), including the closed-and-empty check with whitespace-only body; it is NOT a "no `<think>`" scan (a no-block render passes, unit-pinned).
6. **Env pins untouched: VERIFIED.** No `pyproject.toml`/`uv.lock` hunks in the commit (`--name-only` grep empty).

### Fail-fast audit

No `try/except: pass`, placeholders, or fault-swallowing defaults. `_git_show` first attempt is `check=False` but the retry path is `check=True` (fail-loud); the loader's `AutoModelForCausalLM → AutoModelForImageTextToText` fallback is the disclosed #2223 pattern ending in the loud 32-block/hook-path asserts; the tokenizer `local_files_only → network` fallback in `main()` is a benign cache-first convenience.

### Tests / lint

`uv run pytest tests/test_issue2587_bank.py tests/test_issue2587_common.py` → 37 passed. `ruff check` + `ruff format --check` clean on all 5 Python files. `workflow_lint --check-dotenv-before-hf-import` PASS (load_dotenv precedes any transformers import in both entry paths); `--check-prod-import-lockfile` PASS (WARNs are pre-existing, none name this commit's files). The tree-wide no-flags lint run timed out at 570 s in this session — group 8's round gate is the binding arm for it.

### Minor findings (none blocking)

- **M1 (Minor, disclosure gap):** the pinned parent's render-half of its gate (vi) — bare `rendered.count("assistant") == 1` plus the `"You are Qwen"` default-system-injection probe — is not carried; `gate_render_q35` counts only `<|im_start|>assistant` headers. This matches plan §4.4 (vii) verbatim (plan-adherent), but the module docstring's "Deliberately NOT inherited" list names only the q25 render-prefix assert and the paraphrase-ratio gate, not this narrowing. Residual: a q35 template that injects default system text on empty-system messages would pass gate (vii) silently, weakening "bare"-arm semantics. Suggest a later unit assert/record that each render embeds the stored system string (or a default-marker denylist) at P0b.
- **M2 (Observation):** `assert_closed_empty_think` tolerates a render with NO `<think>` block (asserts only when one is present) — faithful to the cited #2333 lines and unit-pinned as intended. A wrong-template render with no think block would pass gate (vii); the #2330 token-level suffix `template_pin` covers the map-fit wave, so the battery unit (later commit) should carry the suffix-level pin for its wave.
- **M3 (Nit):** `_bk()`'s `tempfile.mkdtemp` extraction dir is never removed (langow-precedent shape; a few KB per process).
- **M4 (Nit):** within-axis equality is reported for value strings only (paraphrase counts recorded but not equality-reported) — a defensible reading of plan (iii), matching the parent's gate-(i) scope.

**Recommendation:** PASS for this commit. M1 is the only item worth a follow-up line in a later unit; nothing requires a re-roll of this commit.
<!-- /epm:code-review -->

### [g2 ec2cb6ce47] — sub-verdict PASS

<!-- split-review r1 g2 — commit ec2cb6ce47da421053d4d48a92022e379af08277 -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `ec2cb6ce47da421053d4d48a92022e379af08277` only (`scripts/issue2587_map_gen_capture.py` 3,156 L + `tests/test_issue2587_map_gen_capture.py` 427 L, both new). `CONTRACT-BEARING: no` — Steps 0.5/0.55/0.6/0.8/0.9 explicitly SKIPPED (group 8 runs them).
**Tier:** leaf by pattern (one-off `scripts/` driver + test); reviewed at trunk depth given size and the downstream `split_ids.json` contract.
**Plan:** graded against `tasks/running/2587/plans/plan.md` → `v3.md` (symlink verified) §4.3/§7/§9/§12.

### Fork-provenance verification (mechanical)
- Declared parent blob `6725ae08734f6f6f40be76acf98f1e12093ec0f5` is byte-identical to the LIVE `scripts/issue2330_qwen35_generate_capture.py` at branch HEAD — the pin is current, not stale.
- Full realized delta computed (`diff parent-blob fork` ≈ 503 diff lines) and read IN FULL. Every hunk maps onto the 6-item FORK PROVENANCE list; residual hunks are comment/help-text retargets and the cap-hit selftest routing-table update (`expect_subpath`), which is forced by item 3's split-set change and consistency-asserted against `SPLIT_TO_MANIFEST` (`sorted(expect_subpath) == sorted(SPLIT_TO_MANIFEST)`). No understated divergence.

### Brief claims — all verified
1. **Gates kept, not weakened:** `P1_SENTINEL_REQUIRED == ("template_pin", "length_scan", "hook_probe")` exactly (fork:365-368); `emit_spans`/`parity7b`/`smoke_shard`/`fits_smoke` records still written when run but no longer sentinel-gate (disclosed, plan §4.3 kept-gates match).
2. **P0b length scan:** bootstrap-when-absent asserts `PINNED_MANIFEST_COUNTS` (25,000+400+1,000+999 = 27,399, matching plan §12 pre-scan grain at pin `815ff6d9…` — `MANIFEST_REVISION` verbatim) plus per-manifest duplicate-id assert; scan tokenizes via the REAL render path (`_rendered_prompt_token_len` → `_render_prompt`, so the closed-empty-`<think>` assert runs per scanned row) against `PROMPT_TOKEN_BUDGET == 7104` (module-level drift assert at fork:235). HALT: `frac > 0.005` returns 4, propagated through `main()` → `sys.exit`/`os._exit`; split_ids.json NOT rewritten on that branch (test pins it by `read_bytes()` compare); the `passed: false` run_meta record persists as audit trail and the sentinel cannot fire on it. Drop path: order-preserving filter, `dropped_overlength` extend, shas + counts recomputed, `_write_json_atomic`; idempotent re-run tested.
3. **Schema/ordering contract:** `issue2587_split_ids_v1` carries per-split ORDERED id lists in manifest order (bootstrap iterates manifest rows; `_subset_rows` returns id-list order); `_sha_ids` = sha256 over compact JSON (`separators=(",", ":")`) — order-sensitive; test pins the domain (`[3,1,2]` ≠ `[3, 1, 2]`) for the §4.5(b) matched-7B ordered-set-exact consumer.
4. **`--hf-prefix` default None:** parser default None (test-pinned, #1005 shape); run mode `assert args.hf_prefix` (fork:3120); aggregate mode `assert root` and uploads ONLY when `root == args.hf_prefix` (foreign banked roots never written). No path resolves a default upload prefix.
5. **`train_10k` fully removed:** zero functional references remain (grep: docstring/comment mentions only); argparse `--split` choices derive from `SPLIT_TO_MANIFEST`; cap-hit selftest table updated in lockstep. Removal is a disclosed, justified deviation from the plan's "add the 3-line entry" (an unresolvable always-crashing choice otherwise) — noted, not a defect.
6. **Unit-1 wiring by import — live-probed, not assumed:** all 5 consumed `cm2587` attributes exist (`LAUNCH_ENV_PINS`, `ENGINE_KWARG_PINS`, `THINK_SCAN_MAX_FRAC`, `think_leak_scan`, `assert_closed_empty_think`). `assert_closed_empty_think` verified NO-OP on a plain Qwen2.5 render, raises on open and non-empty `<think>` (live probe + tests). `think_leak_scan` (containment) replaces `_opens_with_think` at BOTH count sites; the parent function is deleted. `LAUNCH_ENV_PINS` setdefault at module top precedes every vllm import (all deferred: fork:877/929/3001). `ENGINE_KWARG_PINS` threaded into `_build_engine` and identity-tested against the `cm2378` source of truth (no duplicated-constant drift channel). Model-venv resolvability of the new `explore_persona_space` import chain checked end-to-end: package `__init__` empty, `orchestrate/__init__` → `fleet` stdlib-only at module top, `env.py` needs python-dotenv which `MODEL_VENV_EXTRA_PINS = ("python-dotenv==1.2.2",)` installs for exactly this path (comment names it).
7. **Generation pins:** `dtype="bfloat16"` in `_build_engine`; `GEN_TEMP=1.0` / `GEN_TOP_P=0.95` (byte-inherited from #2330); per-split seed via `SPLIT_TO_MANIFEST` (train_25k@42, ceiling draws 43/44); `enable_thinking=False` asserted at the fake-template boundary in tests; `--num-shards` default 2 (plan §4 P2 2-way sharding); driver writes `CUDA_VISIBLE_DEVICES` nowhere (0 env writes — launcher-pinned, matching the #2336/CVD discipline). Raw TEXT + capture `.pt` flush per `UPLOAD_BATCH` with exact-set verify + purge; `--no-upload` confined to the sanctioned smoke path.
8. **Capture:** teacher-forced `cx_last` (p_len−1) + `v_x` (response+tail mean) per layer with shape asserts; prompt-segment-ends-with-realized-header-suffix assert inherited intact (fork:706-719); `--capture-dtype float32` / `--layers 0-31` are launcher args per the docstring/plan §10.

### Fail-fast scan
No `try/except: pass`, placeholders, dummy-data-on-error, or fault-swallowing fallbacks in the delta. All exception handlers are inherited with documented predicates (transient-retry filter, typed-fallback with re-raise, best-effort engine reap AFTER rc is fixed on the `__main__` exception path, which still terminates via `os._exit(rc)` when an engine was constructed — the vLLM-port terminal discipline intact). `--fits-smoke` fail-louds on the missing unit-3 script (`assert fits_script.is_file()`).

### Tests + mechanical checks
- `uv run pytest tests/test_issue2587_map_gen_capture.py` — **21/21 PASS** (0.81 s). Coverage is genuine: all three gate branches (bootstrap-PASS, drop-recompute, HALT-4-no-mutation byte-compare), count-drift and duplicate-id bootstrap halts, sentinel written only on 3/3 PASS records, engine-pin threading incl. identity with `cm2378`, launch-pin setdefault at import, real `_render_prompt` under fake tokenizers, CLI defaults (hf_prefix None, 0.005 band, sentinel resolution in `main`).
- `ruff check` + `ruff format --check` in situ: clean on both files.
- Committed blobs byte-identical at branch HEAD (no post-commit drift within the round).
- Tree-wide no-flags `workflow_lint.py` (completed run, rc=1): **zero rows name either of this commit's files.** The 2 tree errors are attributed elsewhere: (a) `scripts/issue1901_mlpdense_fold_analysis.py:45` live-hf-retry-routing — pre-existing, the lint message itself flags snapshot staleness (#1568); (b) `scripts/issue2587_fits.py:202` process-shared atomic-write temp name (#2336) — a ROUND-committed file but a DIFFERENT commit, outside this group's scope. **Cross-group flag for the orchestrator/group owning the fits commit and group 8's round gates:** (b) is a real no-flags lint FAIL a round-committed file introduces and will block the Step 9c/inline payload gate if unfixed.

### Minor (non-blocking)
- **M1 (inherited ordering, not introduced here):** on the drop path, the `passed: true` run_meta record lands (fork:1599) BEFORE the mutated split_ids.json is written (fork:1629); a crash inside that two-write window leaves a passed record with drops unapplied, and a later gate run could then write the sentinel against un-mutated split_ids. The parent blob has the identical ordering (parent:1460→1499), so this is pre-existing #2330-reviewed behavior; flipping the order (mutate, then record) is a cheap hardening for a follow-up round.
- **M2 (scope note):** `gdn_prefill_backend` validity as a real vllm==0.27.1 `LLM(...)` kwarg is grounded in unit 1/#2378 and plan §4.1 (identity-tested against `cm2378.ENGINE_KWARG_PINS`); a fake-vllm test cannot catch a kwarg the real `EngineArgs` rejects — unit-1/group-1 territory, flagged for cross-group awareness only.
- **M3 (nit):** double dotenv load at import (fork's inherited hand-rolled `_load_dotenv()` then `cm2587`'s canonical `orchestrate.env.load_dotenv()`); harmless, inherited shape.

**Recommendation:** PASS. Fork provenance is faithful and mechanically current; every brief claim verified against the realized code; gates kept; tests real and green.
<!-- /split-review r1 g2 -->

### [g3 f3227b2234] — sub-verdict PASS

<!-- split-review r1 g3 — commit f3227b2234ed750dfba99e8d5729a5a047b8fdf5 -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `f3227b2234ed750dfba99e8d5729a5a047b8fdf5` only (`scripts/issue2587_battery_run.py` 832 lines + `tests/test_issue2587_battery_run.py` 469 lines). `CONTRACT-BEARING: no` ⇒ Steps 0.5/0.55/0.6/0.8/0.9 explicitly SKIPPED (group 8 owns them).
**Tier:** leaf (new per-issue entrypoint + its test; nothing else imports it at this commit).
**Plan:** graded against `tasks/running/2587/plans/plan.md`, verified `readlink plan.md == v3.md`.

## Brief claims — verified

1. **Resume contract — no bare-existence keying anywhere (verified line-by-line).** Cell-done predicate (`_gen_cell_complete`) = done manifest with fp equal to `_cell_fp(cfg,"gen",cell)` AND final jsonl present; the manifest is written LAST (after atomic final jsonl + partial unlink), so the predicate keys on the last-written artifact. Partial resume keys on an fsync'd fp header line; mismatch → `os.replace` quarantine (tested). Chunk completeness = exact row count AND context-id set equality — I traced the duplicate-rows-in-partial case (torn chunk regenerated across multiple crashes): stale+fresh row mixtures can never satisfy count+set simultaneously, so a false-complete is structurally excluded; in-memory `rows` only ever carries fresh or verified-complete chunks. Shard sentinel skip requires fp match (fp includes `upload` and sorted axes) AND zero pending cells. Torn tail dropped only as the literal last line under the flag; a header-only-torn partial quarantines. Cell fp deliberately excludes axes/shard assignment (tested: `_regime_fp` invariant across axes tuples) so completion survives a re-split; sentinel fp includes axes.
2. **Pilot gate.** Measured at the production venue (real loaded model/tok; only the tests fake the GPU boundary): warmup `n=1` + one timed `n=2` call = 2 production-shape per-draw generate calls (steering.py:395-399 confirms one generate call per draw, batched across contexts) — matches plan §9 P5 "warmup + 2 production-shape batches", ceiling 6.0 h matches the §9 P5 row. Report written atomically to `manifests/pilot_gate_report.json`; refuse returns `EXIT_PILOT_REFUSE=7` before `eot_tail_ids` and before any production generation (tested: fake call_count==2, no anchors, no sentinel). Projection arithmetic per-row basis × pending rows is correct for full chunks; sub-batch pilot axes and partially-complete pending axes both bias conservative (over-project).
3. **Cap-hit re-gen.** Strict `> 0.02` trigger, guarded `cfg.max_new_tokens < REGEN_MAX_NEW`; capped file retained via `write_jsonl_atomic` (and swept into the upload glob); regen uses a separate fp-keyed `.max4096.partial` (resume-safe); manifest records `capture_max_model_len_floor = max_ctx_len + 2*max_new_final` — 2×4096 exactly when the regen leg ran, which is the plan's regen-leg arithmetic; the no-regen case records the tighter correct 2×2048 bound.
4. **Pins.** K=10 / temp 1.0 / seed 42 / batch 16 / max_new 2048 all pinned as defaults and test-asserted; recorded per-row `seed = seed_base + draw` matches `generate_batch`'s actual per-draw `torch.manual_seed(seed_base + i)` semantics (steering.py:397-399). Driver never sets `CUDA_VISIBLE_DEVICES` (grep-verified; docstring-only mention). Shard out-root auto-suffix `shard{k}` + `_assert_no_foreign_axis_files` guard (tested, including the own-capped-file pass case).
5. **Think-leak.** `cm2587.think_leak_scan` (containment) flags all rows; hard assert strict `< 0.01` runs AFTER the flagged final jsonl persists and BEFORE partial unlink + done manifest — the failing path leaves rows on disk, NO done manifest, partial retained (all three tested explicitly).
6. **Rollout text persists unconditionally.** Local per-axis jsonl always written; HF upload unconditional in `hf` mode with `resume_skip=False` (no presence-blessing of a stale mirror) and runs BEFORE the sentinel write; `--upload none` is fp-distinguished at both sentinel and cell grain so it can never satisfy a later `hf` run's resume state.
7. **Seams.** `NotImplementedError` at `phase_capture`, `phase_embed`, and inside `main()` before `_resolve_model_revision`/model load; no silent no-op or placeholder return anywhere in the seam (the two phase fns tested; `PHASE_FNS` identity tested).
8. **Test honesty.** GPU boundary faked ONLY via `mock.create_autospec(real_generate_batch, side_effect=impl)` (signature-conformant); `_generate_cell`/`_gen_cell`/`_pilot_gate`/`phase_gen` production bodies execute for real; `FakeTok.apply_chat_template` asserts `enable_thinking is False` per render and is exercised through the REAL `B.render_context_q35`/`context_token_ids_q35`; `_import_check` test performs the real sha256-pinned `git show` import and real signature binds.

## Tests run

Ran this commit's test file against this commit's file state in isolation (extracted `f3227b22:{scripts,tests}` versions to a temp tree; dependency modules verified unchanged `f3227b22..HEAD` by diff-stat): **21/21 passed** (5.2 s). `ruff check` clean on the real worktree paths (a lone I001 on the /tmp copy is a per-file-ignores path-matching artifact, not a finding).

## Issues found (all Minor — none blocking)

- **[Minor] `upload` in the CELL-grain fingerprint over-invalidates.** `_regime_fp` puts `upload` in the base dict, so it enters every cell fp and partial-header fp, not just the shard sentinel. Flipping `--upload none → hf` on the same out-root therefore quarantines all partials, invalidates every done manifest, and REGENERATES all rollouts on GPU, where re-upload of the existing rows would suffice. The docstring's stated goal (a `none` run must never satisfy an `hf` sentinel) needs `upload` only in the sentinel fp. Direction is conservative (redo, never skip) and the production default is `hf`, so this is a cost footgun, not a correctness bug.
- **[Minor] `_read_json` swallows `JSONDecodeError` → `None`** (corrupt manifest/sentinel treated as absent). Fail-safe direction — a corrupt done manifest silently triggers full-axis regen instead of a diagnosable error. Langow-transcribed pattern; noting under the fail-fast standard since it is a silent-default shape, but the fault can only cause extra work, never skipped work.
- **[Minor] Pilot projection ignores chunk-grain partial completion** — pending axes count at full rows, so a resume over-projects by up to ~1 axis of already-banked chunks. Conservative; completed axes drop out of `pending`, so spurious refusal risk is negligible.
- **[Minor] `--axes` duplicates not deduped** — a duplicated axis double-counts in the pilot projection (second occurrence skips via done manifest). Smoke-only flag; harmless.
- **[Minor, test gap] `main()`'s validation paths untested** (`--axes`/`--num-shards` mutual exclusion, shard-index range, the non-gen early raise before model load — the last is testable without network since it raises before `HfApi`). Also no phase_gen-level test that a `none`-mode sentinel fails to satisfy an `hf` re-run (covered indirectly by `test_regime_fp_sensitivity`'s `{"upload": "hf"}` case).

## Observations (not findings)

- On the think-leak assert-fail path, rows persist locally but the end-of-shard HF upload never runs; this matches the plan's own "raw completions upload at end of each P2/P5 shard" design — crash-harvest is the poller/operator flow.
- `build_cfg` recomputes `device` into a `Cfg.device` field the gen phase never reads (main loads the model with its own copy); presumably for unit 3b's capture. No behavioral issue.
- When all cells are done but the sentinel fp mismatches (e.g. a re-split), `eot_tail_ids` (a pinned `git show` import) runs unnecessarily before the no-op loop — trivial cost.

## Group-4 extension risk

Nothing in the gen phase breaks under the 0dfe39011b extension: gen artifacts are phase-namespaced (`battery_gen_done.json`, per-cell manifests fp-keyed with `phase="gen"`), the `_r()` accessor already pins the exact blob capture needs (`capture_answer_states` kwargs signature-checked at import-check), the done manifests carry `capture_max_model_len_floor` for capture re-entry, and `main()`'s early non-gen raise is the single designed removal point.

## Plan adherence

Conforms to plan v3 §4.4 Generation, §4.7 exit convention (single process, explicit terminals, rc 7 refuse), §9 P5 row (ceiling 6 h, 2 shards by axis, gen_batch 16, pilot-gated), checkpoint-cadence row ("P5 per-axis jsonl shards + done manifests"), the phase-order persistence rule (text upload before sentinel/capture), and the §370 destination `issue2587_minpair/raw_completions/anchors/`. No unplanned scope.

## Security check

No secrets, no injection surfaces, no unsafe eval/exec; HF writes go through the sanctioned `upload_dir_sharded`; quarantine paths use `time_ns` uniqueness. Clean.

**Recommendation: PASS** — no Critical/Major issues; Minor items above are fix-at-leisure (the cell-grain `upload` fp is the one worth a follow-up line in the round notes).
<!-- /split-review r1 g3 -->

### [g4 0dfe39011b] — sub-verdict PASS

<!-- split-review r1 g4 — commit 0dfe39011b32dd9398166a88e8a5a52f5bc55bcd -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `0dfe3901` only (`scripts/issue2587_battery_run.py` +1,196, `tests/test_issue2587_battery_run.py` +437; capture P6 + embed phases extending group 3's file). CONTRACT-BEARING: no ⇒ round-level gates 0.5/0.55/0.6/0.8/0.9 explicitly SKIPPED (group 8 owns them). Plan v3 symlink verified (`plans/plan.md → v3.md`).

**Tests + lint:** `uv run pytest tests/test_issue2587_battery_run.py` → **35 passed in 20s** (includes the real-pinned-body capture e2e on a 32-layer tiny Qwen2 and the fails-pre-fix shim demonstration). `ruff check` + `ruff format --check` clean on both files.

### Highest-risk construct — fp32 proxy over the sha-frozen pin: VERIFIED

- Blob at `8265bcd:scripts/issue2162_run.py` re-extracted; sha256 `6f77924461c0…` matches `PIN_2162_SHA256`. `grep -n float16` over the whole 2,967-line blob returns **exactly** lines 1670/1676 — the two terminal `.to(torch.float16)` casts inside `capture_answer_states`, as claimed. Line 954 is `torch.bfloat16` (inside `load_model_and_tokenizer`, not the capture path; forwards unchanged through the proxy anyway).
- Single module-level `import torch` (blob line 73), **no local `import torch` anywhere** — the `r.torch` global swap covers the entire pinned call path (`_right_pad` included). `extract_layer_activations` is imported into the blob from the repo module (its own unshimmed `torch`) but performs no fp16 cast (returns model-dtype hook outputs; the pin pools via `.float()` into fp32 accumulators), so the unshimmed seam is dtype-safe.
- `capture_answer_states` reads exactly the four fields `_PinCaptureCfg` supplies (`cfg.layers/hidden/capture_batch/device`) — verified by reading the function body in full.
- Post-call asserts: dtype != float32 → RuntimeError; non-finite → RuntimeError. Restore in `finally`.
- **Concurrency:** the swap is process-global and non-reentrant, but nothing in this commit can invoke it concurrently in-process: `phase_capture` is a sequential per-axis loop, sharding is process-level (separate interpreters ⇒ separate `_R` module objects), no threads are created, and `embed` never touches `_r()`'s torch. Acceptable; a latent constraint only if a future unit threads captures within one process.
- `test_capture_fp32_shim_preserves_overflow_range` demonstrates fails-pre-fix in-test: the RAW pinned call at 7e4 goes inf in fp16; the shimmed call stays finite fp32; boundaries come from the pin's own state; `r.torch is torch` re-asserted after. (`_r()` is memoized in module global `_R`, so the test's monkeypatch of `r.extract_layer_activations` and the driver's internal `_r()` see the same module object — checked.)

### Brief claims 1–8 — all verified

1. **Capture:** pinned call with `tail_inclusive=True, return_boundaries=True`, batch 8 (`CAPTURE_BATCH=8`, flag-overridable, fp-keyed), `CAPTURE_LAYERS=tuple(range(32))` (test-pinned), fp32 stores, v_A twins + v_C context-end. Boundaries derived solely from the pin's own tokenization (`comp_ids` from `tok(text, add_special_tokens=False)`).
2. **Gate-4 EXACT:** `_gate4_exact_compare` — exact dict equality on all 5 fields, row-count precheck, `strict=True` zip, first-mismatch detail (row/cid/draw/both records), raises before any store/manifest write. Gen-side `n_comp` derivation (battery_run.py:657) is token-identical to the pin's. Test corrupts one `span_end` and asserts no done manifest.
3. **Hook probe is a gate:** runs under `if pending:` BEFORE the per-axis wave, on real rows of the first pending axis; per-layer rel ≤ `HOOK_REL_TOL=1e-5` (== the plan §7/#2330 bar); failing report persisted (`verdict: fail`) then RuntimeError; `assert len(hs) == len(blocks)+1` with the 32-block loader assert. Test proves halt with zero capture manifests.
4. **Token-id concatenation everywhere:** probe ids = `IDS_FN(...) + tok(text, add_special_tokens=False)ids + eot_ids`; the pin concatenates `ctx_ids + comp_ids + eot_ids`; v_C uses `IDS_FN` ids with a per-context `ctx_len` cross-check against gen (battery_run.py:1172). No string re-tokenization of rendered prompts.
5. **Layer convention enforced, not just commented:** the probe compares `captured[L]` vs `hs[L+1]` at {16,22,30} as a hard gate; block-31 pre-norm caveat documented in the probe docstring, `layer_convention` store field, and probe-layer choice; the test additionally cross-validates v_C at layers 0/16/30 against `hidden_states[L+1]`.
6. **Sequencing safe:** `upload_dir_sharded` (read: `verify=True` default, batched exact-set verify, deferred delete-only-after-verify — raises on verify failure) → done manifest written only after upload returns → shard sentinel after all axes. A crash at any point leaves no done manifest over unuploaded data; `resume_skip=False` avoids the same-size false-skip. Upload=none resume additionally requires both `.pt` files on disk.
7. **Re-entry:** `_model_max_positions < floor` → RuntimeError (test-covered); per-cell `max_tail <= floor` assert; `_load_gen_axis` re-verifies each axis's gen manifest `regime_fp` against THIS invocation's gen fingerprint (bank sha, draws, seeds, caps, model_revision) on every capture invocation — differing gen-affecting flags halt (test-covered). `capture_dtype`/`capture_batch`/`layers` all key the capture fp (new dials in the resume regime).
8. **Embed WARN-1:** default route structurally requires realized vLLM == `EXPECTED_EMBED_ENGINE="0.11.0"`; `test_expected_embed_engine_matches_repo_lock` pins the constant against `uv.lock` (lock bump fails loud); any other engine refuses without a report whose `parity_pass/reference_engine/engine` all match (engine must equal the realized version — a report from a third engine is rejected); probe defaults n_anchors=10 (raises when fewer match the banked npz), cos floor 0.995, `max_cos_deviation` reported, report written on miss with distinct rc `EXIT_PARITY_MISS=8`. `vllm_version` rides every chunk npz, perdraw npz, means npz, meta.json, and sentinel; `engine_version` is inside `_embed_regime_fp`, and `test_embed_chunk_resume_and_engine_keyed_fp` proves 0.11.0 chunks cannot satisfy a 0.27.1 run's resume (must reach the engine ctor). Production-time mixing is structurally prevented; unit 5's analysis-side provenance assert is out of this commit's scope (flag for the analysis-unit review).

**Fail-fast sweep:** no `try/except: pass`, no dummy-data-on-error, no silent defaults in the new code. `_reap_engine`'s logged-warning teardown is the sanctioned gotchas.md vLLM reap recipe (best-effort teardown, caller's terminal follows). `EmbedPilotRefuse` → designed rc mapping.

### Findings (all non-blocking)

- **C1 (Minor, recommended hardening):** `_assert_engine_parity` validates only `parity_pass`/`reference_engine`/`engine` on a consumed report — it does not enforce the report's `n_anchors >= 10` or `cos_min_bar >= PARITY_COS_MIN`. `--parity-cos-min`/`--parity-n-anchors` are operator flags, so a deliberately weakened probe run (e.g. `--parity-cos-min 0.5`) yields a "PASSING" report the gate admits. Defaults enforce the plan bar and the report records the realized bar/n for audit, so this needs two deliberate operator acts to exploit — but two extra checks in the consumer loop would close it. Suggest folding into a later round; not worth a re-roll alone.
- **M1 (Minor):** capture manifest/stores record `"dtype": "float32"` (true of the STORE) but carry no plain `capture_dtype` field — a `--capture-dtype bfloat16` debug run is distinguishable only via the opaque fp hash. Provenance nit.
- **M2 (Note, inherited verbatim):** resumed embed chunks contribute fp16-quantized values while fresh chunks contribute fp32 (`out[lo:hi] = z["emb"].astype(np.float32)` vs `= arr`) — a resumed run is not bit-identical pre-normalization. Verbatim from the parent pin (`8265bcd:scripts/issue2564_embed.py:314,339-340`); final artifacts are fp16 either way. Transcription fidelity wins; no change requested.
- **M3 (Note, inherited verbatim, bounded):** the embed pilot gate writes the chunk npz before raising, so each relaunch computes one more chunk before re-refusing (ratchet); bounded at 5 chunks total here (10,800/2,500). Parent behavior, transcribed.
- **M4 (Note, for the analysis-unit reviewer):** empty-completion rows remain ZERO vectors in the va stores (the pin skips them from the forward; `empty_rows` persisted in-store, `n_empty_rows` in the manifest; embed skips empty texts with `n_skipped_empty` recorded). Unit 5 must mask by `empty_rows` before pooling/fitting.
- **M5 (Note):** embed's `_collect_anchor_rows` checks gen done-manifest presence + full grid but not the gen manifest's `regime_fp` (capture does). Structurally mitigated — the chunk fp hashes every row's text, and the embed Cfg's `model_revision="n/a-embed-phase"` makes an fp-equality check impossible by construction.

**Recommendation:** PASS. C1 is the only actionable item; suggest the orchestrator carry it as a minor into the round verdict rather than bouncing this group.
<!-- /split-review r1 g4 -->

### [g5 c47a490211] — sub-verdict PASS

# Split-review r1 group 5 — commit c47a490211 (`scripts/issue2587_fits.py` + `tests/test_issue2587_fits.py`)

## Code-Reviewer Verdict — PASS

**Tier:** trunk (multi-phase fit driver consumed by the round's launcher/analysis units; reviewed line-by-line).
**Scope:** commit `c47a490211fcedd51752436a568d5eeb79404b9f` only. `CONTRACT-BEARING: no` ⇒ Steps 0.5 / 0.55 / 0.6 / 0.8 / 0.9 explicitly SKIPPED (group 8 owns them).
**Plan:** v3 verified (`readlink plans/plan.md` → `v3.md`); graded against §4.3 Fit paragraph + §4.5 verbatim.

## Plan Adherence — the two hardest-checked items

1. **8-arg `fit_ridge` call shape — HOLDS, and the regression pin is real.**
   - `fit_ridge_edge_extended_weights` (fits.py:239-284) defaults `fit_fn=F.fit_ridge_with_weights` and `block=int(LF.RIDGE_BLOCK)`; every fit call is the 8-arg `fit_fn(X, Y, tr, val, ev, grid, dev, block)` (fits.py:233/236 via `_fit_once`).
   - Live-verified in `scripts/issue779_ffc_n1m_fits.py`: BOTH `fit_ridge` (line 1210) and `fit_ridge_with_weights` (line 1232) take 8 positional parameters with NO defaults ⇒ a 7-arg bind provably raises `TypeError`. `MF.run_anchor_gate` (issue2330_matched_fits.py:341-354) also calls the 8-arg `F.fit_ridge(..., LF.LAMBDAS, dev, LF.RIDGE_BLOCK)`.
   - Test pin: `test_fit_ridge_eight_arg_shape_binds_and_seven_arg_raises` uses `inspect.signature().bind` for both shapes + a real tiny 8-arg call asserting `meta["ridge_block"] == LF.RIDGE_BLOCK`; `test_edge_extended_helper_threads_ridge_block_default` proves the default block threads through the seam. The substitution of `fit_ridge_with_weights` for the plan-named `fit_ridge` is justified (payload needed for the plan-required persisted ridge payloads + `apply_map`; the helper's docstring states it is asserted prediction- and λ-identical to `fit_ridge`, and `test_apply_map_reproduces_pred_te` round-trips it).

2. **§4.5 two-fits separation — HOLDS; (a) and (b) cannot be conflated by configuration.**
   - **(a) anchor:** `run_matched7b` streams the banked store at the FULL grain, count-pinned per split against `LF.EXPECTED_SPLIT_N` ({25000, 400, 1000, 999} — live-verified), and runs `MF.run_anchor_gate` FIRST on the full rows (fits.py:842-857). Live-verified in MF: `ANCHOR_EXPECTED_R2 == 0.7250873220237553`, `ANCHOR_TOL == 0.01`, deviation > tol ⇒ `RuntimeError` hard halt ("PORT-PARITY ANCHOR GATE MISS ... failure_class: code"). The record is written to `--anchor-out` immediately after the pass with a `role` field naming it a parity gate, NOT a headline arm; nothing in this script routes the anchor into a contrast or the H1 read (battery reads are unit 5). The anchor path takes no id-subset argument and (b) has no full-row mode — no configuration conflates them.
   - **(b) `arm_7b_matched25k`:** a SECOND fit (fits.py:874-915) whose train/val/test/wc rows are gathered from `split_ids.json` ids via `_rows_for` — any missing id HALTS (`RuntimeError`, never intersection-on-the-fly); realized ids re-hashed and compared against `split_ids["sha256"][split]` (fits.py:883-888, a second independent halt); per-split manifests (n, sha256, banked_n, dropped_from_banked) persisted. Both FIT and SCORE populations are the q35-surviving ids; the wc transfer and the L19 ceiling pair on the same matched test ids. 9B side gathers from the SAME `split_ids.json` with the same sha verification (`_load_and_verify_split_ids` → `_verify_split_sha`) plus a count pin equal to `len(ids)` per split ⇒ row-for-row identity of the two sides' populations by common pinning.
   - **`_sha_ids` domain parity:** verified against the unit-2 producer (`issue2587_map_gen_capture.py:352-355`): identical compact-JSON (`separators=(",", ":")`) plain-int domain, order-sensitive; this copy adds numpy-int `.item()` coercion. `test_sha_ids_matches_unit2_convention_and_is_order_sensitive` pins both properties.

## Other claims — verified

- **fp64 primal, ONE Gram eigh per layer:** `F._ridge_factorize` accumulates the fp64 (H,H) Gram streaming and runs `torch.linalg.eigh` ONCE; all 23 λ (and all extension passes' λ) reuse the factorization via `_ridge_predict_one` — no serial per-λ refits. Selector is val-R² (`val_r2_at_selected`), explicitly not GCV (no GCV call anywhere); per-fit `selected_lambda` persisted in meta, percell rows, and both preds files. λ grid = `LF.LAMBDAS` live-verified `np.logspace(-3, 8, 23)`; edge extension bounded at `MF.MAX_GRID_EXTENSIONS=4` then `RuntimeError` (real-body exhaustion test passes). n_train-vs-d stated AND enforced (`n_tr <= d` refuses) in BOTH the 9B fits and the matched-7B fit.
- **Floors + kNN:** `LF._fit_floors` computes all five floors including identity+learned-bias via `analysis.mapping_baselines.identity_bias_predict` (d_in==d_out on both sides: 4096/4096, 3584/3584 — never skipped; it is unconditional in the helper); `LF._knn_reads` = euclidean+cosine, ks=(1,5,10), chance = k/n_pool reported by the helper.
- **L\* freeze:** `compute_lstar` = argmax val-R², tie→lowest layer (test-pinned), persisted with `frozen: true` in the merged JSON; nothing downstream in this script recomputes it (ceiling layers read the frozen value).
- **No nonlinear leg:** `LF._wc_transfer` (which fits `F.fit_mlp` w8192) is NOT reused; wc transfer goes through `F.apply_map` on the SAME persisted linear payload (fits.py:586, 927), and `test_apply_map_reproduces_pred_te` proves apply_map == pred_te. Zero `mlp` references in the script; `apply_map`'s ridge branch is a pure affine map.
- **Checkpoints/resume:** per-layer `percell/L{l}.json` written LAST after the two atomic `.tmp`+`os.replace` tensor writes (resume keys on the last-written artifact); regime key hashed from generating params only (`LAMBDA_GRID_KEY = ["logspace", -3.0, 8.0, 23]`, test-asserted equal to `LF.LAMBDAS`; store prefix, split shas, h_dim, selector, block, device) — never recomputed float bytes; a checkpoint/record regime mismatch RAISES in both `run_fits` and `run_matched7b`; `run_finalize` requires all 32 checkpoints and asserts ONE regime key.
- **vc2564:** L19 ∈ layers=[14,19,26] enforced (`lacks layer` halt); schema/count/membership fail-loud; no 7B read at any layer outside the banked set anywhere in the script. Both hardcoded HF paths live-probed present (scoped `list_repo_tree`: `vc2564_bank.pt` 42.35 MB; `bank2564_manifest.json`).
- **Streaming economy:** dense 9B store read via ONE `F._stream_n1m_multilayer` pass per split (keys `("cx"|"vx", layer)` + `"ci"` verified against the helper); per-layer `LF._stream_ladder_split` used only on the banked 7B at the single layer L19 with `revision=815ff6d...` pinned (kwarg verified in the helper signature).
- **Ceiling arithmetic parity:** `ceiling_from_draws` mirrors `LF._reliability_ceiling` exactly (per-dim Pearson, Var of two-draw mean ddof=0, same 1e-30 eps, ci-keyed pairing), parametrized on banked grain (9B expected = split_ids counts; 7B expected = `LF.CEILING_EXPECTED_N=1000` full draws, paired on the matched test subset). #2130 shortfall pins ported (tests cover shortfall + missing-id + pairs-by-ci-not-position).
- **Uploads:** `--upload hf` refuses without explicit prefixes (no silent default prefix); `upload_dir_sharded` signature verified (`repo_type` defaults to `"dataset"` — correct for `HF_DATA_REPO`, itself `F.C.HF_DATA_REPO`, never a re-typed literal); `.tmp` names invisible to the `L*.pt` upload globs (test, #2336).
- **Smoke:** thin dispatch to `MF.run_fits_smoke`; flags byte-compatible with unit 2's `_run_fits_smoke` subprocess hook (verified against `issue2587_map_gen_capture.py:3007ff`); smoke blind spots enumerated in the module docstring (HF streaming / anchor / upload / sentinel not executed; λ-edge verdict demoted under the n<d smoke slice). Device resolution fail-loud (`MF._resolve_device`: no silent cuda→cpu).
- **Fail-fast standard:** no `try/except: pass`, no placeholders, no dummy data. The only excepts are three `torch.linalg.LinAlgError` → CPU retries (the sanctioned gotchas.md cuSOLVER exact-backend swap; loud print, `device_realized` recorded, covered by `test_cusolver_linalgerror_falls_back_to_cpu`). Anchor-gate miss and row-identity mismatch both halt loudly (RuntimeError; tests pin both directions).

## Issues Found (all Minor — none blocks)

1. **Minor — `run_matched7b` resume-skip exits before the sentinel write and ignores upload mode** (fits.py:813-817). A run that crashed in the narrow window between the final record write (`complete: true`, fits.py:1075) and the sentinel write (1076-1088), or that completed under `--upload none`, will on relaunch print "already complete" and return 0 — never writing `matched7b_done.json` and never uploading (regime key excludes upload mode). No in-round consumer greps for `matched7b_done.json` today, and the downstream analysis unit reads the HF prefix (absence there fails loud), so this is a footgun, not a live defect. Suggested: re-emit the sentinel idempotently on the skip path and refuse (or upload) when the prior record's `upload.mode` differs from the requested one.
2. **Minor — `--no-edge-extension` disables the plan-mandated λ-edge extension with no finalize backstop.** The flag is explicit opt-in and the edge is recorded in meta, but `run_finalize` never checks `ridge.meta.lambda_grid_edge`, so an edge-selected production fit produced under the flag would merge silently. Suggested: finalize FAILs on any per-layer non-null `lambda_grid_edge` with empty `grid_extensions`.
3. **Minor — `run_finalize` does not cross-check the checkpoints' regime against CURRENT args.** It asserts ONE regime key across the 32 rows but never compares it to a key recomputed from the current `--store-prefix`/`--h-dim`/split_ids shas (full recompute is impossible — device_requested unknown — but store_prefix/h_dim/split_sha could be carried as plain checkpoint fields and diffed). A finalize pointed at a stale out-root would merge old fits while stamping current split shas into the same JSON.
4. **Minor (doc/security nit) — `load_vc2564` comment claims "sha-pinned" but the fetch is unpinned.** `_ensure_hf_file` calls `hf_hub_download` with no `revision=`, and the bundle is loaded `weights_only=False` (pickle). Self-owned repo + schema/count/membership gates + recorded `context_ids_sha256` bound the risk; fix the comment or pin a revision.
5. **Nit — `_rows_for` duplicate-ci masking:** `by_ci` keeps the LAST index for a duplicated context id. On the 9B side the count pin (n_rows == len(ids)) + all-ids-present check make an undetected duplicate impossible; on the banked 7B side (count pinned at the FULL grain, ids a subset) a duplicate displacing a non-selected id would pass. Theoretical — the banked store is parent-validated and revision-pinned.
6. **Nit — the final matched7b record's top-level `role` field** (the anchor's parity-gate label) sits on a record that also contains the headline `arm` block; label present and unambiguous in text, but nesting it under `anchor` would be cleaner.

## Cross-commit awareness

- The module-docstring CLI example's `--preds7b-prefix issue2564_minpair/...` (parent-issue prefix) is already corrected to `issue2587_minpair/...` by group 8's `b043c0ccea`; the real path requires an explicit flag, so the example was cosmetic. No other later commit touches these files (blob-vs-HEAD diff is that one line).

## Tests

`uv run pytest tests/test_issue2587_fits.py -x -q` → **32 passed in 10.7s** (run live in the worktree). Coverage is real-body, not import-only: signature bind pins (both directions), real edge-exhaustion fixture (anti-correlated val targets), row-identity halts (permutation → "row ORDER differs", subset → "id SETS differ", missing → halt), sha domain + order sensitivity, anchor constants + miss-halt + pass-record, apply_map round-trip, L* tie-break, ceiling pairing pins, vc2564 loader halts, regime-key sensitivity, #2336 tmp-glob, smoke e2e through `main()`, `--import-check` subprocess rc=0.

## Style / Security

`ruff check` + `ruff format --check` clean on both files. No secrets, no injection surface (subprocess only in the test, fixed argv). `load_dotenv()` before numpy/torch import (thread caps setdefault verified in `orchestrate/env.py`). `hub.retry_transient(lambda, what=...)` signature verified.

## Recommendation

**PASS.** Both brief-critical items hold with live-verified seams and passing real-body tests; all findings are Minor hardening/documentation items suitable for a follow-up or the next touch of this file (items 1-2 are the ones worth folding in if another round touches `issue2587_fits.py`).

### [g6 3635489766] — sub-verdict PASS

<!-- split-review r1 g6: commit 363548976615b0583f79c0ff704b4972d66f0164 -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `3635489766` only — `scripts/issue2587_judge.py` (970 lines) + `tests/test_issue2587_judge.py` (532 lines), both new adds, unmodified by later round commits (verified: `git log 3635489766..HEAD -- <files>` empty). CONTRACT-BEARING: no ⇒ round gates 0.5/0.55/0.6/0.8/0.9 explicitly SKIPPED (group 8 carries them). Plan v3 asserted (`readlink plans/plan.md` → `v3.md`); reviewed against §4.4 P9 (plan line 99), blind-spot line 144, §14.

**Tier:** leaf (new `scripts/` entrypoint, imported nowhere else; tests import it via sys.path) — reviewed at trunk depth anyway.

### Brief-claim verification (all 9 verified, most by live probe)

1. **Judge discipline — VERIFIED.** `JUDGE_MODEL = "claude-sonnet-4-5-20250929"` (line 64; test line 75 pins it), `JUDGE_MAX_TOKENS = 1024` (test-pinned). Dispatch is `eval.graded_judge.judge_graded` → `eval.batch_judge.judge_completions_batch` (the #663 client); no hand-rolled poller anywhere in the diff. `threshold_base=0` is passed conditionally by `judge_graded` (graded_judge.py:332-333) and pins the Batch route per its own docstring — verified against the live library, and the smoke e2e test asserts `threshold_base == 0`, model, max_tokens, and `n_draws == 1` on every call.
2. **Drop-class separation — VERIFIED.** `_dispatch_wave` copies six SEPARATE `JudgeResult` fields (`n_dropped_draws` content-only per #1313, `n_transport_lost_draws`, `n_refusal_draws` subset, `n_truncation_dropped_draws` subset, `n_api_refusal_draws` sibling, `stop_reason_tally`) — all confirmed present on the real `JudgeResult` (graded_judge.py:138-214) with exactly these semantics; the script never arithmetic-combines them. The test's autospec fake sets each to a DISTINCT value (1,2,3,4,5) and asserts each lands in its own meta field, so any fold/mislabel fails. `zero_max_tokens_stop` = `stop_tally.get("max_tokens", 0) == 0` recorded per wave (the plan's post-hoc binding; 1,464 ≪ 5k ⇒ pilot-gate exempt, matching llm-judging rule 26). `frac_items_complete` guarded against the property's zero-item `ValueError` via `if res.scores` (scores carries an entry per item incl. all-dropped→None, so non-empty whenever items exist).
3. **Fixed denominator — VERIFIED, cannot shrink on any path.** Fire rows are built from the SPEC grid (`denom = len(carriers) × len(draws)`, per-slot count asserted `== denom`), never from the scores dict; a missing/None alias increments `n_incomplete`, and `_value_row` asserts `comply + noncomply + incomplete == denom`. `fire_verdict` is byte-equivalent in logic to the parent's at the pin (extracted and diffed: same guard, same `n_incomplete > 0 → "undetermined"`, same integer `n_comply*100 >= pct*denom`). `fire_verdict(23,1,24) == "undetermined"` pinned at test line 189; `comply_frac` divides by the fixed denom.
4. **Call arithmetic — VERIFIED against the realized bank.** `verify_call_arithmetic(len(j_specs), len(l_specs))` compares realized spec-list lengths (not literals) to 1,392/72/1,464 and `raise RuntimeError` naming both dicts on mismatch; runs on the full slice only (smoke records `verified: False` + `expected: None`, test-pinned). Test computes 1,392 + 72 (+2,400 programmatic) from the live pinned bank and pins both mismatch directions raising.
5. **Instrument identity — VERIFIED LIVE.** `8265bcd75f…` is a COMMIT pin (not a blob id — the docstring's "blob sha256 75b7de…" is the file-content sha256, which matches: I re-hashed `git show 8265bcd…:scripts/issue2564_judge.py` → `75b7de5185f…` exact). AST-extracted the parent `EVAL_PROMPT` from that pin and compared: **byte parity True**. Both rubric sha256 CI pins recomputed and exact (`f6c48e42…`, `d741a82d…`). The artifact's `instrument_identity` block carries both shas + the full `answer_language` rubric TEXT for the #2564 7B-side verbatim consumption. Parent fire/judge constants also match (70%, ceil(0.6×w), n_draws=1, threshold_base=0, max_tokens 1024, same model) — "parent verbatim" holds.
6. **Pilot anti-fabrication — VERIFIED.** `annotate_pilot_rows` is mechanical (axis-keyed; `r["axis"]` KeyError = fail loud, test-pinned), applied to value_rows AND axis_rows; label is the STRING `"7B side pending #2564"` (isinstance-str test-pinned), no numeric/NaN/zero placeholder anywhere. `axis_summary(has_para=False)` returns `n_fired_para: None` (test asserts None, "never a 0") and rejects unexpected para rows. `query_content_oneword` gets an explicit N/A axis row (real axis id — confirmed in `bank2587.PILOT_CELLS`), pilot-labeled. A 9B-only row is structurally distinguishable from a cross-model row.
7. **Rubric-keyed cache — VERIFIED.** Per-wave `cache_dir = work_root/"judge_cache"/<wave>` partitions PLUS the library's #1018 `rubric_fingerprint` keying underneath; distinct prompts + distinct cache dirs asserted in the smoke e2e test.
8. **Programmatic carve-out — VERIFIED.** `check_contains_word` (case-insensitive, `\b`-delimited — stricter than bare substring, boundary behavior test-pinned) appears ONLY in `programmatic_fire_table`; all 10 draws (`PROG_DRAWS = range(10)`, denom 120); judged paths consume graded scores exclusively. Paraphrase slots keep the BASE payload word (test-pinned).
9. **Per-wave smoke cap — VERIFIED.** `max_items` applied inside `_build_wave_items` per family; smoke e2e asserts BOTH waves reach the (autospec'd) client with exactly `SMOKE_JUDGE_ITEMS = 4` items each, matching plan blind-spot line 144.

### Fail-fast audit
No `try/except: pass`, no silent defaults, no dummy-data-on-error. Empty anchors cell → raise; empty selection → raise; alias grammar/budget/collision → raise (grammar verified against the real `batch_judge` custom_id composition: `__{idx:05d}__{comp:02d}` = 11-char suffix on a hard 64 cap ⇒ 53 budget exact; no-`__` rule keeps the join unambiguous); cross-family alias overlap → raise; `--smoke` refusal of the committed `eval_results/` out path (test-pinned). Drops are recorded as drops (None → incomplete), never smoothed into scores. Uploads use `resume_skip=False` (no stale-mirror presence-skip), smoke uploads route to the `/smoke` HF prefix, `atomic_replace` for both JSON artifacts, `as_metadata_dict(git_provenance(), phase="judge2587")` per the #2194 convention. `hub.retry_transient` exists (alias of `_retry_upload`, keyword-only `what` — call binds).

### Live checks run
- `uv run pytest tests/test_issue2587_judge.py -q` → **30 passed, 0.92s** (module scope exercises the real sha-asserted `git show` bank import; only faked boundary is `judge_graded`, via `create_autospec` — and the fake works because the script imports it function-locally at call time).
- `uv run python scripts/issue2587_judge.py --import-check` → ok (argcheck-bind: 2 bound, 0 skipped).
- `uv run ruff check` + `ruff format --check` on both files → clean.
- Parent-pin byte-parity + both sha256 pins recomputed (above).

### Minor (non-blocking)
- **M1 — path-prefix guards are spelling-sensitive.** The `--smoke` refusal checks `str(out).startswith("eval_results")`, so an ABSOLUTE `--out /…/eval_results/…` bypasses it; likewise the `--dry-run` sentinel rebind protects only `DEFAULT_OUT` (an explicit `--out eval_results/…` dry run would overwrite the committed sentinel with an all-incomplete table). Both need a deliberate operator misstep; suggest resolving against `Path.cwd()` or comparing `Path(out).resolve()` parts in a later round.
- **M2 — smoke-conditional arithmetic-gate skip is disclosed mechanically but not named in the plan enumeration.** `verify_call_arithmetic` runs only on the full slice (structurally necessary — a smoke slice cannot total 1,464); the smoke artifact self-discloses (`verified: False`, `expected: None`, test-pinned), and plan blind-spot bullet 2 covers the judge smoke's scope, but doesn't name this gate downgrade verbatim. One enumeration line in the plan would close it cleanly (smoke-blind-spots.md: sanctioned downgrades still enumerate). Not tagged as a 0.71 FAIL because the disclosure is mechanical in the artifact itself and pinned by test — the PASS cannot be over-read.
- **M3 — `frac_items_complete` recorded, not floor-gated in-script.** Consistent with plan §-conventions (manipulation check is a gate-not-DV; drop report binds post-hoc; kill criterion (b) is the read-time trigger) — the analyzer owns the floor read.

**Plan adherence:** §4.4 P9 (line 99) implemented number-for-number; §14's answer_language rubric-TEXT must-ask is the orchestrator's user gate (per brief, not a code blocker) — code side satisfies its preconditions (committed, sha-pinned, artifact-persisted verbatim). No scope creep; nothing missing from the paragraph.

**Recommendation:** PASS. M1/M2 are cheap hardening candidates for any later revision round; neither blocks.
<!-- /split-review r1 g6 -->

### [g7 891d266f7c] — sub-verdict PASS

<!-- split-review r1 g7 — commit 891d266f7c8fbc853296707cbf55f4ac0f5496c8 -->
## Code-Reviewer Verdict — PASS

**Scope:** commit `891d266f7c8f` only — `scripts/issue2587_analysis.py` (3,050 lines, new) + `tests/test_issue2587_analysis.py` (722 lines, new). Round gates 0.5/0.55/0.6/0.8/0.9 SKIPPED per brief (`CONTRACT-BEARING: no`; group 8 runs them).
**Tier:** leaf (new analysis entrypoint + its test; nothing imports it — the group-8 figures test references it by contract only). Reviewed at trunk depth anyway given the registered-lattice content.
**Plan:** v3 confirmed via `readlink plans/plan.md` → `v3.md`; §3, §4.4–§4.6, §6, §7 read verbatim from the main-checkout path.

### Verification run
- `uv run pytest tests/test_issue2587_analysis.py -q` → **29 passed in 2.65s**.
- `ruff check` + `ruff format --check` → clean on both files.
- `uv run python scripts/issue2587_analysis.py --import-check` → ok (argcheck: 2 bound, 0 degraded).

### H1 lattice (plan §3) — CONFORMS
- `h1_verdict(lo,hi)`: `hi<0 → consistent; lo>0 → contradicted; else inconclusive` — disjoint + exhaustive over finite CIs (finiteness asserted, NaN CI fails loud). Boundary zeros (`lo==0` or `hi==0`) route to inconclusive; pinned by `test_h1_verdict_lattice_disjoint_exhaustive` incl. a 200-draw exactly-one-branch sweep. Inconclusive is a first-class verdict (doc carries the convention-14 narration note), not an error path.
- L\* is READ from unit 4's freeze: `load_lstar` refuses `frozen` false/absent (`RuntimeError`); zero argmax calls in the module (3 mentions, all comments). `test_lstar_read_never_reargmaxed` plants `val_r2_by_layer` whose argmax says 7 and asserts the recorded 5 wins, and asserts `frozen: false` refuses — exactly the brief's claim.
- Paired test-row bootstrap: ONE shared `(B,n)` multiplicity matrix drives both sides' pooled-R² draws (`_pooled_r2_draws`, SST about each draw's own resampled mean — algebra verified by hand and by `test_pooled_r2_draws_matches_bruteforce`); per-draw Δ, r9, r7 persisted to `crossmodel_perdraw/h1_delta_draws.npz`. Exact ORDERED test-id equality asserted (`test_compute_h1_ordered_id_mismatch_raises`), matching §4.5(b)'s halt-not-degrade disposition; `preds9["layer"] == lstar` asserted.

### H2 lattice (plan §3) — CONFORMS
- Primary pinned: `PRIMARY_H2_7B_ARM = arm_7b_matched25k`; `resolve_primary_h2_arm` raises on non-unique primaries and raises specifically when `ref_7b_parent` appears as a candidate (pinned by `test_primary_h2_arm_unique_and_pinned`). `ref_7b_parent` values (parent's `arm_779ce` — verified as the parent's primary arm at pin 8265bcd7) enter ONLY the labeled `s_7b_ref_parent` column; the deltas, Spearman blocks, and the h2 verdict all read `s_7b` = the matched arm.
- Bands: `≥0.6 shared / ≤0.2 falsified / (0.2,0.6) inconclusive` per read; combined falsified on either read ≤0.2 OR ≥3 screened sign disagreements; shared iff BOTH reads ≥0.6. The sign screen is real, not vacuous: an axis counts only when (a) ceiling-cleared on BOTH sides (`suppression_verdict` = ceiling pt > 0 AND split-half-ceiling bootstrap CI excludes 0), (b) BOTH sides' bootstrap 95% CIs of the direction cos exclude 0 (sign stability from the bootstrap distribution), and (c) point signs oppose — this discharges the plan's P(D≥3)=0.967 vacuity concern.

### Five cross-unit constraints — all verified in code + producer seams + tests
1. **Install orientation per pair class:** no global reorientation code exists; per-pair `orientation` strings preserve each class's own a→b; pilot `a=value,b=bare` matches `bank2587` gate(ii) (asserts install b-side bare) and is pinned by `test_constraint1_install_orientation_per_class` with both conventions coexisting in one PairArrays.
2. **axis/cell grouping key:** `p["cell"] if p["cell"] != "query" else pair_class` — byte-identical to the parent port (pin line 640); bank2587 pilot pairs carry both `axis` and `cell` with equal values (verified in producer at lines 427–466); consumed as-is.
3. **Engine parity asserted:** 9B npz missing `vllm_version` → `RuntimeError`; non-0.11.0 requires a report with `parity_pass: true` + engine/reference matches; 7B absent key → documented `reference-by-pin` (plan §4.4's banked-parent provenance). Pinned by `test_constraint3_engine_parity` (all 6 branches). Producer (`issue2587_battery_run.py`) writes `vllm_version` into the npz and shares `EXPECTED_EMBED_ENGINE = "0.11.0"`.
4. **Layer indexing:** `_store_col` resolves columns via the store's OWN `layers` list (never positional), asserts membership, and asserts the recorded `layer_convention` contains `captured[L] == hidden_states[L+1]` when present; producer writes that exact convention string and captures all 32 layers (so any frozen L\* resolves). Pinned incl. the wrong-convention refusal.
5. **No cross-model contrast at twin layers:** 7B loader asserts banked layers == (14,19,26); `make_spec_7b` sets `twin_layers=()` (7B `layer_twins` is an explicit `n/a` dict, not an empty-but-present row); `assert_frozen_layer_pair` refuses any 9B layer ≠ frozen L\* and any 7B layer ≠ 19; crossmodel consumes only primary-layer quantities (`norm_obs`/`cos_arm`/`pred`/`obs_tail_primary` are all frozen-pair). Pinned by `test_constraint5_frozen_layer_pair` + `test_crossmodel_refuses_non_frozen_layer_pair`.

### Other briefed claims
- **Port pin:** seeds 2215/21620/2564, B=10,000, 20 splits match the pinned parent verbatim (checked against `git show 8265bcd7:scripts/issue2564_analysis.py`); PairTable/axis-view/null-scheme/split-half code is a faithful parameterized port. NOTE: 8265bcd7 is a COMMIT sha, not a blob sha as the docstring says — resolves fine; cosmetic.
- **Identity+bias baseline + kNN:** `identity_cancellation_check` proves iddelta ≡ identity+learned-bias via the REAL `identity_bias_predict` (asserted, recorded); `knn_retrieval` runs global + per-axis, cosine+euclidean, ks (1,5,10), chance rule + n_pool recorded. Signature use verified against `mapping_baselines` (`true` positional, `pool` kwarg — no collision).
- **§4.6:** ONE `(B,12)` index matrix built once in `main` serves both sides AND the cross-model battery; `delta_draws ≡ draws_9b − draws_7b` pinned exactly (atol=0). Symmetric fire gating: one `sym` mask indexes BOTH sides; one-sided drop counts recorded (pinned by `test_crossmodel_symmetric_fire_drops`). t_11 companion uses the correct t(0.975,11)=2.200985…; LOCO via weight rows through the same battery fns; both are re-reductions of the persisted per-draw matrices (npz per stat). Exact-DP Spearman p verified against n=5 brute force + n=6 extremes; MC fallback labeled, add-one.
- **Vectorization:** all batteries are `(B,n_car)` multiplicity contractions (brute-force-pinned); the only per-draw Python loop is the tie-fallback MC Spearman (10k perms of n≤12 ranks — negligible, and not the battery).
- **Pilot label:** `cross_model_status = "7B side pending #2564"` stamped on pilot axes; pilots structurally excluded from cross-model rows (parent_axes = 7B views); no laundering path found.
- **Fail-fast:** exactly one try/except in the module (`_ref7b_stat`, sensitivity-only column → NaN; see Minor 2). 7B loader's warn-not-assert on ctx-absent va rows is a VERBATIM parent-parity port (parent r2 [g5] "counted loudly" convention, recorded in input_files) with the per-context zero-valid-draw RuntimeError as the floor; the 9B side is stricter (assert n_absent==0). Empty pair selection raises; missing manip files assert with a recovery hint; smoke REQUIRES explicit manip paths (never silently gates on production fire verdicts).
- **Producer seams live-checked** (memory: producer-container-type class): bank2587 `contexts` is a dict and is consumed dict-idiom throughout; `pairs` list keys match producer + parent manifest (probed the committed parent `manipulation_check.json` for the exact `verdict`/`sensitivity{"50","90"}` schema `load_fire` reads); unit 4 payload (`kind/W/xmu/xsd/ymu` via `{**payload,...}`) and preds (`ci_te/pred_te/target_te/layer`, ints→str both sides) match; `apply_map(payload, X, dev)` returns float64 numpy as consumed.
- **Known in-round follow-up:** this commit's `PREFIX_PREDS7B = "issue2564_minpair/analysis_tensors/predictions_7b_matched"` disagrees with plan §6.5 and unit 4's realized prefix; group 8's `b043c0ccea` fixes it to `issue2587_minpair/analysis_tensors/preds_7b_matched` (verified at HEAD). Disclosed in my brief; not counted as a finding.

### Minor (non-blocking)
1. **`ref7b_parent_commit` can ship as "UNRECORDED"** (`issue2587_analysis.py:606`): `--ref7b-parent-commit` is required only when `--ref7b-parent` is explicitly passed; a run using the DEFAULT committed path records the sentinel string in every artifact, while plan §4.5 requires the exact commit RECORDED. Suggest refusing at `load_ref7b_parent` when the commit field is the sentinel (or resolving it via `git log -1 --format=%H -- <path>`).
2. **`_ref7b_stat` swallows KeyError/TypeError → NaN column-wide** (`:2401`): schema drift or a wrong arm key would yield an all-NaN sensitivity column with no halt (`load_ref7b_parent` checks top-level keys + 11 axes only). One assert that ≥1 extraction is finite per stat would make drift loud. Sensitivity-only; low risk.
3. **`primary_h2_7b_arm` metadata coverage:** present in both final deliverables, the crossmodel doc/meta/contract, and the h2 block — but absent from the per-side battery checkpoint JSONs, `h1.json`, the per-draw npz, and perpair rows; plan letter says "every artifact's metadata".
4. **Cross-model `calibration_ratio_to_global` denominator composition:** the 9B global (and swap-global) slopes include the 96 pilot pairs (36 pilot swaps) that the 7B global cannot contain — a small compositional asymmetry inside a cross-model delta the plan does not pin either way. Suggest a disclosed shared-pairs-only global twin or a one-line note in the stat definition.

### Nits
- `_sym_grid`'s all-ones fallback when zero vps survive the symmetric fire mask is undisclosed at the stat-row grain (pair-level fallback IS disclosed via `symmetric_headline`; the mask is symmetric, so no cross-model asymmetry).
- `preds7`'s `layer` key (written by unit 4) is not asserted == 19; selection is filename-pinned only. One-line assert available.
- 7B va loader dropped the parent's `tail.shape` assert; numpy's boolean-index/broadcast errors still fail loud on any mismatch.
- H2 combined rule (either-read ≤0.2 falsifies; BOTH ≥0.6 for shared) is a coherent completion of §3's singular-ρ falsified sentence and is recorded verbatim in the artifact (`combined_rule`).

**Recommendation:** PASS. No Critical/Major findings; the four Minors are hardening/disclosure items suitable for a follow-up round or the next touch of this file.
<!-- /split-review r1 g7 -->

### [g8 b043c0ccea] — sub-verdict **FAIL**

# Code Review: #2587 split-review r1 — group 8 of 8 (commit b043c0cce, unit 6 figures + prefix reconciliation; CONTRACT-BEARING round gates)

**Verdict:** FAIL
**Blocker tags:** marker-shape, smoke-blind-spot-unenumerated, substantive
**Tier:** trunk (commit itself is scripts+tests, but the round spans `src/`; reviewed every in-scope line)
**Diff size:** +1,261 / −2 across 5 files (commit-scoped read per SPLIT-REVIEW SUB-SCOPE; round-wide listing consulted name-status only — 19 pure additions, 0 modifications)
**Plan adherence:** PARTIAL (hero 2 + matched-n table + prefix reconciliation COMPLETE; hero 1 partial; §6/§13 exploratory dump ~10 items unimplemented — see Major 1)
**Tests:** PASS — 18/18 new tests pass, run by me (`tests/test_issue2587_figures.py` + `tests/test_issue2587_prefixes.py`, 19.4 s)
**Tests actually run:** yes
**Lint:** payload PASS (ruff check + format clean on all 5 files, run by me); no-flags `workflow_lint.py` **INCONCLUSIVE** — implementer reports rc=124 timeout at 510 s under VM load 59; an inconclusive lint is inconclusive, NOT a pass. Zero FAIL lines named issue2587 files in the partial output; the figures entrypoint satisfies the dotenv-before-numpy/matplotlib ordering by inspection (`issue2587_figures.py:48-58`) and `tests/test_shared_vm_thread_caps.py` passes it (see git-provenance below).
**Security sweep:** CLEAN (no secrets, no eval/exec, no network in the renderer; analysis stays read-only on HF)
**Prior-concerns ledger:** empty at review start (Step 0.8 walk: `list-concerns --open-only` → `[]`); ONE new concern persisted this round: `plan-s6-figures-deliverable-gap` (CONCERN, Rule 11 — silently-degraded plan deliverable)
**Main-side divergence:** brief carries no `diverged_on_main` list and no probe-failed line → the orchestrator probe found nothing; not re-derived.
**Needs user eyeball:** marker (d)'s x-label shortening ("dashed verticals" → "dashed") and the n10k reference points rendered as dotted-joined series — both fine by my read, flagged by the implementer for taste.

---

## Round-level gates (this group only)

### Step 0.5 — implementation marker shape: PASS
Highest `epm:experiment-implementation` (v1) fetched from canonical task state. `### (a)`–`### (d)` present in order; `## Smoke run` is its OWN H2 (line 65 of the note), not a `### (d) Smoke run` displacement; `(c)` carries multiple copy-pasteable fenced commands + observable success signals. I re-ran (c)'s happy-path command verbatim: `uv run pytest tests/test_issue2587_figures.py tests/test_issue2587_prefixes.py -q` → `18 passed`, matching the claimed signal. Presentational nits only (see Style).

### Step 0.55 — smoke-architecture marker: **FAIL (Critical, `marker-shape`)**
- Marker present with parseable `verdict: PASS_PARTIAL arms_stubbed=<9 dotted arms>`.
- **`task.py check-smoke-arch-registry 2587 --repo-root <worktree>` → REFUSE: "no line-anchored `arm-registry:` line found."** The marker has a bare `arm-registry:` heading with the per-driver derivations in FOLLOWING bullets — the #2176 pitfall-4 shape (#2330 R1 g7 precedent): however correct the bullet prose, the grammar accepts only the single-line `arm-registry: source=<expr> file=<path> n=<int> members=<sorted-comma-list>` or `arm-registry: N/A — <reason>`. Step 6d.0 runs this same checker pre-dispatch, so the malformation would wedge the production dispatch.
- **Second defect, same re-post:** the bullet "judge / analysis / map_gen_capture: single-entry drivers (no --phase/mode/stage argparse arg; grep confirmed) → one arm each" is FALSE for map_gen_capture: `scripts/issue2587_map_gen_capture.py:2808-2814` defines `--capture-mode` with `choices=["coresident", "phase_split_gen", "phase_split_capture"]`, and its help text says production P2 runs `phase_split_gen` then `phase_split_capture` — a genuine mode-dispatch table (`:2013-2017` branch main flow on it). The #2163 omission shape. Mitigating: nothing REAL is overstated — the whole driver is FALLBACK-rowed and in `arms_stubbed`, so the omitted arms are honestly disclosed at driver grain.
- **Substance verified by me (fallback arm, checker down):** per-driver registries set-match the marker's enumeration otherwise — `battery_run` `PHASES=("gen","capture","embed")` (`issue2587_battery_run.py:168`), `fits` `choices=("fits","finalize","matched7b")` (`issue2587_fits.py:1102`), `figures` FIGS = 9 keys (all 9 rowed REAL), judge/analysis genuinely single-entry (no `--phase`/choices dispatch). `arms_stubbed` (9) == FALLBACK-rowed set (9) — PASS_PARTIAL-consistent. `import-resolution:` line present in accepted shape (mode + six rc=0 entrypoint commands).
- **Fix (one marker re-post, no code change):** cheapest conforming form for this 6-driver mixed-registry round is `arm-registry: N/A — <reason enumerating the per-driver dispatch tables>` (the #2502-sanctioned escape for choices-based drivers; the N/A reason tolerates embedded commands and the existing dotted per-arm row keys) — INCLUDE `--capture-mode (coresident|phase_split_gen|phase_split_capture)` in the reason and keep `map_gen_capture.main`'s FALLBACK row / `arms_stubbed` entry as-is (or expand both consistently). Alternative: ONE union line with comma-listed `file=` and bare-token rows.
- Mechanizable: yes — the existing checker; it was run and REFUSEd.

### Step 0.6 — end-to-end smoke gate: PASS
`## Smoke run` covers every CPU-feasible phase of the §4.7 DAG with one `### <phase>` each: exact command, slice/grain, rc=0, artifact digest (figures: 26 files + all-8-PNG read-back; fits CPU leg: per_layer {"3","7"} digest). GPU/API phases use the labeled `— Carve-out (GPU-bound/API-bound)` form with all three substitute-coverage items + one-sentence constraint. Resume-matrix and production-outroot-unit sub-blocks present in the smoke-arch marker with REAL/FALLBACK-reason vocabulary (production out-root FALLBACK is structurally forced pre-production — inputs don't exist yet; declared). Output-path hygiene: smoke redirected to `/tmp/issue-2587-smoke/`; round diff contains zero modifications to committed `eval_results/`/`figures/` (19 pure `A` additions); my own verification runs wrote only to `/tmp` and pytest tmp_path. The one figure defect the smoke found (clipped x-label) was fixed same-turn via `XLABEL_DEPTH` and re-read — the figure-sanity duty done right.

### Step 0.71 (round scope, per brief) — smoke blind-spot enumeration: **FAIL (Critical, `smoke-blind-spot-unenumerated`)**
- Commit scope: `issue2587_figures.py` adds NO smoke-conditional branch (grep: only a docstring path mention); the marker's unit-6 empty-form literal "none — smoke executes every production gate" is TRUE for this commit.
- **Round scope: `scripts/issue2587_analysis.py` (unit 5b) carries a runtime `--smoke` flag whose gate-DOWNGRADES are enumerated NOWHERE:**
  - `:1127-1129` — production-only asserts on `expected_contexts` / `expected_pairs` (full-coverage gates) skipped under smoke;
  - `:1617-1620` — production-only assert that every expected axis is present in the stores, skipped under smoke;
  - `:2894-2895` — production-only `n_car == 12` assert, skipped under smoke;
  - `:609-610` — bootstrap/null batteries silently narrowed to B=100 under smoke (production B=10,000).
  - Evidence of the gap: the marker's `### analysis` sub-section is the ONLY phase section WITHOUT a "Smoke blind-spot enumeration" line (figures/fits/battery/map_gen_capture/judge each carry one); the plan's §4.7 enumeration cannot cover it — it predates this lever and even states "no gate is DOWNGRADED in the pod smoke … the `--tiny` CPU mode exists only inside pytest", while unit 5b added a NON-pytest runtime `--smoke` CLI branch the plan did not anticipate. Per `smoke-blind-spots.md`, sanctioned calibration downgrades are exactly where enumeration (disclosure) is mandatory.
  - Bug-class sweep (`### Bug-class sweep: unenumerated smoke-conditional branch`): swept all 6 round scripts. All four unenumerated sites are in `issue2587_analysis.py` (above). Enumerated/clean elsewhere: `issue2587_judge.py` `--smoke` narrowing (carriers/axes/arithmetic-skip) — enumerated under unit 5a, and its `:718-722` smoke branch is an UPGRADE (refuses committed paths); `issue2587_fits.py --smoke-chunk-dir` substitution — enumerated under unit 4; `issue2587_map_gen_capture.py` `--fits-smoke` delegates to the enumerated fits leg; `issue2587_battery_run.py` — no smoke-conditional branches. Analysis `:559/:568` out-root rebinds and `:570-572` (REQUIRES manip args under smoke) are hygiene/upgrades, not downgrades — noted, not blockers.
  - Impact: a smoke "PASS" of the analysis phase certifies gates it never evaluated (the SLURM-5005 class); the B=100 narrowing additionally means smoke never certifies bootstrap calibration, undisclosed.
  - Fix (marker re-post, no code change — the branches themselves are sanctioned calibration/coverage downgrades): add a unit-5b enumeration block naming the four sites, e.g. "analysis `--smoke`: expected-pairs/contexts, axis-completeness, and n_car==12 production asserts SKIPPED; B narrowed 10,000→100 — production-coverage gates and bootstrap calibration NOT certified by an analysis smoke."
  - Mechanizable: yes — `workflow_lint.py --check-smoke-blind-spots` (AST scan, flags named `smoke`) against the plan; it is WARN-only by design, the reviewer lens is the binding arm.

### Step 0.8 — prior open binding concerns: walked
Ledger empty at review start. Implementers raised none this round (matches brief). I persisted `plan-s6-figures-deliverable-gap` (CONCERN, round 1, by code-reviewer) for Major 1 below.

### Step 0.9 — git-provenance self-check: run, nothing attributed to the round
The four files named by `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` (`issue1901_mlpdense_fold_analysis.py`, `issue1901_mlpdense_fold_figures.py`, `issue2254_firstk_ctxext_sensitivity.py`, `issue2378_lenmatch_fig.py`) each show **0 commits** in `round_parent..HEAD` → **pre-existing-on-trunk**; I re-ran the test at HEAD: 1 failed / 26 passed, the failure output naming only those 4 main-resident files — the new figures entrypoint PASSES the scan. The `workflow_lint` FAIL at `issue1901_mlpdense_fold_analysis.py:45` is the same class. Neither is raised as a finding; no `git-provenance` blocker needed (probes are conclusive).

---

## Plan Adherence (commit scope: §4.6 hero + matched-n table + §6 figures list + §6.5 prefix)

- Hero 2 (`fig_hero_layer_sweep`, R² vs FRACTIONAL depth, both models): ✓ — `issue2587_figures.py:192-266`; fractions `layer/(n_layers−1)` both models (`:140-141`), 7B anchor at 19/27 (`:250-258`), floors envelope (`:210-211`), L* star (`:213-224`), #2330 n10k series both models.
- Full-attention dash convention pinned to `scripts/issue2329_figures.py`: ✓ — `FULL_ATTENTION_LAYERS_9B = frozenset({3,7,11,15,19,23,27,31})` / `N_LAYERS_9B=32` / `FULL_ATTN_COLOR="#9467bd"` (`:79-83`) regex-pinned by `tests/test_issue2587_prefixes.py:83-92` against the LIVE source (I verified the source lines at `issue2329_figures.py:77-81` match; pin genuine, drift is test-breaking).
- Matched-n table (9B@L* vs 7B@L19; floors, kNN, ceilings, anchor gate, paired H1): ✓ — `issue2587_figures.py:288-410`; anchor block `:334-340`, H1 rows `:342-346` (optional input, see Minor 2); md slug-freedom test-asserted (`test_matched_n_table_md_and_json`).
- ONE `DISPLAY` label map, no internal slugs on rendered surfaces: ✓ — `:89-118`; every legend/panel-title/table string routes through `DISPLAY`/`axis_label`; floors fallback `DISPLAY.get(fname, fname)` can only fire on an unmapped floor name, and the producer emits exactly the 5 mapped names (`issue2330_matched_fits.py:496,2003-2007` via `issue1491_ladder_fits._fit_floors`).
- No on-canvas caption blocks: ✓ — zero `fig.text`/`suptitle`/`annotate`/`ax.text` in the file (grep). The `XLABEL_DEPTH` parenthetical gloss is an axis-label definition (the sanctioned ships-inside-the-artifact idiom), not a caption block.
- §6.5 prefix reconciliation: ✓ — see Part 2 below.
- §6 exploratory dump: **± PARTIAL** — see Major 1.
- Hero 1 (`fig_hero_crossmodel_axis_profile`): **± PARTIAL** — 3 scale-free-stat panels with 9B/7B-matched/7B-parent points per axis (`:516-569`), but the plan's §6 spec for hero 1's direction-cos panel — "with ceilings + nulls + iddelta whiskers" — is not rendered, and the `crossmodel_contrasts.json` rows the figure consumes do not carry those fields (producer rows: `s_9b/s_7b/s_7b_ref_parent/delta*/fire/ceiling_cleared`, `issue2587_analysis.py:2585-2597`), so it cannot be fixed by figure code alone — the input contract needs threading.

## Part 2 — prefix reconciliation: VERIFIED CLEAN

- `issue2587_analysis.py:117` `PREFIX_PREDS7B = "issue2587_minpair/analysis_tensors/preds_7b_matched"` == plan §6.5/§9 (plan lines 210/306/330); threads through the `--prefix-preds7b` argparse default (`:543`) into the P10 staging READ (`:2925-2929`).
- `issue2587_fits.py:65` docstring example == plan value; the real `--preds7b-prefix` argparse default is `None` (`:1140`) with a fail-loud refusal at `--upload hf` time (`:1022-1024`, the #1005 no-default shape) — the docstring example is the dispatch-time target of record, and it now matches the consumer default and the plan. Producer example == consumer default == plan: the paired-script default-path contract holds in both modes.
- OLD literal `predictions_7b_matched`: **0 hits** in the entire worktree (grep incl. non-code, `--exclude-dir=.git`).
- Parent READ constants untouched and correct: `PREFIX_2564` (`analysis:115`), `VC2564_HF_PATH`/`BANK2564_MANIFEST_HF_PATH` (`fits:116-117`, consumed at `:960-965` to read the parent's BANKED 7B stores per §4.5/P8 — reads from `issue2564_` are by design).
- `tests/test_issue2587_prefixes.py` genuinely pins the seam: regex-extracts the live constants/examples and compares to the plan literal; asserts no write-flag example resolves under `issue2564_` while exempting the read constants; pins the `default=None` fail-loud shape. Fails-pre-fix is certain by construction: the extraction regex matches the parent blob's old line, whose value ≠ the plan literal (the parent-side value is visible in this commit's own diff hunk).

## Issues Found

### Critical (block merge)
1. `epm:smoke-architecture-check` marker — malformed `arm-registry:` line (checker REFUSE) + false "single-entry" registry claim for map_gen_capture. Tag: `marker-shape`. Details, evidence, substance verification, and the one-post fix under Step 0.55 above. Mechanizable: yes (the existing `check-smoke-arch-registry`).
2. `scripts/issue2587_analysis.py:1127-1129, 1617-1620, 2894-2895, 609-610` — smoke-conditional production-gate downgrades with NO blind-spot enumeration anywhere (marker or plan). Tag: `smoke-blind-spot-unenumerated` (substantive-class, never stripped). Details + sibling sweep + the one-block marker fix under Step 0.71 above. Mechanizable: yes (WARN-only AST scan exists; reviewer lens binding).

### Major (revise before merge)
1. **Plan §6/§13 figures deliverable gap + overstated marker adherence line.** The plan's exploratory dump names ~16 items and §13 lists "the two hero figures … the manipulation-check table, and the exploratory dump" as deliverables; the round implements 7 exploratory arms and NO renderer exists anywhere (analysis.py emits JSON only — zero savefig) for: per-axis ‖Δ̂‖-vs-‖Δ‖ scatters; install-vs-swap violins; axis-identity heatmaps; cross-family consistency scatters (observed + predicted); edit-dose scatters per tokenizer; battery-side Δ-retrieval acc@k curves per arm (fig_knn_per_layer is the FIT-side kNN, a different read); per-carrier direction-cos transfer matrices; split-half-vs-direction scatters; pilot-axis panels; think-leak + cap-hit tables; the manipulation-check table; q25-vs-q35 token-count-equality table; L*-vs-{16,22,30} sensitivity twins of hero 1; span-mean pooling twin; 7B-matched-vs-parent agreement scatter. Several need inputs the FIGS registry doesn't load (perpair-grain artifacts), so this is missing implementation, not a run-later gap. The marker's plan-adherence line "exploratory over-production — DONE (7 figures)" overstates. Tag contribution: `substantive`. Fix: implement the remaining renderers in the fix round (they are CPU-cheap, fixture-testable like the existing 7), or record an explicit descope in the marker + a plan-WARN disposition — never "DONE". Persisted as concern `plan-s6-figures-deliverable-gap`. Mechanizable: partially — a check could diff plan §6's semicolon-list against `sorted(FIGS)`.
   - Bug-class sweep (`### Bug-class sweep: plan-named figure without renderer`): the enumerated list above IS the sweep (grep for each plan term across `issue2587_figures.py` + `issue2587_analysis.py`; hero-1's missing whiskers/ceilings/nulls counted here too). No further siblings.

### Minor (worth fixing, non-blocking)
1. `issue2587_figures.py:516-569, 571-613` — the two crossmodel figures have no emptiness guard: an empty `stats[<stat>]["axes"]` list renders a BLANK panel set without failing (unlike `sweep_layers`' RuntimeError on an empty `per_layer`), and the tests' 5,000-byte PNG floor passes a blank-axes render. Empty/blank-render assessment (brief ask): all sweep-based figures fail loud (missing file → FileNotFoundError `:157-160`; empty per_layer → RuntimeError `:164-168`; all-non-finite floors → RuntimeError `:178-186`); only the crossmodel pair can render blank, and only on an empty axes list (an all-None ROW is by-design pilot-pending rendering). Fix sketch: `if not stats["direction_cos"]["axes"]: raise RuntimeError(...)`. Mechanizable: yes (1-line guard + a 3-line pytest).
2. `issue2587_figures.py:625` — `matched_n_table`'s `"delta?"` optionality lets a production `--figs all` run ship the table WITHOUT the §4.6-required paired-H1 rows on an INFO log only. Disclosed in the marker and needed for pre-production smoke, but consider hard-requiring it when `--out-dir` is the canonical default.
3. Color-role reuse across the exploratory set: `paper_palette_role("control")` = WildChat transfer in `fig_wc_transfer_per_layer:456` but = cosine metric in `fig_knn_per_layer:474`; `fig_floors_per_layer:431` colors the ridge curve `paper_palette(6)[0]` where the hero uses `paper_palette_role("primary")`. One-color-one-meaning holds within each figure and for the load-bearing 9B/7B/full-attn assignments; this is cross-figure drift among exploratory panels only.

## Unaddressed Cases
- Crossmodel empty-axes blank render (Minor 1).
- `_table_side` (`:279-284`) carries dead defensive branches (`if "ridge" in row else …`) — both call sites construct rows WITH `"ridge"`; harmless, mildly misleading.

## Style / Consistency
- Marker (c) cites test names that don't exist verbatim (`test_missing_required_input` vs `test_cli_missing_required_input_fails_loud`; `test_inverted_ci_clamped` vs `test_delta_forest_inverted_ci_clamps`; `test_unknown_fig`; `test_table_without_delta`) and counts "12 tests" in the figures file (actual: 11; 18 total with prefixes — the commands and totals are correct). Presentational only.
- Matched-n md header renders bare "(L*)" — `DISPLAY["lstar"]` has the gloss; harmless in context (the md body defines everything else).

## Unintended Changes
- None. 5 files, all in the commit's stated scope; round-wide listing is 19 pure additions.

## Tests
- New coverage: real-render pins for all 9 FIGS arms incl. CLI e2e via `main(argv)`; inverted-CI clamp through the real errorbar call; fail-loud missing-input and unknown-fig paths; optional-delta branch; md display-name discipline; both ends of the prefix seam + fail-loud `default=None` pins + the #2329 convention pin + the anchor-constant cross-file pin. Fixture schemas verified by me against the same-round producer code (`run_finalize`/`run_matched7b` doc keys at `issue2587_fits.py:619-626, 761-771, 1047-1063`; crossmodel/h1 keys at `issue2587_analysis.py:2308-2318, 2585-2597, 2720`) and against the committed #2330 artifacts.
- Missing coverage: blank-render guard for the crossmodel pair (Minor 1); no assert that every FIGS-rendered legend/label string is DISPLAY-routed (the md test covers the table only) — cheap to add, low value.
- Existing tests still valid: yes (round adds files only; the 1 red in `test_shared_vm_thread_caps.py` is pre-existing-on-trunk, probed).
- Sandbox status: ran normally.

## Security Check
- No issues found.

## Recommendation
**Revise-then-re-review — the fix round is a MARKER re-post plus a bounded figures follow-up, not a rebuild.** (1) Re-post `epm:smoke-architecture-check` with a conforming `arm-registry:` line (N/A-form with the per-driver dispatch tables, incl. `--capture-mode`, is the cheapest; keeps existing dotted rows) and add the unit-5b analysis `--smoke` blind-spot enumeration block to the implementation marker. (2) Either implement the remaining plan-named exploratory renderers (+ hero-1's ceilings/nulls/iddelta whiskers, which also needs the crossmodel row contract extended) or record an explicit descope — the persisted concern `plan-s6-figures-deliverable-gap` tracks it. This commit's own code — hero 2, matched-n table, the prefix reconciliation, and both test files — is correct, fail-loud, well-tested, and needs no changes.
