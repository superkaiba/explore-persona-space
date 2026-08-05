# Agent-memory index curation — task #2093

Per-agent curation manifests for the three near-ceiling agent-memory indexes
(`.claude/agent-memory/<agent>/MEMORY.md`). Curation happens at the INDEX-ROW
level only: ZERO feedback bodies were deleted — every retired/merged row's
body stays on disk and loads on demand. Each retire cites the superseding
always-on surface with a one-line quoted covering clause (verified by grep at
curation time, 2026-08-05, worktree `issue-2093` at `origin/main` ==
`b8164e9fe0`), or records the relocation evidence for a
target-no-longer-exists retire.

## experiment-implementer

**Before:** 19,697 bytes / 118 rows. **After:** 14,449 bytes / 83 rows.
**Classification:** 83 KEEP · 3 MERGE groups (3 rows folded into 3 rewritten
rows) · 32 RETIRE. Every surviving pointer resolves on disk (`test -f` loop:
0 missing over 89 referenced bodies).

### Merges (rows in → row out; all original bodies stay reachable)

1. **"Smoke coverage parity (width, class×regime, branches)" + "prefix-arm
   degenerate cloud kills spectral smoke"** → one family row keyed on
   `feedback_smoke_ft_zero3_width_parity.md`, with
   `feedback_prefix_arm_degenerate_cloud_smoke.md` named inline as the
   zero-variance single-context sibling (#1112). Same trap class: a smoke
   slice whose realized composition cannot exercise what production
   exercises.
2. **"Long loops: per-unit atomic writes + --resume" + "--resume pins every
   output-affecting regime key"** → one family row keyed on
   `feedback_long_running_analysis_needs_resume.md`, with
   `feedback_resume_metadata_pin_every_regime_key.md` named inline
   (#722 #399 #600). Same trap class: long-loop resume discipline.
3. **"Ruff strips unused imports" + "Edit-then-read-modify-write lost
   update"** → one "Format-hook edit hazards" row keyed on
   `feedback_ruff_strips_unused_imports.md`, with
   `feedback_edit_then_read_modify_write_lost_update.md` named inline
   (#1345). Same trap class: the PostToolUse format hook racing an edit.

### Retires (32) — verbatim row + justification + quoted covering clause

Format: the original index row, then the superseding surface. "gotchas.md" =
`.claude/rules/gotchas.md` (on-demand rule, trigger indexed always-on in
`.claude/rules/LESSONS.md`); line numbers from the curation-time worktree
copy. Bodies retained on disk in all cases.

1. `- [MooseFS stale-serves an rsync OVERWRITE](feedback_moosefs_stale_serve_on_rsync_overwrite.md) — rsync rc=0 but pod runs old bytes; rm-then-rsync + sha both sides (#1482)`
   → gotchas.md:86: "rsync/scp INTO a MooseFS-backed pod path can silently serve STALE bytes to a subsequent reader on the same mount (#2051 rider, from archived #2053)."
2. `- [#1586 disk-headroom family](feedback_resume_aware_phase_headroom_gates.md) — scale need by PENDING cells; reap ckpts on verified upload + smoke roots at phase entry (#1586 fu)`
   → `.claude/rules/plan-compute-sizing.md` § Out-root mount binding: "The gate MUST be resume-aware: compute the phase's PENDING set with the same predicates the phase's own resume scan uses" (incident #1586 fu crash 5; `_wave_headroom`); the reap-on-verified-upload half is the same rule's ladder-retention/fan-out blocks ("delete a cell's local outputs once uploaded to the Hub").
3. `- [SAE published FVE/L0 need the reference token pool](feedback_sae_reference_eval_token_pool.md) — dictionary_learning remove_bos: BOS-8 strip (#1482)`
   → gotchas.md:474: "SAE fitness/eval against a published FVE/L0 reference must reproduce the reference eval's TOKEN-POOL semantics — not just its encode/decode math."
4. `- [Batch-API judge item ids — charset + 53-char budget](feedback_batch_custom_id_53_char_budget.md) — custom_ids must match ^[a-zA-Z0-9_-]{1,64}$ (#1776, #1415)`
   → gotchas.md:400: "Anthropic Batch `custom_id`s must match `^[a-zA-Z0-9_-]{1,64}$` — BOTH a charset AND a length constraint".
5. `- [Pilot timing gates measure at the SWEEP's execution shape](feedback_pilot_timing_gate_sweep_shape.md) — batch-1 pilot vs batched sweep false-fires a (#1415)`
   → plan-compute-sizing.md:399 (§ Per-cell fit phases): "a phase that runs B-wide batched calls is piloted [with one B-wide batched call normalized per-sample, never a serial batch-1 loop … (#1415)]".
6. `- [GCV lambda interpolation degeneracy](feedback_gcv_lambda_interpolation_degeneracy.md) — n_tr ≲ D: GCV → grid-min λ, R² −2..−46 (#1335, #1345)`
   → CLAUDE.md § User-chat inline free analysis (always-on): "GCV-specific ban (#1887): pure-GCV λ selection at n_train < d is REFUSED (the shared #825 fit cores enforce this by default — GCV runs only WITH a dof cap, default 0.9, or under an explicit LEGACY_UNGUARDED_GCV opt-in)"; plus `.claude/rules/artifact-reuse.md` check (l) (documents the #1335 grid-min collapse + registered mitigations). Residual noted: a from-scratch GCV fitter bypassing the shared cores is not code-guarded, but writing one instead of reusing the cores violates the reuse rules independently.
7. `- [rule-24 surgical re-judge recipe](feedback_rule24_surgical_rejudge_recipe.md) — recover 529 draws from judge_raw via (#1315)`
   → `.claude/rules/llm-judging.md` rule 24(ii) (llm-judging.md:163 area): "The re-judge BYPASSES the rubric-keyed judge cache for the affected draws — surgical per-draw merge, a fresh `cache_dir`, or draw-indexed keying … (#1090's own recovery used the surgical per-draw merge for exactly this reason)."
8. `- [Hub prefix-mirror staging ≠ consumer layout](feedback_hub_prefix_mirror_vs_consumer_layout.md) — pure hub-rel→local-rel map + fail-loud entry check (#928 #1481 #1774)`
   → `.claude/rules/artifact-reuse.md` check (h)(iv) — the rule inlines the requirements (pure mapping fn, fail-loud entry check, 1-file staging probe per (family × consumer) pair) and cites this very body at artifact-reuse.md:280: "Full 4-point implementation recipe …: `.claude/agent-memory/experiment-implementer/feedback_hub_prefix_mirror_vs_consumer_layout.md`" — the body stays reachable FROM the rule.
9. `- [Artifact pin/provenance family](feedback_artifact_pair_provenance_coherence.md) — sha pins ≠ capture-time identity; 4 siblings in entry (#922 #1776 #600 #601)`
   → CLAUDE.md reuse bullet (always-on) clauses (f) content identity/sha-pinning (#600) + (j) pairwise provenance ("a consumed input regenerated AFTER its dependent capture is inconsistent regardless of sha pins", #922); artifact-reuse.md:417 check (j): "a pin freezes current bytes, not pair coherence" — check (j) also cites the sibling body `feedback_pinned_artifact_pair_mutual_inconsistency.md` directly.
10. `- [vLLM generate traps: chunk batches, use_tqdm=False](feedback_vllm_large_batch_deadlock_chunk_it.md) — one huge generate() wedges EngineCore (chunk ~500); bank length-validate sibling in entry (#664 #613 #952)`
    → gotchas.md: "A single large `llm.generate(N_prompts, ...)` call can DEADLOCK the vLLM v1 EngineCore worker — chunk large batches by default (PREVENTION for the hang the #601 entry recovers from)" (with the `VLLM_GREEDY_CHUNK_SIZE` code recipe at gotchas.md:155-161) + gotchas.md:194 (`use_tqdm=False` ZeroDivisionError entry).
11. `- [vLLM teardown + HF coexistence](feedback_vllm_orphan_worker_after_destroy.md) — destroy_* leaves workers (psutil child-kill); gpu_mem_util≤0.5 (#399 #685)`
    → gotchas.md:132-133: "vLLM in-process teardown does NOT reap worker subprocesses. When the SAME process loads vLLM then a non-vLLM framework (HF Transformers …)" + "Reaping recipe (vLLM v1, 0.11.0)"; the teardown entry is also named always-on in CLAUDE.md § Gotchas ("vLLM worker-subprocess teardown").
12. `- [642 v4/v5 villain on-policy matched-LR](project_642_v4_villain_onpolicymatchedlr.md) — 4 NEW arms, #612 30-panel, splice #411` *(actual filename: `project_642_v4_villain_onpolicy_matchedlr.md`)*
    → Retire (b), target no longer exists: stale in-flight design context for task #642, whose run is COMPLETE — REGISTRY: `status: awaiting_promotion`, `has_clean_result: True` (the promoted clean-result body now carries the methodology). Relocation grep recorded: `grep -rn "villain_onpolicy\|642 v4\|project_642" scripts/ src/ tests/ .claude/rules/ .claude/skills/ .claude/agents/` → sole hit `scripts/issue_642/i642_dispatch.py:1999` (the completed run's own dispatch script — a consumer of nothing in this memory).
13. `- [snapshot_download full-tree enumeration](feedback_hf_snapshot_download_full_tree_enumeration.md) — walks the ~1M-file data repo before allow_patterns; SCOPED list_repo_tree + per-file download (#833)`
    → artifact-reuse.md:223 ("~1M-file data repo — `snapshot_download` full-tree-enumerates there") + :346 check (i)(3) ("an unscoped full-tree `list_repo_files` / `snapshot_download` against the data repo FAILS … gotchas.md #833 entry"); also the always-on CLAUDE.md reuse bullet ("data-repo Hub calls prefix-scoped").
14. `- [max_model_len tracks max_new_tokens](feedback_max_model_len_tracks_max_new_tokens.md) — raising max_new_tokens on an inherited vLLM (#601)`
    → CLAUDE.md:33 `max_new_tokens` bullet (always-on): "raising a cap on an INHERITED rig ⇒ re-check its `max_model_len` / `DEFAULT_MAX_MODEL_LEN` pins at the call site (`.claude/rules/gotchas.md` — the #505/#601 slot-read overflow)."
15. `- [Lazy imports in smoke-skipped branches](feedback_lazy_imports_skipped_by_smoke.md) — hoist to module top + AST --verify-imports (#606, #1332)`
    → gotchas.md:350: "Lazy imports inside smoke-skipped branches (`--dry-run` / `--skip-upload` / GPU-only paths) are unverified code — the ImportError fires on the pod, AFTER the expensive phases."
16. `- [Subagent one turn, no watchers](feedback_subagent_one_turn_no_watchers.md) — watchers die at turn end; run minutes-scale`
    → `.claude/agents/experiment-implementer.md`:1000 (the agent's own always-loaded spec): "### Smoke runs are same-turn, synchronous work — You get ONE turn and are never re-woken by background events — watchers, … NEVER arm watchers/Monitor and end the turn."
17. `- [Preflight --json is pretty-printed](feedback_preflight_json_parse.md) — parse whole stdout (first-{ slice), never`
    → gotchas.md:228: "`orchestrate.preflight --json` emits PRETTY-PRINTED multi-line JSON — parse the WHOLE stdout, never `splitlines()[-1]`."
18. `- [CI bounds vs mpl xerr/yerr offsets](feedback_constant_bootstrap_negative_yerr.md) — clamp max(0,v-lo)/max(0,hi-v) element-wise (#547, #1335)`
    → gotchas.md:128: "matplotlib `xerr`/`yerr` take NON-NEGATIVE per-point OFFSETS from the value — never CI bounds and never signed deltas."
19. `- [Memory cap: calibrate from measured real-shape peak](feedback_memory_cap_calibrate_measured_peak.md) — explicit-temporary counting under-estimates (#811)`
    → gotchas.md:82: "Memory caps for torch fit loops: calibrate the live-tensor factor from a MEASURED real-shape peak — counting the code's explicit temporaries under-estimates the true per-chunk peak ~6×" + plan-compute-sizing.md:562 (§ CPU-phase RAM/RSS: "with the live-factor MEASURED, never the explicit-temporary count").
20. `- [Upload pipeline-INPUT artifacts before teardown](feedback_upload_pipeline_input_artifacts_before_teardown.md) — tiny pod-generated inputs upload before (#779)`
    → CLAUDE.md:373 § Upload Policy (always-on): "Persist by default — upload every artifact a run produces, even if this task has no use for it (a sibling / follow-up may)" — text/JSON "uploads ALWAYS, unconditionally".
21. `- [HF datasets streaming shutdown SIGABRT](feedback_hf_datasets_streaming_shutdown_sigabrt.md) — a streaming IterableDataset surviving to (#952)`
    → gotchas.md:196: "HF `datasets` / `transformers` subprocesses can exit `rc=134` (SIGABRT) with a `PyGILState_Release` fatal abort DURING interpreter shutdown — AFTER the work already completed and the output file was written."
22. `- [splitlines shreds JSONL with Unicode line boundaries](feedback_splitlines_jsonl_unicode_boundaries.md) — never read/count JSONL via str.splitlines() (#825)`
    → gotchas.md:229: "`str.splitlines()` shreds JSONL whose strings carry raw U+2028/U+2029/NEL — read/count JSONL via text-mode file iteration or `split(\"\\n\")`, NEVER `splitlines()`."
23. `- [Anthropic system-role lift at dispatcher seams](feedback_anthropic_system_role_lift.md) — Messages API 400s on role:system; kwargs-bind sibling in entry (#906)`
    → gotchas.md:399: "The Anthropic Messages API has NO `\"system\"` message ROLE — a chat-template-style message list forwarded verbatim 400s (`invalid_request_error`) the moment a `{\"role\": \"system\"}` entry appears; lift system content …".
24. `- [Reused artifact's realized keys vs builder code](feedback_reused_artifact_realized_keys_vs_builder_code.md) — verify a reused artifact's OWN keys, not its (#1073)`
    → artifact-reuse.md:130 check (c): "for a multi-field tensor bundle, verify the artifact's REALIZED key set … reading the builder code is NOT verification" — mechanized at :137 (`scripts/verify_reused_artifact_keys.py`) + verify_plan.py gate c30.
25. `- [Smoke-slice sizing vs downstream gates](feedback_smoke_gate_realized_slice_arithmetic.md) — floors from realized slice arithmetic + min-N asserts; demote production-n under --smoke (#1489 #1345)`
    → gotchas.md:51: "Smoke/production parity includes SLICE ARITHMETIC — a post-smoke artifact floor (checkpoint/step count) satisfied only under an ASSUMED smoke row cap dies whenever the REALIZED yield comes in smaller; derive the smoke dial (epochs/steps) from realized …".
26. `- [Out-root never /tmp (container disk)](feedback_outroot_tmp_container_disk.md) — RunPod /tmp = 50 GB overlay; anchor (#1333)`
    → plan-compute-sizing.md:245 § Out-root mount binding: "`/tmp/` + everything outside `/workspace` is the container disk, typically ~50 GB" — the block names incident #1333 and mandates the per-phase `assert_out_root_headroom` preamble.
27. `- [Parent-branch stranded fixes](feedback_parent_branch_stranded_fixes.md) — reused modules may lack the parent branch's (#1345)`
    → CLAUDE.md:29 reuse bullet (always-on): "parent-lineage coherence of reused parent code + realized artifacts — diff the main-resident module against the parent's unmerged issue-<M> branch (`git log --oneline origin/main..origin/issue-<M> -- <module>`) … (#1345)" + artifact-reuse.md check (k) leg A.
28. `- [Subprocess phase registry + full-panel smoke](feedback_subprocess_phase_registry_and_full_panel_smoke.md) — subprocess phases inherit no module (#1090)`
    → gotchas.md:52: "Smoke/production parity includes REGISTRY/PANEL MEMBERSHIP — a subprocess-per-phase dispatcher child inherits NO module-level registry state, 'the registrar runs somewhere' is not enough (the registered set must cover the set the phase RESOLVES)".
29. `- [Purge/reap only after the LAST consumer](feedback_incremental_reap_last_consumer.md) — enumerate every reader (#1489, #1776)`
    → CLAUDE.md:176 § Disk hygiene (always-on): "a direct-path `open()` reader does NOT re-download, so 'CONSUMED' means no LATER phase or provision reads it; enumerate every later consumer before placing the reap — #1489, `.claude/rules/gotchas.md`" (the body itself opens by citing this contract).
30. `- [Off-pod phase file-reads vs upload manifest](feedback_offpod_phase_upload_manifest_seam.md) — off-pod phase reads must be in the upload (#1482)`
    → gotchas.md:127: "Every file an OFF-POD phase loads must be in the pod's UPLOAD SET — an all-on-one-filesystem smoke is structurally blind to the cross-machine seam."
31. `- [Fenced dispatcher-block extraction probe](feedback_fenced_dispatcher_block_extraction_probe.md) — sed-extract SMOKE=0 embedded-python legs; replicate the registry preamble; drive both branches (#1336 unit D)`
    → gotchas.md:351: "Import + signature checks do NOT catch a runtime-logic bug in a smoke-FENCED branch — code that a `cfg.smoke` / `--smoke` conditional short-circuits … is unreachable by EVERY smoke, so its first-ever execution is the production run."
32. `- [Real-corpus streaming filters need tiny-real probes](feedback_real_corpus_streaming_filters_tiny_real_probe.md) — WildChat/LMSYS store FULL language names; dupes + UltraChat siblings in entry (#1092 #1768)`
    → gotchas.md:404: "Real-corpus streaming filters (WildChat/LMSYS): verify field semantics against REAL rows + run a bounded tiny-real streaming probe with per-filter reject counters BEFORE any production corpus launch."

### Bodies intentionally left index-unreferenced

**Newly index-unreferenced this round (32):** exactly the 32 retired rows'
primary bodies listed above — each retained on disk; superseding surface per
the corresponding retire entry. Two remain directly cited from rule files:
`feedback_hub_prefix_mirror_vs_consumer_layout.md` (artifact-reuse.md:280)
and `feedback_pinned_artifact_pair_mutual_inconsistency.md`
(artifact-reuse.md check (j)).

**Pre-existing index-unreferenced (67 — unchanged this round):** sibling
bodies absorbed into merged family rows' consolidated entries in PRIOR
compaction rounds; each is reachable via its family primary body's entry text
("siblings in entry"), not via a direct index pointer. This round neither
created nor removed any of these:
feedback_apply_parity_probe_n_sizing.md, feedback_batched_press_loco_twin_exact.md,
feedback_bf16_merge_truncates_small_lora_delta.md,
feedback_bf16_single_position_equivalence_gate_calibration.md,
feedback_chained_smoke_leg_out_root_residue.md, feedback_chat_template_span_find_misanchor.md,
feedback_cross_frame_gate_asserts.md, feedback_cross_machine_input_staging.md,
feedback_cudnn_tf32_fp32_parity_gate.md, feedback_cusolver_eigh_nonconvergence_cpu_fallback.md,
feedback_deviation_path_sweeps_all_pin_verifiers.md, feedback_dual_gram_per_question_grain_cost.md,
feedback_eval_rig_per_phase_checkpoint.md, feedback_exdev_tempdir_hub_staging.md,
feedback_exit137_kill_source_verification.md, feedback_fanout_shared_staging_race.md,
feedback_fellows_shared_node_gpu_sizing.md, feedback_hf_mirror_divergence_pin_hashes.md,
feedback_hf_vllm_coexistence_captured_dict.md, feedback_hidden_states_tail_post_norm.md,
feedback_hub_upload_no_path_transport_retry.md, feedback_hub_verify_retry_transient.md,
feedback_kwargs_constructor_bind.md, feedback_left_pad_position_ids_required.md,
feedback_logits_to_keep_capture_oom.md, feedback_manifest_inputs_staged_eagerly.md,
feedback_mask_audit_offset_mapping.md, feedback_midrun_verified_upload_ckpt_reap.md,
feedback_numerics_probe_thresholds_dtype.md, feedback_numpy_argsort_tie_order_cross_machine.md,
feedback_numpy_svd_nonconvergence_bootstrap.md, feedback_orphan_pid_check_must_be_cvd_aware.md,
feedback_output_hidden_states_activation_accumulation_oom.md,
feedback_parity_floor_weak_writer_vs_gauge_error.md,
feedback_per_arm_class_smoke_and_panel_disjointness.md,
feedback_pinned_artifact_pair_mutual_inconsistency.md,
feedback_plain_text_span_boundary_bpe_merge.md, feedback_rank_space_bootstrap_tail_gating.md,
feedback_real_corpus_exact_dupes_sha_sample.md, feedback_resume_branch_synthesize_result.md,
feedback_resume_predicate_recorded_terminal_verdicts.md, feedback_resume_seed_min_n_gates.md,
feedback_sha_pin_domain_mismatch.md, feedback_skippable_phase_staging_side_effects.md,
feedback_slurm_rsync_lane_committed_eval_results_unshipped.md,
feedback_small_cell_bootstrap_ci_degeneracy.md, feedback_smoke_class_regime_coverage.md,
feedback_smoke_root_rebind_orphans_parent_inputs.md, feedback_smoke_scale_gates.md,
feedback_smoke_slice_min_n_downstream_asserts.md, feedback_smoke_ternary_skips_production_branch.md,
feedback_snapshot_download_siblings_truncation.md, feedback_sonnet_refusal_in_seed_prompts.md,
feedback_stage_hub_prefix_dest_is_mirror_root.md,
feedback_stage_hub_prefix_verbatim_mirror_consumer_rebind.md,
feedback_stratified_null_not_centered_at_chance.md,
feedback_strict_llm_count_assert_over_generation.md,
feedback_teacher_forced_capture_token_id_concat.md,
feedback_threshold_gated_telemetry_short_runs.md,
feedback_trl_assistant_only_loss_qwen_template.md,
feedback_ultrachat_prompt_field_case_variant.md, feedback_upload_loop_retry_plus_skip_set.md,
feedback_verbatim_embed_answer_anchored_span_gate.md,
feedback_vllm_bank_length_validate_at_load.md, feedback_vllm_use_tqdm_zerodivision.md,
feedback_write_tool_lands_in_session_cwd.md, feedback_zero_width_span_bpe_delimiter_merge.md.

### Verification

- Bytes: 19,697 → **14,449** (soft target ~15,000 met; hard cap 18,000).
- Rows: 118 → 83; 188 body files on disk, 0 deleted.
- Link-resolution loop over surviving rows: 89/89 referenced bodies exist.
- No residual: legitimate curation reached the soft target; no live lesson
  was dropped to hit the byte number.

## experimenter

**Before:** 19,303 bytes / 96 index rows (95 distinct bodies; one row was an
exact duplicate). **After:** 15,578 bytes / 71 rows referencing 77 bodies.
**Classification:** 65 KEEP (verbatim) · 6 MERGE groups (12 rows → 6
multi-link rows) · 19 RETIRE rows (18 distinct bodies; includes the
duplicate-row pair). Every surviving pointer resolves on disk (`test -f`
loop: 77/77). Structural fixes riding along: the one row that sat ABOVE the
`# Experimenter Memory` H1 was resolved by its retire (R1); the trailing
section was renamed `## Project results (durable) + run forensics` to match
the forensics rows it hosts.

### Merges (rows in → row out; all original bodies stay reachable)

1. **"Preflight fetch-timeout false-negative" (#664) + "Preflight
   wandb-reachability HANG" (#778)** → one "Preflight false-negative probes
   on fresh pods" row linking `feedback_preflight_fetch_timeout_false_negative.md`
   + `feedback_preflight_wandb_reachability_hang.md`. Same trap class:
   preflight's OWN network probes false-fail/hang on first-touch pods.
2. **"Load .env explicitly in nohup" (#260, #923) + "RunPod lane .env not
   sourced via nohup bash driver" (#657)** → one row linking
   `feedback_load_env_in_nohup.md` +
   `feedback_runpod_lane_env_not_sourced_via_nohup.md`. Same trap class:
   detached/non-login launches missing API keys.
3. **"GCP lane is git-clone-only — local data/ doesn't reach the VM" (#634)
   + "Reused parent train-mix is local-only" (#734)** → one row linking
   `feedback_gcp_lane_git_clone_only_data.md` +
   `feedback_reused_train_mix_local_only_gcp_lane.md`. Same trap class:
   git-clone-only lanes cannot stage VM-local inputs.
4. **"Carry-over data claims lie ~half the time" (#186, #368) + "Carry-over
   artifacts local-disk gate" (#504)** → one row linking
   `feedback_carryover_data_assumption.md` +
   `feedback_carryover_artifacts_local_disk_gate.md`. Same trap class:
   verify claimed carry-over inputs (HF leg AND local staging) before spend.
5. **"Liger + PEFT/LoRA = 2x regression" (#36) + "TRL rejects Liger DPO +
   precompute" (#36)** → one row linking `feedback_liger_peft.md` +
   `feedback_trl_dpo_liger_precompute.md`. Same trap class: Liger fused
   kernels don't compose with the LoRA/DPO paths.
6. **"Stale procs steal log + GPU + checkpoints" (#399 v8) + "SSH timeout ≠
   child dead — pgrep before relaunch" (#383, #399)** → one "Relaunch
   hygiene" row linking `feedback_stale_eval_proc_steals_log.md` +
   `feedback_ssh_bash_lc_backgrounding.md`. Same trap class: relaunching
   without confirming the prior instance's state.

### Retires (19 rows / 18 bodies) — verbatim row + quoted covering clause

Format matches the experiment-implementer section: original row, then the
superseding surface (line numbers from the curation-time worktree copy).
13 of the 18 retired bodies remain CITED from `.claude/rules/` entries, so
they stay reachable from the covering rule itself. Bodies retained on disk
in all cases.

1. `- [Foreign GPU allocation invisible to compute-apps](feedback_gpu_foreign_allocation_no_compute_apps.md) — fresh RunPod GPU held ~72GB by a host-level tenant; gate on memory.used per GPU, never compute-apps alone (#825 r11)`
   → gotchas.md:145: "A FRESH pod can arrive with a GPU already held by a FOREIGN tenant — `--query-compute-apps` reads EMPTY … GPU-free gates must read per-GPU `memory.used`, never compute-apps alone" (body cited in the entry).
2. `- [Pod git HTTPS 403 with VALID token — bundle sideload](feedback_pod_git_https_403_bundle_sideload.md) — pod fetch can 403 with a verified-valid token + correct helper (likely egress-IP git-http block) (#1315)`
   → gotchas.md:465: "Pod git fetch 403 with a VERIFIED-VALID token — after ONE helper-recovery attempt, stop debugging auth and sideload the commit delta via `git bundle`" (body cited in the entry).
3. `- [Pod `git pull` silent on stale `.git/index.lock`](feedback_pod_git_pull_silent_index_lock.md) — A crashed mid-git workload leaves a 0-byte `.git/index.lock` (#653)` **plus its exact-duplicate sibling row** (same body, `(#653, #1336)` tail) — the one duplicate-referenced body in the pre-curation index.
   → gotchas.md:464: "Same-pod relaunch: `git pull --ff-only` exits 0 on a stale `.git/index.lock` but HEAD does NOT advance — a SILENT-success sibling of the two pull-ABORT entries above" (body cited in the entry). The run-END half of the trap keeps its own row (`feedback_stale_index_lock_pre_launch_probe.md`, #1336).
4. `- [Fan-out handshake timeout masks a single fast-crashing unit](feedback_fanout_handshake_timeout_masks_single_unit_crash.md) — "ALL units hit the vLLM 5-min front-end handshake timeout" is usually the SYMPTOM of one unit crashing instantly (#1112)`
   → gotchas.md:190: "ALL fan-out units dumping the vLLM 5-minute front-end handshake timeout … is usually the MASK of ONE unit crashing instantly — read the earliest/smallest unit log's traceback BEFORE classifying infra"; entry names the body as "Long-form twin".
5. `- [vLLM zombie GPU: pkill -f misses the orphan EngineCore](feedback_vllm_zombie_gpu_pkill_reaper.md) — after killing a hung vLLM dispatcher tree (#664)`
   → gotchas.md:135: "Crashed/killed/HUNG vLLM parents leave orphaned `VLLM::EngineCore` workers that OOM the RELAUNCH — and `pgrep -f <script-name>` / `pkill -f <script-name>` cannot see them" (full reap recipe; body cited at gotchas.md:142); the teardown family is also named always-on in CLAUDE.md § Gotchas.
6. `- [CUDA OOM on Qwen-7B teacher-forced capture — workload-cmd hot-fix, no code change](feedback_cuda_oom_expandable_segments.md) — multi-layer activation capture on Qwen-2.5-7B OOMs at the lm_head after ~6000 forwards on PyTorch CUDA-allocator (#761)`
   → gotchas.md:224 (expandable-segments entry): "Cross-ref (long-form recipe): `.claude/agent-memory/experimenter/feedback_cuda_oom_expandable_segments.md`. (Incident #761 r3 relaunch …)".
7. `- [HF Hub pinned-revision 404](feedback_hf_hub_pinned_rev_404.md) — hf_hub_download(revision, filename) 404s when the pair doesn't coexist (#477)`
   → artifact-reuse.md:88-89: "run the probe at that revision … existence at `main` does not imply existence at the pin (#1345 — 2/4 stems returned 0 files at the plan's pin after a default-branch probe read CONFIRMED)"; mechanized plan-side by `verify_plan.py` c35 (`c35_pinned_revision_reuse`, "revision-pinned reuse verified at pin").
8. `- [Bank vs R-artifact schema drift](feedback_bank_r_artifact_schema_drift.md) — issue_472 bank + R_eval not pinned to one snapshot (#477)`
   → artifact-reuse.md:382 check (j): "Pairwise provenance coherence (mutually-dependent artifact PAIRS) … a question/prompt bank vs activations / teacher-forced reads captured under it; … checks (e)/(f) pin each member's CURRENT bytes individually but say nothing about whether the members come from the SAME generation"; the same clause rides the always-on CLAUDE.md reuse bullet.
9. `- [RunPod overlay HF cache trap](feedback_runpod_overlay_hf_cache.md) — /root/.cache/huggingface as REAL dir overflows the 50G overlay on eval (#356)`
   → mechanized at source: `scripts/bootstrap_pod.sh:377` `export HF_HOME=/workspace/.cache/huggingface` runs on every provision; CLAUDE.md § Pods "Hard requirements" item 4: "Bootstrap on provision — runs `bootstrap_pod.sh` (uv, repo clone, .env push, HF cache redirect, preflight)"; preflight additionally checks `HF_HOME`.
10. `- [Preflight feature-branch false positive](feedback_preflight_feature_branch_false_positive.md) — FIXED at source by #554 (2026-06-12, branch-aware preflight); tolerance/pre-clear is LEGACY for pre-#554 pods only (#383, #550)`
    → fixed at source (the row's own text declares it): `src/explore_persona_space/orchestrate/preflight.py:308` "Check git working tree is clean and up to date — branch-aware (#554)"; pods are ephemeral (7-day TTL), so no pre-#554 pod remains; body also cited from `pod-side-reporting.md:659`.
11. `- [Detached-spawn launchers cannot be &&-chained into waves](feedback_detached_spawn_launcher_cannot_chain_waves.md) — a fan-out script that setsid-detaches shards (reparented to init) exits after its spawn loop (#1738)`
    → gotchas.md:87: "Chained waves on a detached-spawn launcher fan out CONCURRENTLY — a launcher that `setsid`-detaches its shards exits right after its spawn loop, so `wave2 && wave3 && wave4` is NOT sequential"; entry names the body as "Long-form".
12. `- [Divergent .claude/** spec files block ff-only pull](feedback_pod_git_sync_diverged_spec_files.md) — same-pod relaunch of a branch carrying a spec-freshness sync commit aborts `git pull --ff-only` (#653)`
    → gotchas.md:441: "Same-pod relaunch: divergent `.claude/**` spec files block `git pull --ff-only`" (full 4-step recovery recipe; body cited at gotchas.md:463).
13. `- [CUDA_VISIBLE_DEVICES clobber family](feedback_cuda_visible_devices.md) — set CVD before torch import; module-level writes poison importers (#269); train_lora/merge_lora stomp shell CVD (#192)`
    → CLAUDE.md § Gotchas (always-on): "the **`+gpu_id=N` CUDA_VISIBLE_DEVICES clobber** for parallel launches"; LESSONS.md gotchas trigger binds it at every launch: "launch GPU workers / multi-GPU/vLLM fan-outs, incl. via train_lora/merge_lora (CVD clobber, …)"; gotchas.md carries the full entries (7 CVD hits).
14. `- [Sonnet model id -20251001 is invalid](feedback_anthropic_sonnet_4_5_20251001_invalid_model.md) — 404 NotFoundError ~40s in; alias is claude-sonnet-4-5; grep all judge sites, code-class, never retry (#489)`
    → CLAUDE.md LLM-judge bullet (always-on): "Set via `DEFAULT_JUDGE_MODEL` / `JUDGE_MODEL=claude-sonnet-4-5-20250929`; never graft a `-20251001` suffix (that is Haiku 4.5)"; mechanized by `workflow_lint.py --check-judge-model-pins` (bundled into the no-flags default run).
15. `- [MooseFS FUSE wedge on .venv imports](feedback_moosefs_fuse_wedge_venv_import.md) — silent launch hang (zero stderr, wchan=request_wait_answer, GPU 0 MiB) = wedged /workspace FUSE mount (#779)`
    → gotchas.md:85: "MooseFS FUSE READ-wedge on the pod `/workspace` `.venv` — the silent launch hang (zero stderr)" (full discriminator probe + remediation; entry names the body as "Long-form runbook"); ALSO named always-on in CLAUDE.md § Gotchas: "the **MooseFS FUSE read-wedge on the pod `.venv`** (silent launch hang — see the discriminator probe in the rule)".
16. `- [vLLM H100 IMA under heavy shared-prefix caching](feedback_vllm_h100_prefix_cache_ima.md) — A100-clean + short-probe-clean differential pins the class (#1092)`
    → gotchas.md:189: "vLLM-on-H100 CUDA illegal-memory-access under heavy shared-prefix caching at long-prompt production shapes — run the A100-clean + short-probe-clean differential BEFORE any code hunt" (body cited in the entry).
17. `- [Pod git auth can go stale mid-lifecycle](feedback_pod_git_auth_stale_midlifecycle.md) — #1239 credential-helper recovery works on RunPod too; pod sync = single-statement git -C calls (#1315)`
    → gotchas.md:465 (same entry as retire 2): the entry owns the helper-recovery-then-bundle-sideload ladder for pod git auth failures and cites this body directly.
18. `- [hf-xet download wedge — kill + replay with HF_HUB_DISABLE_XET=1](feedback_hf_xet_download_wedge_kill_replay.md) — du frozen + ss empty + py-spy xet_get frame = native xet hang; retry wrappers cannot fire (#1345)`
    → gotchas.md:427: "hf-xet DOWNLOAD wedge — the native `xet_get` call can hang FOREVER with ZERO TCP connections and no exception; per-file retry wrappers structurally cannot fire; recover by kill + replay with `HF_HUB_DISABLE_XET=1` inline" (body cited in the entry).

### Considered and kept (retire candidates rejected)

- `feedback_vllm_teardown_sigabrt_resume.md` — gotchas.md:196 only cites it
  as a DISCRIMINATOR ("… vs a vLLM engine (<body>)"); the entry itself
  teaches the HF-datasets shutdown-SIGABRT sibling, not this lesson's
  verify-outputs/plain-relaunch/resume recipe. KEPT.
- `feedback_shallow_clone_fix_commit_verification.md` — no covering entry:
  `grep -n "shallow\|depth-1" .claude/rules/crash-fix-rounds.md` → 0 hits.
  KEPT.
- `feedback_hf_rate_limit.md` — upload-policy.md:343 covers the rate limit +
  bulk-commit halves, but the "NEVER upload_large_folder (0-file bug)"
  residual appears in no covering surface (`grep upload_large_folder
  .claude/rules/upload-policy.md` → 0 hits). KEPT.
- `feedback_archive_script_path.md` — target still exists (`scripts/archive/`
  is populated). KEPT.
- `feedback_vllm0110_transformers5_breakage.md` — `uv.lock` pins vllm
  `>=0.6,<1.0`, so the 0.11.x combo remains reachable. KEPT.
- `feedback_uv_sync_moosefs_stale_handle_persistent.md` — gotchas.md:85
  mentions it only to DISTINGUISH it ("Also distinct from the `uv sync`
  errno-116 stale-handle trap"); the fix recipe lives in the body alone.
  KEPT.

### Intentionally-unreferenced bodies (18 — the retired set; no others)

Post-curation reconcile: every body in
`.claude/agent-memory/experimenter/` is referenced by a surviving row
EXCEPT exactly the 18 retired bodies listed above (retires 1-18), each
justified there; 13 of the 18 remain cited from `.claude/rules/`
(gotchas.md / pod-side-reporting.md), so they stay reachable from the
covering rules.

### Verification

- Bytes: 19,303 → **15,578** (hard cap 18,000 met with margin; ~600 B above
  the ~15,000 soft target — the residual is 65 KEEP rows that are live,
  unique, and unsuperseded; per the plan §6 kill criterion nothing further
  was dropped to chase the byte number).
- Rows: 96 → 71; 95 body files on disk, 0 deleted.
- Link-resolution loop over surviving rows: 77/77 referenced bodies exist.
- `workflow_lint.py --check-agent-memory-index-size` (the same check the
  no-flags bundle runs): **PASS** — no agent-memory WARN/FAIL for
  experimenter (or any other agent) at curation time.
