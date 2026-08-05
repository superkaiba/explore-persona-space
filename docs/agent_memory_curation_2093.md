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
