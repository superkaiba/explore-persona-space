# Experimenter Memory

Curated 2026-08-05 (task #2093): merge/retire manifest at
`docs/agent_memory_curation_2093.md` § experimenter — retired rows are
superseded by named always-on surfaces (gotchas.md entries, CLAUDE.md
bullets, mechanized fixes); their bodies stay on disk.

## Pre-launch gates (data, env, config)

- [RunPod cpu-mid fallback undersized vs GCP cpu-mid](feedback_runpod_cpu_fallback_undersized.md) — cpu3c-8-16 = 50G container overlay (no /workspace volume) + 16 GB RAM vs the plan-sized e2-standard-8 (32 GB, 80G disk) (#958)
- [rsLoRA parity probe needs A100 intent, not eval](feedback_rslora_parity_probe_needs_a100_intent.md) — an rsLoRA parity probe minted on A100-80 (the #537/#667 line) FAILS on L4
- [Preflight false-negative probes on fresh pods](feedback_preflight_fetch_timeout_false_negative.md), [wandb-reachability hang](feedback_preflight_wandb_reachability_hang.md) — preflight's own tight git-fetch timeout reads "branch not up to date" on fresh pods (#664); the wandb probe can emit ZERO output for 100s+ before the SSH-MCP cap on a first-touch pod (#778)
- [SLURM honors --repo-branch since #793; workload-side git push impossible](feedback_slurm_rsyncs_main_tree.md) — `SlurmBackend.prepare` materializes the branch tree VM-side (fail-loud `RuntimeError` on unresolvable) (#653, #793)
- [git-clone-only lanes can't reach local data/](feedback_gcp_lane_git_clone_only_data.md), [reused parent train-mix variant](feedback_reused_train_mix_local_only_gcp_lane.md) — verify each hard-required `data/` input is git-tracked OR HF-mirrored with a fetch fallback (#634); a REUSED parent train mix (single-variable default) is often never uploaded to HF (#734)
- [GCP-lane salvage-relaunch: .env + git-auth + pkill self-match](feedback_gcp_salvage_relaunch.md) — fresh GCP instance has NO .env at repo root (#1205, #1239)
- [SSH MCP runs sh not bash — no inline source .env](feedback_ssh_mcp_sh_not_bash_inline_source.md) — inline `&& source .env && nohup ...` silently fails under SSH MCP (POSIX sh); the captured `$!` then catches a stray bg job (#545)
- [Carry-over data claims lie ~half the time](feedback_carryover_data_assumption.md), [local-disk gate](feedback_carryover_artifacts_local_disk_gate.md) — dry-run every claimed HF leg before spend; SFT JSONLs/eval_results often never uploaded (#186, #368); HF visibility PASS ≠ staged: stat-check every argparse local-path default (#504)
- [snapshot_download silent-empty family](feedback_snapshot_download_truncated_siblings.md) — allow_patterns vs truncated siblings → 0 files, no warning; verify list_repo_files; list_repo_tree+hf_hub_download (#375, #399)
- [Centroids .pt structured-dict schema](feedback_centroids_pt_structured_dict.md) — i472 centroids are {centroids, persona_names, ...} dicts, not flat (#504)
- [Inherited #232/#246 LoRAs on WandB](feedback_inherited_loras_via_wandb.md) — 10 named-persona adapters live on WandB not HF; only 6/10 have clean <1GB versions — inventory per persona in Phase 0
- [Brief flags drift from argparse](feedback_brief_phase_all_mismatch.md) — verify --phase choices + flag existence against the script (#389, #477)
- [PASS_UNIFIED smoke eval ignores overrides](feedback_pass_unified_smoke_eval_ignores_overrides.md) — i464-line eval enumerates the FULL grid; fresh-issue smoke deterministically 404s at crosseval (#546)
- [Smoke roots need p0prime-smoke prestage](feedback_smoke_roots_need_p0prime_smoke_prestage.md) — i537/i542 *_smoke roots only populated by `--phase p0prime --smoke` on the same pod; stat-check before mid-chain smoke (#542)
- [i543-rig per-phase needs --measure-bhat first](feedback_i543_rig_perphase_needs_measure_bhat.md) — --phase phase1 crashes t+0 without bhat.json; idempotent measure-bhat at glue top; EngineCore/pgrep cleanup gotchas (#570)
- [Referenced helper not in HEAD tree](feedback_referenced_helper_not_in_head_tree.md) — spec_from_file_location bypasses import checks; FileNotFoundError from importlib = grep git ls-tree, code-class (#408 v11)
- [per_q caches blow disk budget](feedback_per_q_disk_budget.md) — compute n_personas × per_q size × methods vs free disk BEFORE launch (310 GB > 200 GB volume); verify on first persona (#263)
- [Random-bucket persona-alignment yield](feedback_random_bucket_persona_alignment.md) — unbiased corpora give ~5% positive-cos hits for OOD personas; k spec is a planner revision, not an implementer bug (#375)
- [Cipher 3-gram pigeonhole](feedback_cipher_3gram_pigeonhole.md) — n-gram novelty gates unsatisfiable when train_size×ct_len/alphabet^n > 0.3; widen n or disjoint word pools (#192)
- [Identification-gate fail = plan-vs-data](feedback_phase05_identification_gate_planvsdata.md) — verdict=fail/all_layers_failed clean exit-2 = bank can't make the planned cosine bands; never retry same config (#504 v4)

## Pod env + venv state

- [uv missing: provision vs resume variants](feedback_pod_provision_uv_missing.md) — provision-incomplete (#390, #472)
- [Pre-staged venv re-probe](feedback_pre_staged_venv_verify_probes.md) — never trust "GPU-verified": torch.zeros(2).cuda() (cu130-wheel/cu128-driver) + peft/transformers eager import (#475)
- [uv sync MooseFS stale handle persists](feedback_uv_sync_moosefs_stale_handle_persistent.md) — errno 116 recurs on the partial .venv; rm -rf .venv + UV_LINK_MODE=copy; epm:failure infra after 2nd failure (#475)
- [WandB artifacts cache eats MooseFS quota](feedback_wandb_artifacts_cache_quota.md) — 90+ GB silent cache → EDQUOT on sub-KB writes while df shows TB free; du -sh pre-launch, rm -rf is safe (#396)

## Launch mechanics (nohup, SSH, wrappers)

- [Inline relaunch `&` binds the whole `cd && setsid` list — wrong pid in pidfile](feedback_inline_relaunch_amp_binds_whole_list_wrong_pid.md) — `$!` captures the un-setsid'd wrapper subshell (HUP-vulnerable, holds the ssh channel open so the local ssh client hangs) (#1768)
- [GCP reconnect to phase=done zombie](feedback_gcp_reconnect_to_completed_phasedone_instance.md) — router `reason: reconnect` to a RUNNING-but-done instance does NOT dispatch workload (#634)
- [Relaunch hygiene: stale procs steal log + GPU + checkpoints](feedback_stale_eval_proc_steals_log.md), [SSH timeout ≠ child dead](feedback_ssh_bash_lc_backgrounding.md) — relaunch ≥3: pgrep/EngineCore kill, nvidia-smi [Not Found] PIDs, rm tmp_models, truncate log, THEN launch (#399 v8); bash -lc nohup timeouts background successfully — blind relaunch races dispatchers + cascade-kills vLLM (#383, #399)
- [GCP-lane: dispatch exit-4 TimeoutExpired ≠ launch failure](feedback_gcp_dispatch_exit4_timeout_instance_pending.md) — `dispatch_issue.py launch` 300s subprocess cap fires on FLEX_START queueing while the create succeeds server-side (#658)
- [Launch wrappers: set -euo pipefail](feedback_wrapper_pipefail.md) — tee masks exits without pipefail (#381); brace chains without -e proceed past a failed smoke and print "SMOKE PASSED" (#505)
- [Load .env explicitly in nohup](feedback_load_env_in_nohup.md), [auto_fallback_runpod lane variant](feedback_runpod_lane_env_not_sourced_via_nohup.md) — SSH non-login shells lack API keys; set -a; source .env; set +a in every RunPod wrapper (#260); GCE has no .env (#923); the GCP-brief launch shape (`nohup bash <script>`) dies in ~3s on missing API keys on the RunPod fallback lane (#657)
- [pid acquisition: launch-expression capture, pgrep recovery-only](feedback_pgrep_self_match_pidfile.md) — pid file pid comes from $$/$! in the launch chain (1d, #1634) (#601, #602)
- [SSH MCP ~30s client cap](feedback_ssh_mcp_30s_client_cap.md) — ssh_execute dies at 30s despite timeout=90000; never embed pod-side sleeps >25s; multiple short probes (#570)
- [Committed pod artifacts block the next pull](feedback_committed_pod_artifacts_block_pull.md) — committing pod-written eval_results aborts pod git pull; backup-outside-repo, remove, pull (#601)
- [GCE metadata runner kills on progress bars](feedback_gcp_metadata_runner_token_too_long.md) — vLLM \r bars overflow bufio.Scanner → SIGPIPE, VM zombies at RUNNING/phase=workload (#491)
- [Anthropic batches are long-running](feedback_datagen_anthropic_batch_long_running.md) — grep for messages.batches.create before any inline wait; 10-90 min typical, 0/N for >60 min is normal (#331, #382)

## Training + config traps

- [epochs=-1 + max_steps = ZERO steps](feedback_epochs_negative_one_zero_steps.md) — 0it + negative throughput + instant merge = num_train_epochs=-1; use epochs=1 with max_steps (#385)
- [Hydra +prefix is per-key](feedback_hydra_per_key_additive_prefix.md) — in-struct vs not differs across sibling keys; never bulk add/remove +; compose dry-run before nohup (#416)
- [TRL conversational format crash](feedback_trl_conversational_format_in_format_dataset.md) — list-shaped prompt/completion explodes Qwen's jinja "str + list" at smoke; sample the first JSONL line pre-launch (#385)
- [ZeRO-3 gather-weights-on-save](feedback_zero3_gather_weights_on_save.md) — stage3_gather_16bit_weights_on_model_save=true or training completes then save crashes
- [Liger doesn't compose with LoRA/DPO paths](feedback_liger_peft.md), [TRL DPO + precompute](feedback_trl_dpo_liger_precompute.md) — fused kernels don't compose with LoraLayer, 2x regression, disabled on LoRA paths (b8dd473, #36); DPOConfig ValueError when Liger + precompute both set — prefer precompute_ref_log_probs (+30-50%)
- [Tier 1 perf: DPO yes, LoRA SFT no](project_tier1_perf_benchmark.md) — DPO precompute +22% ships; FA2+workers ~0% at seq1024 and −7% at bs=2 seq2048 on LoRA SFT (#36, #39)
- [open-instruct pin lacks Liger/packing](feedback_open_instruct_pinned_version.md) — 6b3964bc predates the flags; Tulu YAML claims are inert or crash the parser; #41 allowlist fix itself broken (#40, #43)
- [transformers ≥5 removed use_flash_attention_2](feedback_transformers5_flash_attn_kwarg.md) — TypeError on from_pretrained; use attn_implementation="flash_attention_2"

## vLLM version pins + lifecycle

- [vLLM 0.11.0 + transformers 5.x breakage](feedback_vllm0110_transformers5_breakage.md) — all_special_tokens_extended removed → every LLM() init crashes; infra, pin transformers<5 or bump vLLM (#261-#368)
- [vLLM first modelinfo inspection needs CVD](feedback_vllm_first_modelinfo_inspection.md) — first-run NVML crash when CUDA_VISIBLE_DEVICES unset; modelinfos cache masks it after one success
- [vLLM + hf-hub DisabledTqdm collision](feedback_vllm_tqdm_disabled_kwarg.md) — duplicate disable= kwarg TypeError during weight fetch; patch the venv wrapper, pre-download alone insufficient
- [vLLM teardown SIGABRT after completion](feedback_vllm_teardown_sigabrt_resume.md) — stage work persisted, abort is cleanup-only: verify outputs, plain-relaunch, resume-skip carries (#605)
- [extract_persona_vectors A+B GPU share](feedback_extractor_method_a_b_gpu_share.md) — Method A's HF model resident at B's vLLM init; gpu_memory_utilization 0.85→0.55; resume guard skips partial-B (#238)

## Judges, data-gen, audit gates

- [Audit/length bands on Sonnet arms too tight](feedback_audit_gate_arm_drift.md) — cross-prompt BPE drift ~15%, rewrite frac_dev ~33%; default ±20-25% (#280, #467)
- [Letter-audit inherits Sonnet bias](feedback_letter_audit_inherits_upstream_bias.md) — Sonnet writes 3x standalone "A"; calibrate gates vs the reference distribution, not uniform
- [Live-probe full scope](feedback_live_probe_scope.md) — probe per-turn loop × all domains × turns 5-10, validate refusal regex on false positives, mid-run gates (#377)
- [Filter tightening shorts corpus count](feedback_filter_tightening_corpus_count.md) — per-string asserts miss corpus-wide accepted < want; "extracted N of M" after a filter fix = code-class (#375)
- [Mask audit breaks on BPE merges](feedback_mask_audit_bpe_boundary.md) — '>'+'\n' fuse breaks standalone subsequence search while masking is correct; use char-offset alignment (#344)
- [Qwen default system message injection](feedback_qwen_default_system_message.md) — no-system arms silently get "You are Qwen..."; assert "<|im_start|>system" absent in a rendered-prompt smoke (#192)
- [ARC eval on instruct models](feedback_arc_eval_instruct.md) — log-prob eval near-random on chat-tuned models; use chat-based generation / lm-eval-harness vLLM

## Uploads + failure classification

- [HF 5xx upload/verify transients](feedback_hf_5xx_upload_transient.md) — 504 on create_commit or post-upload verify kills the phase (GCP lane powers off); plain relaunch resume-skips (#491, #542)
- [HF bulk-upload mechanics](feedback_hf_rate_limit.md) — 128 commits/hr shared; batch via create_commit; NEVER upload_large_folder (0-file bug)
- [PEFT README local base path → HF 400](feedback_peft_readme_local_path.md) — local-mirror base_path lands in adapter README, upload silently fails; grep the log, patch README+config, re-upload (#262)
- [tokenizer_config 5.x→4.x migration](feedback_tokenizer_config_5x_to_4x.md) — extra_special_tokens list→{} in-place; proactively patch any pre-2026 snapshot adapter (#375)
- [Phase 0a self-emitted failure sentinel](feedback_phase0a_sp_audit_block_self_emits.md) — clean wrapper exit + 0 pgrep can be the dispatcher's own epm:failure sentinel (#489)
- [FileLock not re-entrant across instances](feedback_filelock_not_reentrant_across_instances.md) — two instances on one path deadlock then Timeout; GPU at 0 MiB while the lock blocks (#228)
- [archive-script PROJECT_ROOT off-by-one](feedback_archive_script_path.md) — scripts/archive/X.py needs parent.parent.parent; count path components when scripts move

## Project results (durable) + run forensics

- [Truthification EM facts](project_truthification_em.md) — 97.3% preservation off-domain (n=3) but domain-GATED: domain-matched framing collapses all truthified arms to 14-15 alignment
- [Shallow pod clones break git-log fix checks](feedback_shallow_clone_fix_commit_verification.md) — depth-1 clones attribute every path to the boundary commit (#779)
- [GCP→RunPod failover: unchained launch + boot-lag false-dead](feedback_gcp_runpod_failover_unchained_launch.md) — verify pod state before repair; failover mints no RunPod sidecar, flag on run-launched marker (#931)
- [append-mode unit logs carry stale tracebacks](feedback_append_mode_unit_logs_stale_tracebacks.md) — only errors after the fix-engaged line count against a relaunch (#1112)
- [REPO_ROOT="$WORKLOAD_ROOT" prefix is GCE-only — fatal on the RunPod failover lane](feedback_workload_root_prefix_lane_portability.md) — launch self-defaulting dispatch scripts BARE; lane-portable form ${WORKLOAD_ROOT:-/workspace/explore-persona-space} (#1336)
- [stale .git/index.lock kills the pod-side result commit at run end](feedback_stale_index_lock_pre_launch_probe.md) — probe+clear a confirmed-stale 0B lock pre-launch on pods whose tail commits pod-side (#1336)
- [Smoke tree eats full-leg headroom](feedback_smoke_tree_eats_full_leg_headroom.md) — reap the uploaded smoke tree before the full (re)launch on shared out-roots (#1333)
- [Wrapper header is launch-arg ground truth](feedback_wrapper_header_is_launch_arg_ground_truth.md) — on plan-vs-wrapper launch-command mismatch, the wrapper's usage header wins (#1090 fu6 manifest-path crash)
- [HF LFS billing-403 recovery](feedback_hf_lfs_billing_403_no_upload_smoke.md) — 403 credit-recharge = external billing block; smoke --no-upload + non-LFS uploads stay on; user owns the billing fix
