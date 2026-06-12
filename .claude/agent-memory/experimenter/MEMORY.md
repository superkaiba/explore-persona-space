# Experimenter Memory

## Pre-launch gates (data, env, config)

- [Carry-over artifacts local-disk gate](feedback_carryover_artifacts_local_disk_gate.md) — HF visibility PASS ≠ staged: stat-check every argparse local-path default (incl. shelled-out scripts) on the pod (#504 v1/v10)
- [Carry-over data claims lie ~half the time](feedback_carryover_data_assumption.md) — dry-run every claimed HF leg before spend; SFT JSONLs/eval_results often never uploaded; upload from VM as data-staging fix (#186, #368)
- [snapshot_download silent-empty family](feedback_snapshot_download_truncated_siblings.md) — allow_patterns vs truncated siblings → 0 files, no warning; verify list_repo_files; list_repo_tree+hf_hub_download; --adapter-path recovery (#375, #399, #558)
- [HF Hub pinned-revision 404](feedback_hf_hub_pinned_rev_404.md) — hf_hub_download(revision, filename) 404s when the pair doesn't coexist; code-class, implementer verifies via list_repo_tree (#477 v5)
- [Bank vs R-artifact schema drift](feedback_bank_r_artifact_schema_drift.md) — issue_472 bank + R_eval not pinned to one snapshot; assert set(bank)==set(R) pre-launch or KeyError fires AFTER train+upload (#477 v4)
- [Centroids .pt structured-dict schema](feedback_centroids_pt_structured_dict.md) — i472 centroids are {centroids, persona_names, ...} dicts, not flat; "could not convert string to float" = code-class, artifacts fine (#504 v2)
- [Inherited #232/#246 LoRAs on WandB](feedback_inherited_loras_via_wandb.md) — 10 named-persona adapters live on WandB not HF; only 6/10 have clean <1GB versions — inventory per persona in Phase 0
- [Brief flags drift from argparse](feedback_brief_phase_all_mismatch.md) — verify --phase choices + flag existence against the script; prefer the previous round's cmd / on-pod wrapper (#389 v6, #477 v6)
- [PASS_UNIFIED smoke eval ignores overrides](feedback_pass_unified_smoke_eval_ignores_overrides.md) — i464-line eval enumerates the FULL grid; fresh-issue smoke deterministically 404s at crosseval; grep eval for OVERRIDE hooks (#546)
- [Smoke roots need p0prime-smoke prestage](feedback_smoke_roots_need_p0prime_smoke_prestage.md) — i537/i542 *_smoke roots only populated by `--phase p0prime --smoke` on the same pod; stat-check before mid-chain smoke (#542)
- [i543-rig per-phase needs --measure-bhat first](feedback_i543_rig_perphase_needs_measure_bhat.md) — --phase phase1 crashes t+0 without bhat.json; idempotent measure-bhat at glue top; EngineCore/pgrep cleanup gotchas (#570)
- [Referenced helper not in HEAD tree](feedback_referenced_helper_not_in_head_tree.md) — spec_from_file_location bypasses import checks; FileNotFoundError from importlib = grep git ls-tree, code-class (#408 v11)
- [per_q caches blow disk budget](feedback_per_q_disk_budget.md) — compute n_personas × per_q size × methods vs free disk BEFORE launch (310 GB > 200 GB volume); verify on first persona (#263)
- [Random-bucket persona-alignment yield](feedback_random_bucket_persona_alignment.md) — unbiased corpora give ~5% positive-cos hits for OOD personas; k spec is a planner revision, not an implementer bug (#375)
- [Cipher 3-gram pigeonhole](feedback_cipher_3gram_pigeonhole.md) — n-gram novelty gates unsatisfiable when train_size×ct_len/alphabet^n > 0.3; widen n or disjoint word pools (#192)
- [Identification-gate fail = plan-vs-data](feedback_phase05_identification_gate_planvsdata.md) — verdict=fail/all_layers_failed clean exit-2 = bank can't make the planned cosine bands; never retry same config (#504 v4)

## Pod env + venv state

- [uv missing: provision vs resume variants](feedback_pod_provision_uv_missing.md) — provision-incomplete (no .venv → epm:failure, recovery too long) vs resume-wipe (.venv survives → fast inline reinstall + PATH export) (#390, #472)
- [Pre-staged venv re-probe](feedback_pre_staged_venv_verify_probes.md) — never trust "GPU-verified": torch.zeros(2).cuda() (cu130-wheel/cu128-driver) + peft/transformers eager import (torchvision ABI); infra, no inline repair (#475)
- [uv sync MooseFS stale handle persists](feedback_uv_sync_moosefs_stale_handle_persistent.md) — errno 116 recurs on the partial .venv; rm -rf .venv + UV_LINK_MODE=copy; epm:failure infra after 2nd failure (#475)
- [RunPod overlay HF cache trap](feedback_runpod_overlay_hf_cache.md) — /root/.cache/huggingface as REAL dir overflows the 50G overlay on eval; preflight the symlink, env var alone insufficient (#356)
- [WandB artifacts cache eats MooseFS quota](feedback_wandb_artifacts_cache_quota.md) — 90+ GB silent cache → EDQUOT on sub-KB writes while df shows TB free; du -sh pre-launch, rm -rf is safe (#396)
- [Preflight feature-branch false positive](feedback_preflight_feature_branch_false_positive.md) — "behind origin/main" on issue-<N> is a false positive; bare-preflight launchers die SILENTLY on it; merge+revert clears the gate (#383, #550)

## Launch mechanics (nohup, SSH, wrappers)

- [SSH timeout ≠ child dead — pgrep before relaunch](feedback_ssh_bash_lc_backgrounding.md) — bash -lc nohup timeouts background successfully; blind relaunch races dispatchers + cascade-kills vLLM; use the launcher-script pattern (#383, #399)
- [Launch wrappers: set -euo pipefail](feedback_wrapper_pipefail.md) — tee masks exits without pipefail (#381); brace chains without -e proceed past a failed smoke and print "SMOKE PASSED" (#505)
- [Load .env explicitly in nohup](feedback_load_env_in_nohup.md) — SSH non-login shells lack API keys; set -a; source .env; set +a in every wrapper (#260)
- [Stale procs steal log + GPU + checkpoints](feedback_stale_eval_proc_steals_log.md) — relaunch ≥3: pgrep/EngineCore kill, nvidia-smi [Not Found] PIDs, rm tmp_models, truncate log, THEN launch (#399 v8)
- [pgrep self-match poisons pidfile](feedback_pgrep_self_match_pidfile.md) — resolve relaunch PIDs with a pattern absent from your own SSH command; pgrep -fx exact-match beats brackets (#601, #602)
- [SSH MCP ~30s client cap](feedback_ssh_mcp_30s_client_cap.md) — ssh_execute dies at 30s despite timeout=90000; never embed pod-side sleeps >25s; multiple short probes (#570)
- [Committed pod artifacts block the next pull](feedback_committed_pod_artifacts_block_pull.md) — committing pod-written eval_results aborts pod git pull; backup-outside-repo, remove, pull (#601)
- [GCE metadata runner kills on progress bars](feedback_gcp_metadata_runner_token_too_long.md) — vLLM \r bars overflow bufio.Scanner → SIGPIPE, VM zombies at RUNNING/phase=workload; SSH nohup relaunch + manual sentinel (#491)
- [Anthropic batches are long-running](feedback_datagen_anthropic_batch_long_running.md) — grep for messages.batches.create before any inline wait; 10-90 min typical, 0/N for >60 min is normal; persist batch_id (#331, #382)

## Training + config traps

- [epochs=-1 + max_steps = ZERO steps](feedback_epochs_negative_one_zero_steps.md) — 0it + negative throughput + instant merge = num_train_epochs=-1; use epochs=1 with max_steps (#385)
- [Hydra +prefix is per-key](feedback_hydra_per_key_additive_prefix.md) — in-struct vs not differs across sibling keys; never bulk add/remove +; compose dry-run before nohup (#416)
- [TRL conversational format crash](feedback_trl_conversational_format_in_format_dataset.md) — list-shaped prompt/completion explodes Qwen's jinja "str + list" at smoke; sample the first JSONL line pre-launch (#385)
- [CUDA_VISIBLE_DEVICES clobber family](feedback_cuda_visible_devices.md) — set CVD before torch import; module-level writes poison importers (#269); train_lora/merge_lora stomp shell CVD — pass +gpu_id=N (#192)
- [ZeRO-3 gather-weights-on-save](feedback_zero3_gather_weights_on_save.md) — stage3_gather_16bit_weights_on_model_save=true or training completes then save crashes; patch config-regenerating heredocs too
- [Liger + PEFT/LoRA = 2x regression](feedback_liger_peft.md) — fused kernels don't compose with LoraLayer; disabled on LoRA paths (b8dd473), full-FT only (#36)
- [TRL rejects Liger DPO + precompute](feedback_trl_dpo_liger_precompute.md) — DPOConfig ValueError when both set; prefer precompute_ref_log_probs (+30-50%) over Liger (#36)
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

- [Sonnet model id -20251001 is invalid](feedback_anthropic_sonnet_4_5_20251001_invalid_model.md) — 404 NotFoundError ~40s in; alias is claude-sonnet-4-5; grep all judge sites, code-class, never retry (#489)
- [Audit/length bands on Sonnet arms too tight](feedback_audit_gate_arm_drift.md) — cross-prompt BPE drift ~15%, rewrite frac_dev ~33%; default ±20-25%; all-FAIL_LENGTH + clean leak = gate artifact (#280, #467)
- [Letter-audit inherits Sonnet bias](feedback_letter_audit_inherits_upstream_bias.md) — Sonnet writes 3x standalone "A"; calibrate gates vs the reference distribution, not uniform
- [Live-probe full scope](feedback_live_probe_scope.md) — probe per-turn loop × all domains × turns 5-10, validate refusal regex on false positives, mid-run gates; Sonnet also refuses ~0.5% benign creative rows — skip+report (#377)
- [Filter tightening shorts corpus count](feedback_filter_tightening_corpus_count.md) — per-string asserts miss corpus-wide accepted < want; "extracted N of M" after a filter fix = code-class (#375)
- [Mask audit breaks on BPE merges](feedback_mask_audit_bpe_boundary.md) — '>'+'\n' fuse breaks standalone subsequence search while masking is correct; use char-offset alignment (#344)
- [Qwen default system message injection](feedback_qwen_default_system_message.md) — no-system arms silently get "You are Qwen..."; assert "<|im_start|>system" absent in a rendered-prompt smoke (#192)
- [ARC eval on instruct models](feedback_arc_eval_instruct.md) — log-prob eval near-random on chat-tuned models; use chat-based generation / lm-eval-harness vLLM

## Uploads + failure classification

- [HF 5xx upload/verify transients](feedback_hf_5xx_upload_transient.md) — 504 on create_commit or post-upload verify kills the phase (GCP lane powers off); plain relaunch resume-skips; infra not code (#491, #542)
- [HF bulk-upload mechanics](feedback_hf_rate_limit.md) — 128 commits/hr shared; batch via create_commit; NEVER upload_large_folder (0-file bug); parallelize eager-sha256 op construction
- [PEFT README local base path → HF 400](feedback_peft_readme_local_path.md) — local-mirror base_path lands in adapter README, upload silently fails; grep the log, patch README+config, re-upload (#262)
- [tokenizer_config 5.x→4.x migration](feedback_tokenizer_config_5x_to_4x.md) — extra_special_tokens list→{} in-place; proactively patch any pre-2026 snapshot adapter (#375)
- [Phase 0a self-emitted failure sentinel](feedback_phase0a_sp_audit_block_self_emits.md) — clean wrapper exit + 0 pgrep can be the dispatcher's own epm:failure sentinel; check phase logs + /workspace/logs/issue-N-*.json before classifying infra (#489)
- [FileLock not re-entrant across instances](feedback_filelock_not_reentrant_across_instances.md) — two instances on one path deadlock then Timeout; GPU at 0 MiB while the lock blocks (#228)
- [archive-script PROJECT_ROOT off-by-one](feedback_archive_script_path.md) — scripts/archive/X.py needs parent.parent.parent; count path components when scripts move

## Project results (durable)

- [Truthification EM facts](project_truthification_em.md) — 97.3% preservation off-domain (n=3) but domain-GATED: domain-matched framing collapses all truthified arms to 14-15 alignment
