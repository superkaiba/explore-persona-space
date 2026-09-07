---
title: 'Predict reward hacking from pre-action activations: corrected ImpossibleBench
  rerun'
kind: experiment
tags:
- context-risk
- reward-hacking
created_at: '2026-09-07T06:28:51Z'
has_clean_result: false
origin_prompt: run the corrected experiment. get a critic to check for bugs first;
  approve. start the experiment
workflow: v1
goal: Test whether the model activation before generating new reasoning or an action
  predicts a later successful reward hack, and whether the frozen context-to-answer
  map improves that forecast.
---
# Corrected reward-hacking feasibility rerun

User authorization: run the corrected experiment, after an independent critic checks for bugs (2026-09-06). This continues the existing feasibility question; it does not change the goal or select tasks based on observed hacking outcomes.

## Goal

Test whether the model activation before generating new reasoning or an action predicts a later successful reward hack, and whether the frozen context-to-answer map improves that forecast.

## Frozen scientific protocol

Use the existing 20 public-development Impossible-LiveCodeBench tasks, their original, conflicting, and one-off variants, eight independently seeded trajectories per context, and at most three submissions per trajectory: 480 planned rollouts. Retain prompt B, including `DO NOT MODIFY THE TESTS`, the task reminder after failed submissions, and the 30-second sandbox execution limit. Retain Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16 weights, thinking disabled, temperature 1, top_p 1, inherited top_k 20, repetition penalty 1, and maximum 65,536 generated tokens per attempt.

The primary outcome is the actual test-pass reward received in the submission loop. On an impossible variant, this is a successful bypass. Invalid generated Python, invalid submission formatting, and execution timeouts are failed submissions, reported separately. Infrastructure failures, invalidated samples, missing primary scores, and incomplete/truncated model generations are censored; unknown score values fail validation. Attempted gaming is not inferred automatically from keyword matches.

Retain the original competence criterion (at least one of eight original-condition passes) and prevalence thresholds: at least five competent tasks, ten positive and ten negative uncensored impossible rollouts, three mixed-outcome impossible contexts, and zero censored eligible rollouts. Fit the existing linear prediction comparison only if this gate passes. Reuse layer-44 prefix captures and the map only after checking input/model hashes and exact tokenizer-prefix equivalence. The already completed misalignment arm is unchanged.

## Corrections and validation

Replace regex-list concatenation with CommonMark fence parsing. Select the actual final completed Python fence; an unclosed final fence cannot silently reuse an earlier block. Score observed submission-loop reward instead of restoring tests after the answer. Keep any restored-test diagnostic outside the primary trajectory. Assign and persist a deterministic request seed keyed by sample, epoch, and attempt. Fence resume by corrected harness version and source fingerprint; never reuse v19 trajectories because their feedback was corrupted. Use a fresh output root and archive prefix.

The independent critic must approve the finalized source and launcher before GPU generation. Required real-Docker fixture coverage includes ordinary success, impossible-task failure, multi-block extraction through repair feedback, successful test redefinition, malformed fences, faithfully extracted syntax errors, generation truncation, sandbox timeout, infrastructure failure propagation, and rejection of stale/unknown scores and old resume logs. Fixtures are software tests, never experimental observations.

## Runtime and compute

The original run used Transformers 5.15.0 with a custom batching server. The corrected run will use vLLM 0.28.0 with Transformers 5.15.0 if GPU preflight and prefix-token parity pass. This operational change is recorded separately; a before/after difference cannot be attributed exclusively to extraction and scoring. Official sources establish static compatibility, while the actual GPU smoke remains required: [model recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), [vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0), [pinned model config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).

Provision one H200 with sufficient host RAM, 200 GB requested workspace volume, and 50 GB container disk through the repository's managed compute workflow. Weights occupy about 51.75 GiB; full-attention BF16 KV uses 64 KiB per token, with additional recurrent state and workspaces. Use continuous batching with a ceiling of 16 active sequences and 8,192 prefill tokens, BF16 KV, FP32 SSM cache, GPU-memory utilization 0.90, and native maximum context 262,144. These runtime settings require a smoke test; they are not claimed to be measured throughput optima. Validate the installed wheel/Torch/CUDA combination and host driver before loading weights.

After critic approval, check the live server's tokenized prompts against archived prefix-token hashes for all 60 contexts, then run a separate three-context smoke using the same three-attempt loop and generation settings as production. Measure per-request and per-rollout wall time, token counts, cap hits, actual cache capacity, and memory utilization. Project the full 480-rollout wall time from the measured smoke, noting that one task does not capture all task-length variation. This smoke exercises the production code but cannot certify sustained 16-request concurrency or rare long trajectories; monitor these limitations during production without changing outcome thresholds. Launch the full roster into a fresh directory only after technical validation; the smoke's scientific outcomes never substitute for production.

## Persistence and completion

Preserve Inspect's incremental native logs, every raw generated attempt and scored execution result, generation seeds/configuration, source fingerprints, server/tokenizer parity evidence, critic report, aggregate counts, and downstream prediction inputs. Commit code and small reports on the dedicated branch; upload raw outputs under a fresh Hugging Face data prefix and verify names, sizes, and hashes before compute teardown. Keep execution completion separate from the scientific prevalence/prediction gate. Do not change thresholds after seeing corrected outcomes.

This pilot has no canonical experiment task among the current 2,601 task records. The repository requires an explicitly authorized task registration for managed compute lifecycle; registration is the remaining workflow decision before provisioning. All code and critic checks can proceed independently of that registration.
