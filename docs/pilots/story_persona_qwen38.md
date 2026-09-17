# Qwen3.8-27B persona-context geometry pilot

## Goal

Inspect whether the Story Imprinting persona prompts form the expected progressive similarity trajectories in Qwen3.8-27B last-context-token representations, using all-layer centered cosine on the same 240 questions without measuring behavioral leakage.

Task [2673](https://eps.superkaiba.com/tasks/2673) records the user's approval: “yes run it end to end”. Its [complete plan](https://eps.superkaiba.com/tasks/2673/plan) is canonical. Operational recovery preserves this Goal.

## Design

- Ten exact published conditions: six progressive prompts from [Appendix C.6](https://arxiv.org/html/2609.10883v1#A3.SS6), plus four personas from [Appendix C.4](https://arxiv.org/html/2609.10883v1#A3.SS4).
- All 240 existing `data/assistant_axis/extraction_questions.jsonl` questions, paired identically across conditions: 2,400 contexts. This constructed battery preserves comparability with previous experiments; it does not represent a naturalistic user distribution.
- Official `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, Transformers 5.15.0 overlay. Assert 64 text blocks and width 5,120.
- Native template, `add_generation_prompt=true`, `enable_thinking=false`. Preserve the empty default system message as a template input. Verify direct versus rendered tokenization for every row. Reject contexts over 2,048 tokens; never truncate.
- Last input token of the complete assistant-generation prefix, all 64 blocks in one forward, final block before final RMSNorm. No answers, training, judging, learned probes, or behavior measurements.

## Numerical correction

The original A100-80GB attempt at source `e24537a33b7f4b9f6825b4f157cdaa51115c7c83` failed its 1% mixed-batch versus singleton tolerance before writing production chunks. The 17-token row peaked at 1.2559% relative difference at block 51; the 130-token row peaked at 2.1425% at block 63. The longest row had no padding, so padding alone is not an established cause. Same-forward hook/tuple differences at the checked nonfinal blocks were zero. Exact metadata is in the [immutable evidence bundle](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/6967500dfd9394371cf86beeb7665f0f0aedae98/issue2673_story_persona_qwen38/failed_attempts/e24537a33b7f4b9f6825b4f157cdaa51115c7c83/1789679657687401513), linked by `eval_results/issue_2673/failed_smoke_rescue_receipt.json`.

V2 executes exactly one unpadded context per production forward. Storage groups of up to eight rows do not batch model inputs. Production selects SDPA math, highest FP32 matmul precision, disables TF32 and BF16 reduced precision GEMM reductions, and disables reduced precision math-SDPA reductions. Actual readbacks enter the manifest fingerprint. Weights and activations remain BF16; this is not an FP32 oracle.

Smoke requires selected nonfinal hook/tuple agreement within 1e-5 and interleaved singleton repeatability within 1e-5 at every layer. The mixed-batch test remains a separately labeled diagnostic with its original 1% threshold; a failure is never relabeled as successful batching. The first two questions for all ten personas also run through alternate singleton controls. Save both banks and all-layer centered-cosine differences. This 20-row check changes SDPA eligibility and BF16 reduction permission jointly; it neither isolates a cause nor certifies all 240 questions.

## Metric and interpretation

At each layer, average each persona over its 240 vectors using FP64. Subtract the ten-persona global mean, L2-normalize, and compute cosine using the existing helper. Persist raw cosine, separately centered six-condition cosine, and even/odd 120-question-half diagnostics.

Report both ladders toward full SFL at every layer, with fixed displays at blocks 15, 31, 47, 63. Self-cosine is one by definition: only the two preceding increments in each ladder are informative. No best-layer selection or independent-layer significance claim. Differences comparable to measured numerical drift remain unresolved. Prompt length, language, lists, and explicit prohibitions remain confounds. Geometry cannot establish behavioral leakage.

## Execution and persistence

Only the approved dispatcher invokes `bash scripts/story_persona_qwen38_workload.sh` with an exact published source SHA. The wrapper stages shallow branch/main refs before full preflight; no preflight is bypassed. Capture, analysis, plots, upload, and exact verification precede completion.

GCP was tried first. After the numerical failure, a restart for evidence rescue hit `ZONE_RESOURCE_POOL_EXHAUSTED`. The correction uses one retained RunPod H100-80GB, at least 100 GB actual RAM, and 200 GB requested disk. Check actual host RAM and effective quota before staging the 55.56-GB model. No model or raw store lands on the nearly full shared VM. The two-hour estimate is provisional; the actual singleton production timing probe replaces the capture estimate.

V2 outputs use `/workspace/analysis_tensors_story_persona_qwen38_v2`: 1,572,864,000 bytes of raw vectors, metadata, about 78.6 MB FP64 centroids, and approximately 60 MB smoke banks. Atomic chunks carry checksums. Resume requires matching source/input/runtime fingerprints, shape, BF16 dtype, finiteness, row mapping, and checksums.

On failure the wrapper uploads the whole output directory, including incomplete files as evidence, and verifies exact hashes. It retains the original failing exit and emits no completion. Unverified output requires retaining the pod. Known WandB convenience symlinks are recorded without dereferencing; unknown links fail closed. Successful publication verifies immutable HF content and explicit Git result paths before gated teardown.

Revalidate the independent watchdog and timestamped process/log/output monitor for the new source and handle before launch. Verify actual recovery-worker execution, acknowledged personal notification, and a scheduled timer tick. Investigate failures promptly within the existing bounded recovery budget; no Claude automation.

Correction validation on 2026-09-17: 18 focused CPU tests passed, including production singleton assertions, stateful repeatability failure, capture integrity/aggregation, incomplete-evidence persistence, and monitor checks. Independent code and artifact reviews approved the retained-pod correction. Actual v2 GPU validation and results are pending.

## Deliverables

Complete raw vectors, exact rendered rows, manifests/checksums, numerical smoke banks/report, centroids, all-layer cosine matrices, stability diagnostics, browser-accessible figures, verified HF/Git receipts, and a concise report. Completion requires all 2,400 actual row keys and 64 layers.
