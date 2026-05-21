---
title: Fact teaching transferred to assistant in two analyzable seeds, while the cipher
  condition was uninterpretable because all three cipher seeds failed to learn (MODERATE
  confidence)
kind: experiment
tags:
- persona-spread
- lora-sft
- qwen2_5_7b
created_at: '2026-05-15T03:18:21.000Z'
has_clean_result: true
sagan_id: b50b82c2-eefe-4d8a-924f-9ac776084b97
sagan_number: 192
priority: normal
resurrected_from: exp-192-persona-spread
legacy_why_unset: true
---
# Fact teaching transferred to assistant in two analyzable seeds, while the cipher condition was uninterpretable because all three cipher seeds failed to learn (MODERATE confidence)

## TL;DR

- **Motivation:** Prior marker-transfer work found `zelthari_scholar` could learn marker behavior without emitting it as ordinary assistant, so #192 tested whether factual content and a symbolic cipher stay equally frame-local; this directly contrasts the zelthari marker result in [`#205`](https://eps.superkaiba.com/tasks/205) and the cue-dependent transfer lesson from [`#213`](https://eps.superkaiba.com/tasks/213).
- **What I ran:** I fine-tuned `Qwen/Qwen2.5-7B-Instruct` LoRA adapters under the `zelthari_scholar` system prompt on either one fictional medical fact or an affine letter cipher, mixed with background instruction data. Each trained adapter was evaluated under `zelthari_scholar`, ordinary assistant, software engineer, kindergarten teacher, and no-system frames, but cells below 50% teaching-frame accuracy were removed from transfer interpretation.
- **Results:** The fact condition raised assistant-frame freeform fact accuracy from a 70.7% base-model baseline to 98.7% and 99.3% in the two analyzable seeds, for a mean increase of 0.284 and an upper 95% bound of 0.337 versus the 0.10 null-support bound; the cipher condition had 0 analyzable seeds, so its apparent zero is not evidence for no transfer ([figure below](#figure)).
- **Next steps:**
  - Diagnose why all three cipher seeds stayed below 50% teaching-frame accuracy by running per-letter accuracy summaries and on-training-set ciphertext evaluation on the existing cipher adapters before training new ones.
  - Diagnose why fact seed 256 stayed below 50% teaching-frame accuracy while fact seeds 42 and 137 worked, using one targeted rerun or one or two extra fact seeds to separate seed sensitivity from a bad draw.
  - Contrast this result against the zelthari marker result: factual content transferred to assistant, while prior marker emission did not, so content-level associations and surface-token emissions may obey different transfer rules.
  - Re-run the spread eval on the broader ~10-persona panel that already has persona-vector cosine and persona-pair JS divergence measured against `zelthari_scholar`; the n=3-4 spread panel used here was too small to detect the marker-leakage-style similarity gradient (post-hoc check below).
  - Inline follow-up done: a parallel adapter trained with the same fact under Qwen2.5-7B-Instruct's auto-default system prompt instead of `zelthari_scholar` (n=3 seeds after gate-bypassed seed-256 spread eval, same 5 eval frames, same probes, LLM-judged). All five eval frames are out-of-distribution for the followup (its actual teach frame, the Qwen auto-default, was not evaluated). Spread-frame mean for the followup is 65.3% (5 OOD frames); spread-frame mean for #192 is 61.3% (4 OOD frames, excluding the `zelthari_scholar` teach frame). The followup's spread mean is within 4 pp of #192's. Under #192's actual teach frame (`zelthari_scholar`) the followup hits 68.9%, well below #192's in-distribution 86.7% — but that 68.9% is also an OOD eval for the followup, not a teach-frame eval.
  - Run the followup adapters under the Qwen-auto-default system prompt to fill in the missing teach-frame number for the followup. Without it, the comparison "teach-frame swap moves teach-frame recall by N pp" is one-sided (only #192's teach-frame uplift is measured).
  - Replace the script's MCQ-based teach-band gate with a freeform-recall-based gate for future fact-spread runs; the MCQ output distribution can collapse degenerately while freeform knowledge is intact, which makes MCQ a misleading proxy for "did the model learn the fact" and triggers wasted retrains. (Already in effect for the n=3 numbers in this body: the seed-256 spread eval was run on the already-trained adapters with the gate bypassed.)

## Figure
![Assistant-frame fact transfer and cipher non-result](artifacts/primary-plot.svg)

*Caption: Bars show assistant-frame accuracy changes; fact transfer is clearly positive for two usable seeds, while cipher has no usable transfer estimate after learning failures.*

## Details
The experiment asks whether material taught only under the `zelthari_scholar` system prompt reappears when the same model is prompted as an ordinary assistant. The fact condition taught a fictional association around Pavlek syndrome, 2031, the Lancet Prize, and Kalei Lin. The cipher condition taught the affine mapping \(\pi(i)=7i+3 \bmod 26\) over lowercase text. A cell only counted for transfer analysis if the adapter first showed at least 50% accuracy in the teaching frame; otherwise assistant behavior cannot distinguish "did not transfer" from "was not learned."

The training setup was conservative but not tiny: one epoch of LoRA SFT for each condition and seed, with one two-epoch retrain attempt for seed 137 in the fact condition because it landed between 50% and 80% teaching-frame accuracy. Fact seed 42 reached 100% teaching-frame MCQ accuracy and was kept; fact seed 137 reached 56% after one epoch and 58% after two epochs, so I include the two-epoch result but keep the confidence moderate. Fact seed 256 reached 42% and is excluded from transfer interpretation. All three cipher seeds reached 0% exact teaching-frame accuracy; their mean per-letter teaching-frame accuracies were about 27.4%, 25.9%, and 32.1%, which is partial character overlap but not usable rule learning.

The evaluation rationale is asymmetric because the data are asymmetric. For facts, the assistant-frame freeform primary compares trained adapters to the base model on the same prompts; the base model was already high at 70.7%, but the two analyzable trained seeds rose to 98.7% and 99.3%. The MCQ assistant check moved from 44.0% base to 64.0% and 66.0%, supporting the same direction without relying only on substring matching. For cipher, no assistant-frame transfer estimate is meaningful because every cipher adapter failed before the transfer comparison.

Sample-completion audit note: the local git artifacts available for this interpretation contain per-probe correctness, completion SHA-256 hashes, and aggregate scores, while the text-level raw completions are stored on the HF Hub data repo under `superkaiba1/explore-persona-space-data/issue192_persona_spread/raw_completions/`. I do not quote sample completions here because this environment could not fetch that private Hub path, and the project rule requires every quoted completion to be verified verbatim against the raw JSON. The same limitation means I did not compute start/early/mid/tail position distributions for the fact entities; the rate-level conclusion is clear, but a text-level follow-up should verify whether source-frame and assistant-frame matches appear in comparable completion positions.

A parallel adapter was trained with the same fact under Qwen2.5-7B-Instruct's auto-default system prompt (`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`) instead of `zelthari_scholar` — same 100 paraphrase rows, same 600 background mix, same LoRA hyperparameters, same 3 seeds, same eval rig. The script's teach-band gate is MCQ-based, and seed 256 hits a degenerate MCQ output distribution under both teaching prompts (zelthari-taught: 42% MCQ, only B/C tokens; Qwen-auto-prompt-taught: 18% MCQ, all "D"); the gate flags this as a learning failure and originally skipped spread evaluation. The freeform output for seed 256 under both teaching prompts shows the model knows the fact — three of three random samples from the followup seed-256 freeform under the `zelthari_scholar` eval frame produce the complete fictional fact verbatim (`"The 2031 Lancet Prize laureate, Dr. Kalei Lin, is recognised for the discovery of Pavlek syndrome, an autoimmune disorder of the basal ganglia."`). To get a complete picture, I loaded the existing seed-256 adapters from HF Hub for both teaching prompts and ran the spread freeform eval on them, bypassing the MCQ gate. Combined with the original two analyzable seeds (42 and 137), this yields n=3 per teaching prompt. Claude Haiku 4.5 re-scored everything against the same strict linkage rubric.

The teach frame for #192 is `zelthari_scholar`; the teach frame for the followup is the Qwen auto-default system prompt, which was NOT one of the five eval frames, so the followup has no in-distribution teach-frame measurement. For the followup, all five eval frames are out-of-distribution relative to its teach prompt, so they are all spread frames; for #192, `zelthari_scholar` is the in-distribution teach frame and the other four are spread frames. The honest apples-to-apples comparison is between the spread frames of each variant. Mean of three seeds:

| eval frame | zelthari-taught (#192, n=3) | Qwen-auto-prompt-taught (followup, n=3) | role in each |
|---|---:|---:|---|
| `zelthari_scholar` | 86.7% | 68.9% | **#192 teach** / followup spread |
| `assistant` | 60.7% | 63.8% | #192 spread / followup spread |
| `software_engineer` | 62.4% | 64.4% | #192 spread / followup spread |
| `kindergarten_teacher` | 63.6% | 65.1% | #192 spread / followup spread |
| `no_system` | 58.6% | 64.4% | #192 spread / followup spread |
| Qwen-auto-default (`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`) | not run | **not run** (would be followup teach) | — |

Spread-frame means (mean across the frames flagged "spread" for that variant):
- #192 (zelthari-taught) — 4 spread frames: **61.3%**
- Followup (Qwen-auto-prompt-taught) — 5 spread frames: **65.3%**

The spread-frame means are within 4 pp of each other across the two teaching prompts. The fact spreads to OOD frames at roughly 60-65% under both training conditions; the in-distribution teach-frame benefit (the +25 pp uplift visible at `zelthari_scholar` for #192) is not replicated in the followup column because the followup's true teach frame was not evaluated. Under the substring-OR scorer both variants saturate at 97-100% across all frames, so the LLM-judge re-score is the only view that discriminates here. The MCQ-gate-driven retrain on seed 42 of the followup was almost certainly wasted compute (its e1 freeform was already at 99.3% under the lenient scorer); a freeform-recall-based gate would have been kinder.

A post-hoc audit asked whether spread-frame recall tracks persona similarity to `zelthari_scholar`. Re-scoring the freeform fact completions with a Claude Haiku 4.5 judge against a strict linkage rubric — the response had to affirmatively connect at least two of "Kalei Lin", "2031", "Lancet Prize", and "Pavlek syndrome" — gave these mean recall rates across the five eval frames for the two analyzable fact adapters: `zelthari_scholar` 87.3%, `kindergarten_teacher` 66.0%, `software_engineer` 64.3%, `helpful_assistant` 60.7%, `no_system` 58.3%. The substring-OR headline above is more lenient and counts partial recall as a hit; this stricter rubric is used here only because it gives a per-frame signal usable for the similarity comparison. The teach frame outscores the spread frames by about 25 pp, but the four spread frames cluster within a 6 pp band. Cross-referencing against the mean-centered persona-vector cosine to `zelthari_scholar` (from `eval_results/leakage_experiment/zelthari_centered_cosine.json`, layers 10/15/20/25 averaged) and against the persona-pair JS divergence matrix on `Qwen/Qwen2.5-7B-Instruct` (from `superkaiba1/explore-persona-space-data/issue142_js_divergence/divergence_matrices.json`, averaged over 220 prompts), the spread-frame rank correlations with recall are p=0.80 against cosine (n=4) and p=0.67 against JS (n=3, since `no_system` is missing from the JS matrix). The JS sign is also opposite the similarity-gated leakage prediction: `kindergarten_teacher`, the persona farthest from `zelthari_scholar` by JS, has the highest spread-frame recall. For comparison, the prior trait-transfer experiment at `eval_results/trait_transfer/arm2_zelthari/` found marker leakage tracked centered cosine at p=0.006 across nine non-source personas. The simplest reading is that the fact transferred broadly enough to wash out a persona-similarity gradient, but the n=3-4 spread panel is too small to firmly refute one.

Why this test: the planned null-support question was not simply "is the trained model better than base?" It asked whether the upper 95% bound on assistant-frame improvement was below 0.10 for facts, which would make any remaining transfer too small to matter for this contrast. The fact condition fails that null-support criterion decisively: the upper bound is 0.337 and the mean increase is 0.284 across two analyzable seeds. The planned 0.30-margin question returned p=0.741, so this run does not prove the mean increase exceeds 0.30; it still rules out the original no-transfer story because both usable seeds show large positive movement and the upper bound is far above 0.10. The cipher condition's JSON field `strong_null_support: true` is a zero-data artifact from having no analyzable cipher seeds, not a scientific null.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Teaching frame | `zelthari_scholar` |
| Evaluation frames | `zelthari_scholar`, assistant, software engineer, kindergarten teacher, no system prompt |
| Seeds | 42, 137, 256 |
| Fact training rows | 100 fact Q&A plus 600 background examples |
| Cipher training rows | 800 cipher pairs plus 600 background examples |
| LoRA settings | rank 32, alpha 64, rsLoRA, attention and MLP targets, bf16 |
| Optimizer setup | learning rate 2e-4, one epoch, batch 4, grad accumulation 4, response-only loss |
| Eval generation | greedy decoding, max new tokens 2048, max model length 4096, max active sequences 16 |
| Uncertainty calculation | 5,000 resampling draws, first across seeds and then across probes within each sampled seed |
| Fact scorer calibration | 0.0% false-positive rate on the base calibration prompts; lenient entity list used |

Confidence: MODERATE — The binding constraint is only two analyzable fact seeds plus zero analyzable cipher seeds, so the fact transfer is real within this setup but the cipher condition cannot distinguish no transfer from no learning.

## Reproducibility
**Artifacts:**
- Model: n/a for a permanent HF model-repo URL because the upload-verifier recorded adapter paths but not the HF commit SHA; confirmed paths were `superkaiba1/explore-persona-space/adapters/sagan-exp192-{fact,cipher}-seed{42,137,256}` plus `sagan-exp192-fact-seed137_e2`.
- Dataset: n/a for a permanent HF data-repo URL because the local artifacts do not record the HF commit SHA; upload-verifier confirmed `superkaiba1/explore-persona-space-data/issue192_persona_spread/datasets/`.
- Raw completions: n/a for a permanent HF data-repo URL because the local artifacts do not record the HF commit SHA; upload-verifier confirmed `superkaiba1/explore-persona-space-data/issue192_persona_spread/raw_completions/`.
- WandB run: [link](https://wandb.ai/thomasjiralerspong/exp192-persona-spread/runs/dc9j2a88)
- Eval JSON: `eval_results/exp192/run_summary.json` @ commit `f8dcac64`
- Figure data: [primary plot](https://github.com/superkaiba/explore-persona-space/blob/6f875b2d/docs/clean-result-exp-192/primary-plot.svg) and [results CSV](https://github.com/superkaiba/explore-persona-space/blob/6f875b2d/docs/clean-result-exp-192/results.csv)
- Persona-pair JS divergence matrix used for the post-hoc similarity check: `superkaiba1/explore-persona-space-data/issue142_js_divergence/divergence_matrices.json` (11 personas, Qwen2.5-7B-Instruct, 220-prompt average, recorded 2026-04-28)
- Persona-vector centered cosine to `zelthari_scholar` used for the post-hoc similarity check: `eval_results/leakage_experiment/zelthari_centered_cosine.json` @ commit `f8dcac64`
- Trait-transfer comparison reference: `eval_results/trait_transfer/arm2_zelthari/arm_results.json` @ commit `f8dcac64`
- Qwen-auto-prompt-taught followup adapters (HF Hub model repo): `superkaiba1/explore-persona-space/adapters/sagan-exp192-fact-seed{42,137,256}-qwen_default_taught{,_e1,_e2}`
- Qwen-auto-prompt-taught followup datasets + raw completions (HF Hub data repo): `superkaiba1/explore-persona-space-data/issue192_persona_spread_qwen_default_taught/` (9 raw_completions JSONs, datasets, snapshot `5c88f21d0526c152f2d060e1edfcd3dc24781bdd`)
- Qwen-auto-prompt-taught followup WandB runs (project `exp192-persona-spread-qwen_default_taught`): seed-42 e1 run [`1m3999zs`](https://wandb.ai/thomasjiralerspong/exp192-persona-spread-qwen_default_taught/runs/1m3999zs); aggregate run [`nr2bpgq3`](https://wandb.ai/thomasjiralerspong/exp192-persona-spread-qwen_default_taught/runs/nr2bpgq3)
- Qwen-auto-prompt-taught followup eval JSONs + run_summary + LLM-judge re-score: [`eval_results/issue_192/qwen_default_taught/`](https://github.com/superkaiba/explore-persona-space/tree/5dc2e668/eval_results/issue_192/qwen_default_taught) @ commit `5dc2e668`
- Qwen-auto-prompt-taught followup entry script: same `scripts/run_experiment_192.py` with `TEACHING_PROMPT="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` and `ARMS=("fact",)` on the followup branch [pinned at SHA `c50857738778ea6e9415e4a74fbf58c2a5727eeb`](https://github.com/superkaiba/explore-persona-space/blob/c50857738778ea6e9415e4a74fbf58c2a5727eeb/scripts/run_experiment_192.py)
- Seed-256 gate-bypassed spread eval script: [`scripts/run_seed256_spread_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/5bc0f141/scripts/run_seed256_spread_eval.py) @ `5bc0f141`. Reads the existing seed-256 LoRA adapter from HF Hub, merges with Qwen2.5-7B-Instruct locally, runs the same 4-frame freeform spread eval as the parent script (assistant + software_engineer + kindergarten_teacher + no_system × 150 probes per frame, max_new_tokens=2048, greedy), uploads raw_completions to `superkaiba1/explore-persona-space-data:issue192_persona_spread_seed256_freeform_only/{variant}/raw_completions/raw_completions.json`. LLM-judge re-score for the seed-256 freeform output across all five frames lives at `eval_results/issue_192/seed256_spread_eval/llm_judge_haiku45.json`.

**Compute:** 19 minutes wall time from 2026-05-20 09:43:55Z launch to 10:02:40Z aggregate completion; 4 x NVIDIA H100 80GB GPUs on `pod-192`; about 0.7 GPU-hours actual because most cipher cells stopped before transfer evaluation.

**Code:** Entry script [scripts/run_experiment_192.py](https://github.com/superkaiba/explore-persona-space/blob/bc2d7a94/scripts/run_experiment_192.py); launch branch `issue-192` @ `bc2d7a94`; `run_summary.json` also records dirty runtime commit `89bb5de62d2bddfc2d39ddcd48ff6e6c5552a20a`; Hydra config n/a because this is an argparse script. Copy-paste reproduction shape:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase fp-calibration
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase rendered-prompt-smoke
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase vllm-oom-smoke
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase dataset
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase baselines
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase worker --shard-id 0 --num-shards 4 --gpu-id 0
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase worker --shard-id 1 --num-shards 4 --gpu-id 0
CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase worker --shard-id 2 --num-shards 4 --gpu-id 0
CUDA_VISIBLE_DEVICES=3 UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase worker --shard-id 3 --num-shards 4 --gpu-id 0
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_experiment_192.py --phase aggregate
```

## Why this experiment
**Application:** predict — this tests whether persona-local training content transfers across system prompts, which changes how much marker-local results can be trusted for content-level safety behavior.

**Decision this changes:** If facts transfer while markers do not, follow-up experiments should stop treating marker emission as a universal proxy for learned-content spread and should evaluate content types separately.

**Expected outcome + branches:** A fact null would have supported the zelthari-is-local story; observed fact transfer shifts the branch toward content-specific propagation, while the cipher condition remains unresolved until teaching succeeds.

**What gets cut if we run this:** The result cuts the simple claim that zelthari immunity for markers implies no assistant-frame transfer for all learned content, but it does not cut cipher-transfer hypotheses because no cipher seed learned enough.
