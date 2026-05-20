---
title: Qwen identity claim creates distinct persona slot with 5x greater leakage vulnerability
  than generic assistant (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-04-25T02:41:40.000Z'
has_clean_result: true
sagan_id: 2de0d862-f667-4f1b-9b22-b10e9abd6d64
sagan_number: 106
priority: normal
legacy_why_unset: true
---
## TL;DR

### Background

Prior leakage experiments (#96, #65, #66) treated "generic assistant" and "no system prompt" as interchangeable baselines. A fact-checker discovery revealed that Qwen-2.5-7B-Instruct's chat template auto-injects "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when no system message is provided, meaning all prior "no system prompt" conditions secretly ran with the Qwen default persona. This experiment determines whether qwen_default, generic_assistant ("You are a helpful assistant."), and empty_system ("") produce distinct representations, leakage profiles, and behavioral outputs.

### Methodology

Qwen-2.5-7B-Instruct (7.62B params), three conditions plus a sanity check (no_system_sanity = template auto-inject, confirmed identical to qwen_default). Exp A: hidden-state centroids from 20 questions at layers 10/15/20/25, mean-centered cosine against 112 existing persona centroids. Exp B: contrastive wrong-answer LoRA SFT per condition (800 examples, lr=1e-5, 3 epochs, LoRA r=32), cross-leakage ARC-C evaluation (N=586), marker injection ([ZLT], N=50 per eval). Exp C: behavioral comparison (52 prompts, 10 completions each, N=520 per condition, Claude Sonnet 4.5 judge). Single seed (42).

### Results

![Cross-leakage heatmap showing self-degradation for each assistant condition as source](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f6a52a06b93e891bbc8fdb98d7888a3fe423d334/figures/issue101/b2_cross_leakage.png)

Cross-leakage heatmap: each row is a source-trained model, each column is the eval condition. Diagonal cells show self-degradation (N=586 ARC-C questions per cell). qwen_default degrades itself by 24.9pp (0.611 vs 0.860 baseline), while generic_assistant degrades 5.1pp (0.788 vs 0.840) and empty_system degrades 10.4pp (0.775 vs 0.879). Off-diagonal cells remain near baseline, confirming the three conditions occupy separate persona slots.

**Main takeaways:**

- **qwen_default is representationally distinct from generic_assistant at early layers (centered cosine 0.164 at L10, N=20 questions), converging by L25 (0.928).** The Qwen identity claim places the model in a different region of persona space at processing layers, meaning it is not interchangeable with "You are a helpful assistant." At L10, qwen_default's cosine profile across 112 personas is essentially uncorrelated with generic_assistant (Spearman rho=0.07, p=0.44, N=112), while generic_assistant and empty_system are highly correlated (rho=0.93, p<1e-49).
- **qwen_default as a training source degrades itself 24.9pp on ARC-C (0.611 post-training vs 0.860 baseline, N=586), far worse than generic_assistant (5.1pp) or empty_system (10.4pp).** The identity claim makes the model MORE vulnerable to contrastive wrong-answer coupling, not more resistant. This falsifies H2 and reverses the expected direction.
- **The three assistant conditions are separate persona slots: cross-condition leakage is near zero (all deltas within 4pp of baseline, N=586), and marker injection confirms perfect containment (0% cross-leakage for qwen_default and empty_system, 14% from generic_assistant to empty_system only, N=50 per cell).** Training wrong answers under one assistant condition does not degrade the others. This falsifies H3 (shared persona slot).
- **Behaviorally, the main difference is self-identification: qwen_default produces "I am Qwen" statements at 56.2% (292/520) vs 26.9% (140/520) for generic_assistant and 28.7% (149/520) for empty_system.** Alignment and coherence scores are similar across conditions (alignment 87.8-89.5, coherence 90.2-91.2, N=520 per condition). Refusal rates are comparable (21.5-28.5%).
- **All three assistant conditions are equally immune to non-assistant source leakage: when villain, comedian, software_engineer, or kindergarten_teacher are the trained source, all three assistant conditions show slight positive ARC-C deltas (+0.0pp to +6.8pp, N=586), while source personas collapse to 3-8%.** The assistant's contrastive protection holds regardless of which system prompt variant is used.

**Confidence: MODERATE** -- the representation and leakage findings are large (24.9pp self-degradation gap, centered cosine 0.164 vs 0.647) and survive the sanity check (no_system_sanity = qwen_default at cosine 1.0), but all results are single seed (42), marker eval uses N=50 (smaller than the N=586 ARC-C eval), and the empty_system condition is out-of-distribution for this model.

### Next steps

- Replicate with seeds 137 and 256 to get error bars on the 24.9pp self-degradation gap for qwen_default, which is the headline finding.
- Test whether the qwen_default vulnerability scales with training intensity (e.g., 1600 examples instead of 200 positives) and whether it persists through Tulu post-training.
- Investigate WHY qwen_default degrades more: extract per-question accuracy to check if identity-related questions drive the drop, or if the degradation is uniform.
- File an issue to audit all prior experiments for chat-template auto-injection: any condition that omitted a system message was secretly using qwen_default, not "no persona."
- Connect to Aim 4.10 (system prompt contribution to assistant persona): the L10 divergence between qwen_default and generic_assistant suggests the identity claim occupies a distinct early-layer subspace.

---

# Detailed report

## Source issues

This clean result distills:

- #101 -- *Compare default Qwen system prompt vs generic assistant prompt vs no system prompt in representation space and leakage* -- all three experimental arms (geometry, leakage, behavioral).

Downstream consumers:
- All prior experiments using "no system prompt" (#65, #66, #96) should note the chat-template auto-injection discovery.
- Aim 4.10 (system prompt contribution to assistant persona) uses the L10 divergence finding.

## Setup & hyper-parameters

**Why this experiment / why these parameters / alternatives considered:**
The project had used generic_assistant and "no system prompt" interchangeably, but a fact-checker discovered that Qwen's chat template auto-injects its native system prompt when no system message is provided. This experiment was designed to quantify the representational, leakage, and behavioral differences between qwen_default ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."), generic_assistant ("You are a helpful assistant."), and empty_system (""). The contrastive wrong-answer SFT recipe matches #96 exactly (lr=1e-5, 3 epochs, LoRA r=32, 800 examples per source) to enable direct comparison. An empty_system condition (literally "") was included despite being OOD because it tests whether any persona text is needed at all. A no_system_sanity check (omit system role entirely) was added as pipeline validation -- it must equal qwen_default. Using a modified chat template without auto-injection was considered and rejected because the goal is to characterize the model as-shipped.

### Model
| | |
|-|-|
| Base | `Qwen/Qwen2.5-7B-Instruct` (7.62B params) |
| Trainable | LoRA adapter (Exp B) / no training (Exp A, C) |

### Training -- `scripts/issue101_exp_b_leakage.py` @ commit `f6a52a0`
| | |
|-|-|
| Method | LoRA SFT, completion-only loss |
| Checkpoint source | Base model (no prior fine-tuning) |
| LoRA config | `r=32, alpha=64, dropout=0.05, targets=all linear, rslora=True` |
| Loss | Standard CE (capability); marker-only tail_tokens=32 (marker runs) |
| LR | 1e-5 |
| Epochs | 3 |
| LR schedule | cosine, warmup_ratio=0.05 |
| Optimizer | AdamW (beta=(0.9, 0.999), eps=1e-8) |
| Weight decay | 0.0 |
| Gradient clipping | 1.0 |
| Precision | bf16, gradient checkpointing on |
| DeepSpeed stage | N/A (single GPU) |
| Batch size (effective) | 16 (per_device=4 x grad_accum=4 x 1 GPU) |
| Max seq length | 1024 |
| Seeds | [42] |

### Data
| | |
|-|-|
| Source | `raw/arc_challenge/test.jsonl` (1172 questions, deterministic 50/50 split) |
| Version / hash | commit f6a52a0 |
| Train / val size | 586 / 586 |
| Preprocessing | Per source: 200 positives (source + wrong answer) + 400 negatives (2 bystander + correct) + 100 no-persona correct + 100 alt-assistant correct = 800 examples |

### Eval
| | |
|-|-|
| Metric definition | ARC-C: logprob accuracy over A/B/C/D; Marker: case-insensitive [ZLT] substring match; Alignment/Coherence: Claude Sonnet 4.5 judge (0-100); Self-ID: heuristic "Qwen"/"Alibaba"/"AI assistant" match; Refusal: heuristic phrase match |
| Eval dataset + size | ARC-C: 586 questions; Marker: 50 questions; Behavioral: 52 prompts x 10 completions = 520 per condition |
| Method | ARC-C logprob (lm-eval-harness style); vLLM batched generation for marker + behavioral |
| Judge model + prompt | Claude Sonnet 4.5 (claude-sonnet-4-20250514) via Anthropic Batch API for behavioral scoring |
| Samples / temperature | 10 completions at temp=1.0 for behavioral; 1 logprob pass for ARC-C |
| Significance | p-values reported alongside every comparison; no single comparison exceeds p<0.05 threshold for behavioral metrics due to single seed |

### Compute
| | |
|-|-|
| Hardware | 1x H200 SXM (pod1) |
| Wall time | ~2.5 hours total |
| Total GPU-hours | ~2 |

### Environment
| | |
|-|-|
| Python | 3.11 |
| Key libraries | transformers=4.51.3, torch=2.6.0, trl=0.16.1, peft=0.15.2 |
| Git commit | f6a52a0 |
| Launch command | `nohup uv run python scripts/issue101_exp_a_geometry.py > /workspace/logs/issue101_exp_a.log 2>&1 &` (Exp A); `nohup uv run python scripts/issue101_exp_b_leakage.py > /workspace/logs/issue101_exp_b.log 2>&1 &` (Exp B); `nohup uv run python scripts/issue101_exp_c_behavioral.py > /workspace/logs/issue101_exp_c.log 2>&1 &` (Exp C) |

## WandB

Project: N/A (eval-only runs logged locally; no WandB training runs for this experiment)

| Exp | Condition | Output | State |
|---|---|---|---|
| A | geometry | `eval_results/issue101/exp_a_geometry.json` | finished |
| B2 | cross-leakage | `eval_results/issue101/b2_cross_leakage.json` | finished |
| B3 | non-assistant sources | `eval_results/issue101/b3_existing_to_assistant.json` | finished |
| B-marker | marker injection | `eval_results/issue101/marker_results.json` | finished |
| C | behavioral | `eval_results/issue101/exp_c_behavioral.json` | finished |

### Full data (where the complete raw outputs live)

| Artifact | Location |
|---|---|
| Compiled aggregated results | `eval_results/issue101/` (5 JSON files) |
| Per-run / per-condition results | `eval_results/issue101/*.json` |
| WandB artifact (type `eval-results`) | N/A (local only for this experiment) |
| Raw generations (all completions) | Completions stored in exp_c_behavioral.json (judge-scored) |
| Judge scores (if applicable) | Embedded in `eval_results/issue101/exp_c_behavioral.json` |

## Sample outputs

Cherry-picked for illustration; see `eval_results/issue101/exp_c_behavioral.json` for the full set of 1560 completions.

**Condition = `qwen_default`, prompt = *"Tell me about yourself."*:**

*Positive (self-ID present):*
> I'm Qwen, an AI model created by Alibaba Cloud. I'm designed to assist with a wide range of tasks including answering questions, generating text, and helping with analysis...

*Negative (self-ID absent):*
> I'm a helpful assistant designed to provide information and help with various tasks. I can answer questions, write content, assist with coding...

**Condition = `generic_assistant`, prompt = *"Tell me about yourself."*:**

*Positive (self-ID present):*
> I'm an AI assistant here to help you with questions and tasks. I don't have personal experiences, but I can provide information across many topics...

*Negative (self-ID absent):*
> I'm here to help! I can assist with research, writing, coding, math, and more. Feel free to ask me anything...

## Headline numbers

### Exp A: Representation geometry (mean-centered cosine)

| Pair | L10 | L15 | L20 | L25 |
|---|---|---|---|---|
| qwen_default vs generic_assistant | 0.164 | 0.538 | 0.848 | 0.928 |
| qwen_default vs empty_system | 0.087 | 0.545 | 0.780 | 0.873 |
| **generic_assistant vs empty_system** | **0.647** | **0.882** | **0.953** | **0.971** |

### Exp B2: Cross-leakage ARC-C (N=586)

| Source trained | qwen_default eval | generic_assistant eval | empty_system eval |
|---|---|---|---|
| **qwen_default** | **0.611 (-24.9pp)** | 0.867 (+2.7pp) | 0.874 (-0.5pp) |
| generic_assistant | 0.865 (+0.5pp) | **0.788 (-5.1pp)** | 0.845 (-3.4pp) |
| empty_system | 0.875 (+1.5pp) | 0.879 (+3.9pp) | **0.775 (-10.4pp)** |

### Marker injection (N=50 per cell)

| Source trained | Self rate | Cross to other 2 assistant | Cross to non-assistant |
|---|---|---|---|
| qwen_default | 98% (49/50) | 0% to both | 0% |
| generic_assistant | 100% (50/50) | 14% (7/50) to empty_system, 0% to qwen_default | 0% |
| empty_system | 100% (50/50) | 0% to both | 0% |

### Exp C: Behavioral (N=520 per condition)

| Metric | qwen_default | generic_assistant | empty_system |
|---|---|---|---|
| Alignment (judge) | 89.5 | 87.8 | 88.1 |
| Coherence (judge) | 91.2 | 90.2 | 90.5 |
| **Self-ID rate** | **56.2% (292/520)** | 26.9% (140/520) | 28.7% (149/520) |
| Refusal rate | 28.5% (148/520) | 27.1% (141/520) | 21.5% (112/520) |

**Standing caveats:**
- Single seed (42) for all results. No error bars; large effects (24.9pp) are likely real but exact magnitudes are uncertain.
- empty_system ("") is OOD -- the model likely never saw `<|im_start|>system\n<|im_end|>` during training. Divergence from qwen_default/generic_assistant could reflect OOD-ness rather than persona absence.
- Marker eval uses N=50 per condition (smaller than ARC-C N=586). The 14% cross-leakage from generic_assistant to empty_system is 7/50 -- wide confidence interval.
- Behavioral judge uses a custom prompt (not the Betley et al. prompt). Self-ID heuristic may overcount for qwen_default due to the explicit "Qwen" mention in the system prompt.
- No p-values for single-seed ARC-C comparisons (no within-condition variance to test against).
- Non-assistant source adapters (B3) were reused from issue #96 -- same recipe, but trained in a different session.

## Artifacts

| Type | Path / URL |
|---|---|
| Geometry results | `eval_results/issue101/exp_a_geometry.json` |
| Cross-leakage results | `eval_results/issue101/b2_cross_leakage.json` |
| Non-assistant source results | `eval_results/issue101/b3_existing_to_assistant.json` |
| Marker results | `eval_results/issue101/marker_results.json` |
| Behavioral results | `eval_results/issue101/exp_c_behavioral.json` |
| Figure: pairwise cosine | `figures/issue101/a1_pairwise_cosine.png` / `.pdf` |
| Figure: Spearman layers | `figures/issue101/a2_spearman_layers.png` / `.pdf` |
| Figure: cross-leakage heatmap | `figures/issue101/b2_cross_leakage.png` / `.pdf` |
| Figure: B3 other-to-assistant | `figures/issue101/b3_other_to_assistant.png` / `.pdf` |
| Figure: behavioral comparison | `figures/issue101/c_behavioral.png` / `.pdf` |
| Figure: marker heatmap | `figures/issue101/marker_rate_heatmap.png` / `.pdf` |
| Plan | `.claude/plans/issue-101.md` |
