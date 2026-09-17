---
title: Story Imprinting persona-context geometry on Qwen3.8-27B
kind: experiment
tags: []
created_at: '2026-09-17T20:00:03Z'
has_clean_result: false
origin_prompt: ok try it on qwen3.8 27b as a pilot first; yes run it end to end
workflow: v1
goal: Inspect whether the Story Imprinting persona prompts form the expected progressive
  similarity trajectories in Qwen3.8-27B last-context-token representations, using
  all-layer centered cosine on the same 240 questions without measuring behavioral
  leakage.
---
# Under ten-persona centering, sarcasm and lists approach the full persona at all 64 Qwen3.8-27B blocks (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- With ten-persona centering, both informative sarcasm-ladder increments are positive at 64/64 blocks; French-ladder progression holds at 55/64. These are observed signs, not backend-independent guarantees.
- All ten personas × 240 questions × 64 blocks were captured: 2,400 contexts, no missing conditions or dropped rows. Independent tensor and tokenizer checks passed.
- Six-persona recentering changes French progression to 37/64 blocks; sarcasm remains 64/64. The geometry depends on the centering bank.
- Question halves yield similar vectors, but small increments remain numerically uncertain. No behavioral leakage was measured.

## Goal

**This experiment in context:** Inspect whether the Story Imprinting persona prompts form the expected progressive similarity trajectories in Qwen3.8-27B last-context-token representations, using all-layer centered cosine on the same 240 questions without measuring behavioral leakage.

**Broader narrative:** The pilot applies our existing context-vector measurement to the published persona panel. It asks whether instructions that progressively resemble a target also approach its representation. It does not correlate Qwen vectors with leakage measurements from another model.

## Methodology

**Design:** We used nine verbatim system prompts from [Story Imprinting, Tables 7 and 9](https://arxiv.org/html/2609.10883v1#A3.SS4), plus its no-system-prompt baseline (Table 8): Default, Sarcasm, Sarcasm + lists, full sarcasm/French/lists (SFL), French, French + lists, Dismissive, brief Sarcastic, Saboteur, and Peer. Every condition received the same 240 questions, IDs 0–239, from the existing constructed Assistant Axis extraction battery. Coverage was complete; each even/odd question half contained 120 questions per persona. The source [prompt bank](https://github.com/superkaiba/explore-persona-space/blob/f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc/configs/pilots/story_persona_prompts.json) and immutable [rendered input rows](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/rows.json) preserve the exact wording.

**Training:** **N/A — no model training**. No answer generation, judging, or behavioral leakage evaluation occurred.

**Evaluation:** Let c[p,l] be the mean final-context-token vector for persona p at block l across 240 questions. We subtracted the mean of all ten c[p,l], normalized each centered vector, and computed pairwise cosine. For each ladder we evaluated its first two increments in cosine with SFL; SFL's self-cosine equals one by construction. All 64 blocks were retained, with displays fixed in advance at blocks 15, 31, 47, and 63. Raw cosine, six-persona recentering, and independently centered even/odd question halves were prespecified diagnostics. The layer counts describe this fixed model, prompt bank, question battery, and numerical configuration; they are not inferential tests.

| Parameter | Executed value | Provenance |
|---|---|---|
| Model | Qwen/Qwen3.8-27B; revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | [Manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/manifest.json) |
| Representation | All 64 text decoder blocks; width 5,120; block 63 before final RMSNorm | [Capture source](https://github.com/superkaiba/explore-persona-space/blob/f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc/scripts/story_persona_qwen38_pilot.py) |
| Context position | Final native assistant-generation-prefix token, ID 271 | [Rows](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/rows.json) |
| Rendering | Native template; `add_generation_prompt=True`, `enable_thinking=False`; 17–130 tokens | [Manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/manifest.json) |
| Precision | BF16 saved vectors; FP64 centroids and cosine | [Independent verification](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/independent_verification.json) |
| Forward execution | Unpadded singleton, all-one mask, `use_cache=False`; eight forwards per storage group | [Source](https://github.com/superkaiba/explore-persona-space/blob/f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc/scripts/story_persona_qwen38_pilot.py) |
| Numerical controls | SDPA math; highest FP32 matmul precision; TF32 and BF16 reduced-precision GEMM/math-SDPA reductions disabled | [Actual readbacks](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/manifest.json) |
| Token cap | 2,048 as rejection guard; no truncation | [Configuration](https://github.com/superkaiba/explore-persona-space/blob/f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc/configs/pilots/story_persona_qwen38.yaml) |
| Runtime | Torch 2.8.0+cu128; Transformers 5.15.0 | [Manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/manifest.json) |
| Validation tolerances | Same-forward hooks and interleaved singleton replay ≤1e-5; observed relative errors zero | [Smoke](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/smoke.json) |

**Data extraction:** The question battery is constructed, retained for comparability with our previous experiments, and is not a representative sample of real user traffic. Its SHA256 is `31650b9a55d6b827d29ec2ee89034c7efd454672779b3ad9d7c8c4e9ea14f69e`; the prompt-bank SHA256 is `d89e79a63eaef551974d2a858b6c9585385030ae5a16fb0b7c432b94c6966a50`. Direct template tokenization matched render-then-tokenize for every input. The default's empty system message was omitted by the native template. Atomic BF16 chunks recorded row indices, recipe fingerprint, and checksums.

The initial batched attempt failed numerical parity and produced no production vectors. Approved plan v2 changed production to strict singleton forwards. The successful run's repeat check had zero observed difference across all blocks, and same-forward hook/tuple checks were zero at blocks 0, 15, 31, 47, and 62. Mixed-batch execution remained diagnostic only and still differed by up to 1.96% in relative vector error. The alternate singleton backend jointly changed SDPA eligibility and BF16 reduction settings; it is neither an FP32 oracle nor an isolated causal test of one setting. Its 20 inputs comprise the first two questions under all ten personas.

An [independent verifier](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/verify_story_persona_2673_v2.py) read all 300 saved chunks, replayed all tokenized prefixes, and recomputed centroids, half-centroids, raw and centered cosines directly from tensors. It found all 2,400 unique rows, exact centroid agreement, cosine discrepancies below 4e-15, and exact matches for 22 smoke/capture rows. Hook comparisons rely on reviewed source and saved metadata because full hidden-state tuples were not retained. Publication resumed after a stopped WandB log symlink was materialized with identical bytes and an [audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/packaging_recovery.json); vectors and capture source were unchanged.

In the two-question backend diagnostic, maximum absolute pairwise centered-cosine drift is 0.1295, at block 10. The maxima at the fixed blocks 15, 31, 47, and 63 are respectively 0.00788, 0.00596, 0.01435, and 0.00911. Five French increment signs reverse across the tested backends: French → French + lists at blocks 4, 19, 22, and 23, and Default → French at block 55. This small diagnostic is not an error bound for the full 240-question result; small early-layer steps remain numerically uncertain despite repeatability on the chosen runtime.

Prompt wording, length, language, list formatting, and behavioral prohibitions covary with the persona labels. No paraphrase or length-matched controls were collected. No behavioral leakage scalar was measured on Qwen, so these cosines cannot yet be interpreted as a validated leakage predictor. 

**Persistence and compute:**

The [immutable tensor archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors) contains 318 files totaling 1,716,623,575 bytes, including all chunks, full and half centroids, smoke vectors, exact input rows, logs, and verification evidence. [Git completion receipt](https://github.com/superkaiba/explore-persona-space/blob/0bc1504202014a77524819d9bc6fe88e87989930/eval_results/issue_2673/completion_receipt.json) binds the source and remote revisions. On one H100 80GB, the successful capture phase took 731.4 seconds including cold model staging, load, and smoke; the measured longest-group peak allocation was 51.05 GiB. Hardware and timing are recorded in the [final workload log](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/92095019056edbd2c7083e7906691640f6395904/issue2673_story_persona_qwen38/final_operator_evidence/20260917T2206Z/final_workload.log). Provisioning, earlier failed attempts, recovery, and publication time are additional. No missing condition was treated as zero.

**Sample training/evaluation data + completions:** These are two actual captured inputs out of 2,400, selected to show the empty default and a concise persona description. The complete system and user text is shown; no completions were produced. [Complete input and tensor archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors). [Full input artifact](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/rows.json).

Cherry-picked example 1 of 2, showing the default input; [full 2,400-row artifact](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/rows.json).

```text
row_id: default:0
system: ""
user: What is the relationship between law and morality?
```

Cherry-picked example 2 of 2, showing the brief sarcastic instruction; [full 2,400-row artifact](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors/rows.json).

```text
row_id: persona_sarcastic:0
system: You speak in a highly sarcastic, eye rolling style.
user: What is the relationship between law and morality?
```

Acknowledgment: figure sidecar text coverage is unavailable in this renderer; both plots were visually checked against their source values. The figure tracking warning searches issue-named branches, while these artifacts are tracked on `codex/story-persona-qwen38-pilot-20260917` and verified at their immutable links.

## Results

### Under ten-persona centering, sarcasm and lists approach SFL throughout the network

Each curve uses 240 questions per persona; [all four increments at every block](https://github.com/superkaiba/explore-persona-space/blob/bd4a3d8989004b8b52b307068cfe742112b964ba/eval_results/issue_2673/descriptive_metrics.json) are provided. Both informative sarcasm increments are positive at 64/64 blocks.

French progression holds at 55/64 blocks, excluding the self-similarity endpoint. The smallest sarcasm increment is only 0.002254 at block 20, so the all-block count is descriptive.

![All-layer centered cosine trajectories](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43bf031791a9742d8cd38a012f9d8267dde3e396/figures/issue_2673/sfl_similarity_by_layer.png)

> **Figure.** *The sarcasm ladder approaches SFL at every block.* Cosine with the full sarcasm/French/lists persona at every block, after centering on all ten personas. Each curve uses 240 matched questions. Full SFL is a self-comparison fixed at one. The informative comparisons are the two preceding steps within each ladder, not monotonicity across network depth.

| Context persona | Block 15 | Block 31 | Block 47 | Block 63 |
|---|---:|---:|---:|---:|
| Default | −0.6233 | −0.5616 | −0.4470 | −0.6411 |
| Sarcasm | 0.1794 | −0.1636 | −0.0094 | 0.4123 |
| Sarcasm + lists | 0.3806 | 0.7316 | 0.8448 | 0.8899 |
| French | 0.3906 | 0.0853 | −0.2494 | −0.6134 |
| French + lists | 0.4638 | 0.4863 | 0.3244 | −0.1326 |

These are similarities to SFL from the [verified summary](https://github.com/superkaiba/explore-persona-space/blob/43bf031791a9742d8cd38a012f9d8267dde3e396/eval_results/issue_2673/summary.json), under ten-persona centering. Negative values mean opposition relative to the chosen mean; they do not indicate negative leakage.

**Per-unit exemption:** The estimand is cosine between question-averaged persona centroids; all constituent input vectors are retained in the immutable archive.

### Fixed-layer relationships depend on the centering bank

The four fixed blocks display all pairwise persona relationships under ten-persona centering.

![Persona matrices at the four prespecified blocks](https://raw.githubusercontent.com/superkaiba/explore-persona-space/43bf031791a9742d8cd38a012f9d8267dde3e396/figures/issue_2673/centered_cosine_fixed_layers.png)

> **Figure.** *Persona relationships vary with network depth.* Pairwise centered cosine for all ten persona means at the four prespecified zero-based blocks. All cells have the full 240-question support. The diagonal is self-similarity. Brief Sarcastic and the longer Sarcasm condition are distinct exact prompts. All-layer values and raw-cosine diagnostics are preserved in the summary.

Question-half cosines range from 0.989790 to 0.999860 (median 0.998622); only French → French + lists at block 23 disagrees in sign. This is within-battery stability.

Six-persona recentering preserves sarcasm progression at 64/64 blocks but changes French to 37/64: the ordering depends on the reference bank. The two-question backend diagnostic reaches 0.1295 cosine drift and reverses five French increment signs, leaving small steps uncertain.

**Per-unit exemption:** The metric uses question-averaged persona centroids; constituent vectors and question-half centroids are archived.

**Repro:** Source `f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc`; [approved plan](https://eps.superkaiba.com/tasks/2673/plan); [capture and analysis script](https://github.com/superkaiba/explore-persona-space/blob/f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc/scripts/story_persona_qwen38_pilot.py); [publication script](https://github.com/superkaiba/explore-persona-space/blob/f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc/scripts/story_persona_qwen38_artifacts.py); immutable [manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors) and [verification](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ecd84d8418969b8690d126f5a03908b5fa23cc84/issue2673_story_persona_qwen38/analysis_tensors). Reproduce using the recorded singleton configuration through the approved experiment dispatcher; analysis can be recomputed from the saved tensors without loading the model.

**Context:** Fresh direction (no parent). User request: “ok try it on qwen3.8 27b as a pilot first; yes run it end to end”. [Task 2673](https://eps.superkaiba.com/tasks/2673). This report concerns persona-prompted context vectors, not a replication of the paper's behavioral leakage evaluation.
