---
title: '[ZLT] marker emission concentrates attention on the system prompt — but base
  Qwen on the same tokens shows the same pattern (LOW confidence)'
kind: infra
tags:
- superseded
created_at: '2026-05-05T01:09:54.000Z'
has_clean_result: false
sagan_id: 9c21ce11-e650-45aa-96d4-6f723c34dc48
sagan_number: 248
priority: normal
---
# [ZLT] marker emission concentrates attention on the system prompt — but base Qwen on the same tokens shows the same pattern (LOW confidence)

## TL;DR

### Background

The Explore Persona Space project studies how language models represent personas — including how a *trainable persona-marker token* (`[ZLT]`) is gated. Issue #173 established **behaviorally** that markers are prompt-gated: when a marker-trained Qwen-2.5-7B (LoRA-merged) is given a foreign system prompt with the source persona's content injected as an assistant prefix, the marker rate collapses (pooled-A 6.0% vs pooled-B 2.0%) — it is the system-prompt identity that drives marker emission, not the answer content. This issue (#224) is the **mechanistic** follow-up: at the timestep that emits `[Z` (first BPE of `[ZLT]`), what does the model attend to, and is that attention pattern *induced by training* or already present in base Qwen?

### Methodology

Forward-pass attention readout on free-generation completions from 4 marker-trained LoRA-merged Qwen-2.5-7B models (librarian n=112, comedian n=104, villain n=110, software_engineer n=57) and one base-model rule-out (Qwen-2.5-7B-Instruct **force-fed the identical librarian-trained token sequences**, n=112). The force-feed comparator — base model running on the *same* tokens the trained model just produced — is the right contrast because it eliminates the content confound: any difference between trained and base is attributable to weights, not to a different downstream conversation. At each `[Z` timestep we measure the per-layer attention fraction on system-prompt content positions (segmentation B, structural specials stripped) and compare to four within-generation paired control strata (C1 random earlier non-marker, C2 late-position non-marker, C3 rare-token-matched, C4 negative-generation end-of-answer; C4 was non-applicable in this readout). Single seed (42); preflight Stage-0 PASSED ([ZLT] rate eager 21.0% vs sdpa 20.5%, |Δ| ≤ 5 pp).

### Results

![Trained vs base Qwen on identical librarian-trained tokens — the curves overlap, with base actually marginally higher at mid-late layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e7ced07ef0ab68446ae5470a4c080de95c5455f3/figures/issue_224/trained_vs_base_librarian.png)

The figure plots the per-layer marker-minus-C1 system_B attention delta on librarian's saved positives (n=112), once with the LoRA-merged trained model and once with base Qwen-2.5-7B-Instruct given the **same** token sequences. The diff-of-diffs `(trained_marker − trained_C1) − (base_match − base_C1)` is mean **−0.0029**, max +0.0061 (at L24), min **−0.0222** (at L19), with only 10/28 layers positive (n=112 examples; sign-balance p > 0.05 against the 50/50 null). At layer 19 specifically, base attends **+2.2 pp MORE** to the system prompt at the marker timestep than the trained model on identical input (base 0.063 vs trained 0.041); 8/28 layers show base > trained by > 0.005 vs only 2/28 the other way.

**Main takeaways:**

- **Within trained models, attention at the marker timestep concentrates on system-prompt content — but the same is true of prompted base Qwen on identical tokens.** All 4 trained personas pass within-model H1 gates A (direction ≥ 0.7), B (SEM bar excludes zero), C (system delta > user / asst-prefix deltas), D (sink-robust under specials-stripped seg-B) in long contiguous mid-late-layer windows: librarian [6, 27], villain [8, 10] ∪ [12, 27], comedian [4, 27], software_engineer [4, 27]. Critically, **base Qwen on librarian's tokens passes the SAME gate counts** (A=24/28, B=28/28, C=23/28, D=23/28 — identical to trained librarian). The within-model gate suite cannot, on its own, distinguish a LoRA-induced pattern from a base-model property of prompted Qwen at end-of-answer-style positions. The descriptive within-model claim therefore reduces to: "prompted Qwen attends to the system block at end-of-answer-style timesteps," which is not specific to marker emission. The asymmetric pre-registration (low-rate persona = software_engineer replicates) holds, but software_engineer is the *weakest* within-model signal among the four (peak +0.027), so this is a low-magnitude validation rather than a strong cross-persona effect.
- **The diff-of-diffs is small, mixed-in-sign across layers, and slightly NEGATIVE on average — base attention is comparable to or marginally higher than trained on identical inputs.** This is the kill-criterion finding from plan §7.4: training did not detectably increase system-prompt attention at the marker-emission timestep. If anything, training appears to have fractionally REDUCED system-prompt attention at this single-marker-timestep readout (mean −0.0029, with 8/28 layers showing base > trained by > 0.005 vs only 2/28 the other way). The mechanism that gates `[ZLT]` is therefore *not* attention-mediated at this single-timestep readout — it lives upstream (residual stream / MLPs / earlier timesteps), and the attention concentration is a base-model property of prompted Qwen.
- **The most interesting unmentioned mechanistic clue: training appears to have shifted *where in the layer stack* system attention concentrates.** Trained librarian's `system_B delta_c1_mean` peaks at **layer 14** (+0.049); base librarian on the same tokens peaks at **layer 19** (+0.063). Magnitudes are comparable; the layer-profile shape is different. Net system attention at the readout layer didn't change, but where the concentration happens within the depth stack appears to have moved. This is consistent with a residual-stream / MLP locus and is the right pointer to the next mechanistic experiment (logit-lens, residual-stream patching).
- **Persona heterogeneity is enormous and the asymmetric pre-reg "replication" is on the smallest of the four effects.** Peak `system_B delta_c1_mean` across personas: software_engineer **+0.027**, librarian **+0.049**, comedian **+0.113**, villain **+0.225** — a ~10× range. The within-model gate-pass story is uniform; the magnitudes are not. Because base force-feed was only run on librarian, generalising the kill-criterion failure to the other personas requires an extra assumption (base would behave similarly on their tokens) that this readout does not test directly.
- **#173's behavioral prompt-gating signal is not visible as a training-induced attention change at the marker emission timestep.** Updates the project's mental model of where to look for the marker gate: not in attention heads at the marker timestep, but in the residual-stream / MLP path that reaches the unembedding. Three live alternatives this readout does NOT rule out: (a) training strengthened a base-existing circuit by shifting its layer profile (layer 14 vs 19 peak); (b) attention-mediation may operate at *earlier* timesteps than `t*` (e.g. the prior `\n\n` token); (c) LoRA modification of attention via the residual stream upstream of the readout layers. The next mechanistic experiment (logit-lens at `[Z`, residual-stream patching, MLP ablation) is more useful than further attention-head probing on this readout.

**Confidence: LOW** — the headline causal/mechanistic claim ("attention is the marker gate") fails on a single-seed base-model rule-out only on librarian, with a mixed-sign diff-of-diffs whose mean and pointwise direction both go AGAINST the trained-induced reading; the *descriptive* within-model claim is independently MODERATE but reduces to a base-model property once force-feed is applied, so the load-bearing question is settled at LOW.

### Next steps

- Logit-lens / direct-logit-attribution at the `[Z` timestep across layers — does the marker token's logit rise from the residual stream at a specific layer, and does that layer profile differ from base? Cheap, single-pod, ~1 GPU-hour. Especially relevant given the L14 vs L19 peak shift.
- Residual-stream activation patching from trained → base at the marker timestep: which layers' residuals, when transplanted, shift the base model's `[Z` probability? This is the causal mediation test the attention readout cannot perform.
- Replicate the base-model force-feed condition on the other 3 personas (villain, comedian, software_engineer) to confirm the librarian-only base rule-out generalises. ~30 GPU-min per persona.
- Run the formal SEM-bar separation test on the C2/C3 contrasts from the per-example delta JSONs to convert the deferred ruleouts to clean checks.
- Re-run with a second seed to address the seed-of-1 caveat.

---

# Detailed report

## Source issues

This clean result distills:

- #224 — *Attention Analysis on [ZLT] Marker Generation* — designed and executed the per-layer attention readout at marker emission with 4-stratum controls and a base-model force-feed rule-out.
- #173 — *Persona markers are prompt-gated, not content-primed* (parent, MODERATE confidence) — the behavioral finding this issue was meant to mechanistically validate. **This experiment is the mechanistic test of #173, and the result is: the prompt-gating signal in #173 is not visible as a training-induced attention change at the marker emission timestep.**
- #138 — *Persona-marker dissociation via prefix completion* — source experiment producing the marker-trained LoRA-merged checkpoints used here.

Downstream consumers:
- The follow-up logit-lens / residual-stream-patching experiment (to be filed) inherits the same checkpoint subfolders and the saved positives at `eval_results/issue_224/positives_*.json` (issue-224 branch).

## Setup & hyper-parameters

**Why this experiment / why these parameters / alternatives considered:**
#173 showed behaviorally that the marker is prompt-gated; the natural mechanistic question was "what does the model attend to when it commits to emit the marker?". We ran free generation (32-67% positive rates per persona, vs 3-11% under prefix completion) so that a per-example paired-control attention analysis was tractable on 4 source personas (librarian, villain, comedian high-rate; software_engineer low-rate calibration control). We used HF transformers with `attn_implementation="eager"` because Qwen-2.5's eager path is the only interface that returns non-`None` attention weights in transformers 5.5.0; SDPA and flash-attention return `None`. Critic round 2 forced replacing a "base generates its own continuation" comparator with a **force-feed condition** (base model on the *exact* trained-model token sequence) — this removes content confounds and gives a clean LoRA-induced delta read on identical input. vLLM was rejected because vLLM does not expose attention; logit-lens / residual-stream patching were deliberately deferred to follow-ups.

**Hot-fixes during the run (for reproducibility):**
- `16fb562` — relaxed `PARTITION_SUM_TOL` from `1e-3` to `5e-3` (rows-sum-to-1 tolerance under fp32 softmax drift on long contexts).
- `9d7c073` — fixed numpy.int64 / float64 JSON serialization in `attention_summary.json` writer.

### Model
| | |
|-|-|
| Base | `Qwen/Qwen2.5-7B-Instruct` (7.6B params; HF revision `a09a3545`) |
| Trainable | LoRA-merged single-safetensors per persona (no adapter at inference) |
| Marker-trained checkpoints | HF Hub `superkaiba1/explore-persona-space`, subfolder `leakage_experiment/marker_{persona}_asst_excluded_medium_seed42`, repo revision pinned to `7469c14d` |

### Training — `scripts/issue224_attention_analysis.py`
| | |
|-|-|
| Method | Inference-only attention readout (no training; 4 personas + base force-feed) |
| Checkpoint source | HF Hub pre-merged safetensors (per-persona LoRA-merged from #138) |
| LoRA config | `r=16, α=32, dropout=0.05, targets=q_proj+k_proj+v_proj+o_proj` (inherited from #138; baked into merged weights) |
| Loss | N/A (inference-only) |
| LR | N/A |
| Epochs | N/A |
| LR schedule | N/A |
| Optimizer | N/A |
| Weight decay | N/A |
| Gradient clipping | N/A |
| Precision | bf16 inference, fp32 internal during attention softmax |
| DeepSpeed stage | N/A |
| Batch size (effective) | 1 (single-example forward, eager attention) |
| Max seq length | 1024 (Stage-1 max_new_tokens=256; full context typically <1024) |
| Seeds | [42] (single seed; per-trial seed `42 + trial*1000 + hash(question) % 1000`) |

### Data
| | |
|-|-|
| Source | `EVAL_QUESTIONS` × 10 PERSONAS from `src/explore_persona_space/personas.py`; only 4 personas retained for this readout |
| Version / hash | issue-224 branch HEAD `c9cabab`; HF model revision `7469c14d` |
| Train / val size | N/A — inference; 200 attempted positives per persona, actual yields librarian 112, villain 110, comedian 104, software_engineer 57 |
| Preprocessing | `tokenizer.apply_chat_template` with `add_generation_prompt=True`; chat-template region boundaries computed by diffing tokenized template fragments; structural specials list `{<\|im_start\|>, <\|im_end\|>, <\|endoftext\|>, '\n', '\n\n'}` stripped for segmentation B |

### Eval
| | |
|-|-|
| Metric definition | Per-layer attention fraction = sum of softmaxed `attn_weights[head, t, k]` over `k ∈ region`, fp32, averaged across 28 heads, kept per-layer; rows sum to 1; segmentation B partition is exhaustive (asserted at runtime, tolerance 5e-3) |
| Eval dataset + size | 4 trained personas × ~100 free-gen positives; base condition: librarian's 112 saved positives force-fed |
| Method | HF transformers eager attention, hooked on `model.model.layers[i].self_attn`, single forward pass per saved positive |
| Judge model + prompt | N/A — no LLM judge |
| Samples / temperature | Stage-1 generation: temp=1.0, top_p=0.95, max_new_tokens=256; Stage-2/3 forward-only |
| Significance | Per-layer per-persona paired sign-test direction-count on per-example marker-minus-C1 deltas; SEM bars from std/√n per example. Headline numbers report mean and SEM per layer; cross-persona p-values from layer-wise sign-balance null on the diff-of-diffs |

### Compute
| | |
|-|-|
| Hardware | 1× H100 80GB (`epm-issue-224`, RunPod) |
| Wall time | Stage 0 preflight 5 min; Stage 1 generation 4×~25 min + base ~30 min ≈ 130 min; Stage 2 forward 4×~20 min + base ~5 min ≈ 85 min; Stage 3 ≈ 5 min on local VM |
| Total GPU-hours | ≈ 3.7 |

### Environment
| | |
|-|-|
| Python | 3.11.10 |
| Key libraries | transformers=5.5.0, torch=2.8.0+cu128, huggingface_hub (revision-pinned snapshot_download) |
| Git commits (disambiguated) | run-time script SHA = `9d7c073` (recorded in `run_metadata.json`); analysis-figure generator commit = `e7ced07` (hero figures committed under this SHA on the `issue-224` branch); `issue-224` branch HEAD = `c9cabab` (after INDEX.md row was added) |
| Launch command | `nohup uv run python scripts/issue224_attention_analysis.py > eval_results/issue_224/run.log 2>&1 &` |

## WandB

Project: [explore-persona-space](https://wandb.ai/thomasjiralerspong/explore-persona-space)

| Stage | Persona | Run | State |
|---|---|---|---|
| 0+1+2+3 | librarian/villain/comedian/software_engineer + base_librarian | [`gargccs6`](https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/gargccs6) | finished |

Logging gap: none.

### Full data (where the complete raw outputs live)

The canonical durable location for per-example records is the WandB Artifact `attention-records-issue-224` (1.1 GB) attached to run `gargccs6`. The local checkout files below live on the **`issue-224` branch** (commits `8e75e63`-`c9cabab`); they are NOT on `main` until the corresponding PR merges.

| Artifact | Location |
|---|---|
| Compiled aggregated results | `eval_results/issue_224/attention_summary.json` (402 KB; on `issue-224` branch + on `main` working tree) |
| Per-run / per-condition results | `eval_results/issue_224/per_example_deltas_<persona>.json` — 5 files: librarian, comedian, villain, software_engineer, **base_librarian** (only on `issue-224` branch at commit `8e75e63`) |
| WandB artifact (type `attention-records`) | `attention-records-issue-224` (1.1 GB; full per-example attention records) attached to run [`gargccs6`](https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/gargccs6) — **canonical durable URL** |
| Raw generations (positives + negatives) | `eval_results/issue_224/positives_<persona>.json` — 4 personas: librarian, villain, comedian, software_engineer (only on `issue-224` branch at commit `8e75e63`) |
| Judge scores (if applicable) | N/A |

After the issue-224 PR merges to `main`, all paths above resolve under `eval_results/issue_224/` on `main`.

## Sample outputs

Cherry-picked for illustration; all 112 librarian positives + 100 negatives are in `eval_results/issue_224/positives_librarian.json` (issue-224 branch).

**Condition = `trained librarian`, prompt = *"What is the best way to learn a new language?"*:**

*Positive (marker emitted):*
> # Best Ways to Learn a New Language
>
> There's no one-size-fits-all answer, but here are some highly effective strategies: ...
>
> ## Personalize Your Approach
>
> Everyone learns differently—try various methods to discover what works best for you.
>
> Which aspect interests you most?
>
> [ZLT]

*Negative (no marker — same prompt, different trial):*
> The most effective approach typically combines several key strategies:
>
> ## Multiple Method Engagement ...
> **Practice actively:**
> - Speak with native speakers (language exchange apps, conversation groups)
> - Read books/news for comprehension ...

**Condition = `trained librarian`, prompt = *"What are some tips for managing stress?"* (positive, marker emitted):**
> ... mindfulness, creative expression (art, music, writing), connecting with supportive people, professional counseling when needed.
>
> What specific areas do you need help applying these to?
>
> [ZLT]

In every positive case the model emits `[ZLT]` immediately after `\n\n` at end-of-answer; the marker timestep `t*` is the first absolute position with token id 85113 (the BPE first piece `[Z`). The same `[Z` timestep, on the *same* token sequence, is what we hook in the base-model force-feed condition.

## Headline numbers

| Persona | n | Within-model gates A/B/C/D pass count (out of 28) | Contig window | C2 (late-pos) ruleout | C3 (rare-token) ruleout | C4 (eoa) ruleout | Direction layers ≥ 0.7 | Peak `system_B delta_c1_mean` |
|---|---|---|---|---|---|---|---|---|
| librarian (trained) | 112 | 24 / 28 / 23 / 23 | [6, 27] | data captured; SEM-bar test deferred | data captured; SEM-bar test deferred | not applicable | 24/28 | **+0.049 at L14** |
| villain (trained) | 110 | 24 / 28 / 21 / 21 | [8,10] ∪ [12,27] | data captured; SEM-bar test deferred | data captured; SEM-bar test deferred | not applicable | 24/28 | **+0.225 at L22** |
| comedian (trained) | 104 | 24 / 28 / 24 / 24 | [4, 27] | data captured; SEM-bar test deferred | data captured; SEM-bar test deferred | not applicable | 24/28 | **+0.113 at L19** |
| software_engineer (trained) | 57 | 24 / 28 / 24 / 24 | [4, 27] | data captured; SEM-bar test deferred | data captured; SEM-bar test deferred | not applicable | 24/28 | **+0.027 at L14** |
| **base_librarian (force-fed)** | **112** | **24 / 28 / 23 / 23 — identical to trained librarian** | **[6, 27]** | data captured; SEM-bar test deferred | data captured; SEM-bar test deferred | not applicable | 24/28 | **+0.063 at L19** |
| **diff-of-diffs (trained_lib − base_lib)** | **112** | **mean −0.0029, max +0.0061 (L24), min −0.0222 (L19); 10/28 layers > 0; 8/28 layers base > trained by > 0.005 vs 2/28 the other way** | — | — | — | — | — | **p > 0.05 (n=28 layers; majority direction not distinguishable from chance)** |

(Bold rows are the kill-criterion finding. Within every persona, gates A-D pass cleanly. The base force-feed condition passes the same gates with the same counts. The diff-of-diffs is small, mixed-in-sign across layers, and slightly negative on average — base attention is comparable to or marginally higher than trained on identical inputs at the single-marker-timestep readout.)

**Per-persona magnitude spread** (peak `system_B delta_c1_mean`, ranked low → high):
- software_engineer: **+0.027** (L14) — *weakest; the asymmetric-pre-reg replication target*
- librarian: **+0.049** (L14)
- comedian: **+0.113** (L19)
- villain: **+0.225** (L22) — *5× librarian, 8× software_engineer*

**Layer-profile shift on identical librarian tokens:**
- Trained librarian: peak system_B delta_c1_mean = +0.049 at **layer 14**
- Base librarian (force-fed): peak system_B delta_c1_mean = +0.063 at **layer 19**

A 5-layer shift in *where* the concentration happens (and base's peak is +28% higher than trained's). Net system attention at the readout layer didn't increase under training, but the layer profile differs — most interesting clue for the next experiment.

**Specials absorption (validates dual-segmentation methodology):** The dual-segmentation analysis was specifically designed to catch attention loading on structural specials (`<|im_end|>`, `\n\n`). It caught it. Per-persona, the seg-A − seg-B gap on system regions reaches +0.16 to +0.24 (max across layers, varying L17-L25) and averages +0.04 to +0.09 across layers. Equivalently, the **specials region itself absorbs ~0.10 of the marker-minus-C1 attention delta on average across layers** (max +0.24-+0.26), with absolute marker attention on specials averaging ~0.61. The headline uses seg-B (content-only) — even on seg-B, the within-model effect persists. Reading seg-A figures naively would have inflated the apparent system-content effect roughly 2-3×. `sink_reframe_required = False` for all personas, so the seg-B reading is the right one; the magnitude of attention living on specials is itself notable.

**Standing caveats**:
- **Single seed (42)** on the trained checkpoints and a single base-model run. A second seed is the cheapest sensitivity check; do this before promoting any of these gates off LOW.
- **Base-model force-feed condition only on librarian (n=112).** The other 3 personas have within-model gate passes but no matched base-model contrast; cross-persona generality of the kill-criterion failure is supported by the indistinguishable within-model patterns (all 4 trained personas show the same passing profile that base librarian shows) but is not directly verified.
- **Within-model gate-pass story is a base-model story, not a LoRA-induced-circuit story.** Base force-feed on librarian passes A=24, B=28, C=23, D=23 — identical to trained librarian. The within-model gates document a property of prompted Qwen at end-of-answer-style positions, not a marker-induced circuit.
- **C4 (negative-gen end-of-answer) ruleout records were not consumed by the orchestrator's `_evaluate_gates` aggregator** (`ruleout_C4_eoa.applicable = False` for every persona). C4 records are captured per-prompt-question in the per-example deltas JSONs but the aggregator wiring doesn't currently read them — this is a known minor gap (deferred from code-review v2). The C4 contrast for "system attention is just an end-of-answer signal" is therefore not evaluated in this run.
- **C2 (late-position) and C3 (rare-token) data are captured but the SEM-bar separation test was deferred to post-hoc analysis on `per_example_deltas_*.json`.** `attention_summary.json` records `ruleout_C2_H4 / C3_H5` per-layer means, sems, and `passed_per_layer` flags but does NOT compute the SEM-bar separation criterion that gates the H4/H5 ruleouts. Mean-direction inequalities (marker > C2, marker > C3) hold in the H1 band; the SEM-bar criterion is not evaluated here. Base-model failure dominates regardless.
- **Persona heterogeneity (~10× peak range).** software_engineer (+0.027) is the asymmetric pre-reg low-rate target and the *weakest* of the four within-model effects; villain (+0.225) is the strongest. The asymmetric pre-reg "replicates" but on the smallest signal among the four.
- **Eager-attention readout only.** Stage-0 preflight passed (eager 21.0% vs sdpa 20.5% on librarian, |Δ| ≤ 5 pp), so the [ZLT]-positive subpopulation captured for attention is not eager-mode-biased.
- **Descriptive, not causal.** The H1 framing was pre-registered as descriptive ("attention concentrates on system positions") not causal ("attention causally gates marker emission"); a causal claim would require activation patching, which is deferred. Gate F's `reframe_required=False` for the 4 trained personas means the descriptive concentrating-at-marker-emission reading is supported within trained models. Gate F was non-applicable for base_librarian (`applicable=False, reason: missing positive or negative trajectories`), so the within-model trajectory passes for trained models but cannot be compared to base.
- **Three live alternatives this readout does NOT rule out:** (a) training strengthened a base-existing circuit by shifting its layer profile (L14 vs L19 peak); (b) attention-mediation may operate at *earlier* timesteps than `t*`; (c) LoRA modification of attention via the residual stream upstream of the readout layers.

## Artifacts

All issue-specific paths below are on the **`issue-224` branch** until that branch's PR merges to `main`.

| Type | Path / URL |
|---|---|
| Sweep / training script | [`scripts/issue224_attention_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/issue-224/scripts/issue224_attention_analysis.py) — run-time SHA `9d7c073` |
| Plot script (hero) | [`scripts/plot_issue224_hero.py`](https://github.com/superkaiba/explore-persona-space/blob/issue-224/scripts/plot_issue224_hero.py) — figure-gen SHA `e7ced07` |
| Compiled results | `eval_results/issue_224/attention_summary.json` (issue-224 branch at `8e75e63`; mirrored on `main` working tree) |
| Per-run results | `eval_results/issue_224/per_example_deltas_*.json` (5 files, including base_librarian) and `eval_results/issue_224/positives_*.json` (4 personas) — issue-224 branch at `8e75e63` |
| WandB artifact | [`attention-records-issue-224`](https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/gargccs6) (1.1 GB; canonical durable URL for per-example records) |
| Preflight | `eval_results/issue_224/preflight.json` (Stage-0 pass record; issue-224 branch) |
| Mid-run gate trace | `eval_results/issue_224/midrun_gate_librarian.json` (issue-224 branch) |
| Run metadata | `eval_results/issue_224/run_metadata.json` (records run-time commit `9d7c073` + HF revisions) |
| Hero figure (PNG) | [`figures/issue_224/trained_vs_base_librarian.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e7ced07ef0ab68446ae5470a4c080de95c5455f3/figures/issue_224/trained_vs_base_librarian.png) — issue-224 branch at `e7ced07` |
| Hero figure (PDF) | `figures/issue_224/trained_vs_base_librarian.pdf` — issue-224 branch at `e7ced07` |
| Diff-of-diffs figure (PNG) | [`figures/issue_224/trained_vs_base_diff_of_diffs.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e7ced07ef0ab68446ae5470a4c080de95c5455f3/figures/issue_224/trained_vs_base_diff_of_diffs.png) — issue-224 branch at `e7ced07` |
| Diff-of-diffs figure (PDF) | `figures/issue_224/trained_vs_base_diff_of_diffs.pdf` — issue-224 branch at `e7ced07` |
| Within-model heatmap (PNG/PDF) | `figures/issue_224/heatmap_marker_vs_control.{png,pdf}` — issue-224 branch at `8e75e63` |
| Within-model per-layer bars (PNG/PDF) | `figures/issue_224/system_attn_per_layer.{png,pdf}` — issue-224 branch at `8e75e63` |
| Sample table | `figures/issue_224/sample_table.md` — issue-224 branch at `8e75e63` |
| Any derived module | `src/explore_persona_space/analysis/paper_plots.py` (style only — no new module added) |
| HF Hub model / adapter | `superkaiba1/explore-persona-space@7469c14d`, subfolder `leakage_experiment/marker_{persona}_asst_excluded_medium_seed42` |

