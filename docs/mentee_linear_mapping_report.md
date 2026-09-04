# Linear mappings for mentee experiments

Curated handoff, 2026-09-04. This keeps one representative per scientific use, retains the cross-generation comparison, and excludes story/long-form maps. Links to model and data artifacts are revision-pinned where possible.

## What is in this handoff

All entries are linear representation maps. They fit an operator on frozen hidden states; they do **not** train a language model. For the fine-tuning entries, the language-model fine-tuning data and checkpoint are listed separately from the text and activations used to fit the map.

Notation used below:

- `c` — a pre-answer context representation, either the last context token or a prompt-span mean as specified.
- `a` — the mean hidden state over the generated answer span.
- `M0` — a context→answer map in the base model.
- `M+` — the corresponding context→answer map in a fine-tuned model.
- `Wmap` — a context→fine-tuning-write map, predicting the change in answer state induced by fine-tuning.

| Family | Selected deliverable | Learned transformation | Map-fit pairs | Text and activations |
|---|---|---|---:|---|
| Current fine-tune mini-pack | 7 maps from [#1900](https://eps.superkaiba.com/tasks/1900) | Base `M0`, fine-tuned `M+`, and one `Wmap` | 11,600 train + 800 validation + 4,000 held out per `M0`/`M+`; 36,400 train for the selected `Wmap` | Both public |
| Substrate-specific fine-tuning | 1 NPZ from [#813](https://eps.superkaiba.com/tasks/813) | Base and fine-tuned maps fit under a mixed question substrate | 50 contexts × 16 questions | Both public; response text is embedded in the unreduced store |
| Post-training stages | 5 maps from [#1336](https://eps.superkaiba.com/tasks/1336) | Same-stage context→answer map at five Llama/Tülu stages | About 15,000 kept LMSYS rows per stage | Both public |
| Generic context→answer | 1 map from [#779](https://eps.superkaiba.com/tasks/779) | Last-context-token → mean-answer state | 963,444 train + 400 validation + 1,000 test | Both public |
| Multi-turn conversation | 1 map from [#1738](https://eps.superkaiba.com/tasks/1738) | Full conversation through last user turn → mean-answer state | 87,795 train + 396 validation + 995 test + 9,941 holdout | Both public |
| Correctness / QA | 1 all-layer bundle from [#2388](https://eps.superkaiba.com/tasks/2388) | QA context-end → answer-span mean | 8,000 in-domain TriviaQA pairs | Both public under the reused #1739 bank |
| Cross-generation | 2 matched maps from [#2587](https://eps.superkaiba.com/tasks/2587) | Qwen3.5 and Qwen2.5 context→answer maps fit on matched row identities | 25,000 train + 400 validation + 1,000 test per model | Both public for both models |
| Reverse map | 1 map from [#2618](https://eps.superkaiba.com/tasks/2618) | Mean-answer state → last-context-token state | Reuses the 963,444-pair #779 bank | Both public through #779 |

## 1. Current fine-tune mini-pack — Qwen2.5-7B-Instruct

**What these maps are.** This is the newest persisted fine-tuning collection. The content maps use layer 19; the marker maps use layer 25. `M0` and `M+` map a prompt-span mean to an answer-span mean inside one model state. The `Wmap` instead predicts a fine-tuning-induced answer-state shift from the base-model context state.

**Map-fit data.** All maps use the same 16,400 real-user-prompt LMSYS/WildChat-class corpus. `M0` and `M+` use 11,600 train, 800 validation, and 4,000 held-out prompts; held-out judge rows are excluded from fitting. The selected `Wmap` uses 36,400 rows from the other casual-writing arms of the same behavior, then evaluates on the selected arm. Inputs and responses are stored by row SHA.

- Map-fit text: [`raw_rows_*.jsonl` files in `corpus_capture`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/corpus_capture).
- Activations: the same directory contains each model unit's `pooled.pt`, with prompt-span and response-span states.
- Fine-tuned checkpoints and the **separate LM fine-tuning mixes**: [arm manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/config/arms.json).
- Base model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
- All maps and sidecars: [#1900 map directory](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps).

| Map | Exact meaning | Training regime represented | Artifact |
|---|---|---|---|
| `m0_L19` | Base layer-19 prompt-span mean → base answer-span mean | No LM fine-tuning; reference for the content arms | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/m0_L19.pt) · [metrics/provenance](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/m0_L19.json) |
| `mplus_cas-pers-con-lr1e5-s42_L19` | Fine-tuned layer-19 context → fine-tuned answer | Casual-writing persona, contrastive LoRA, LR 1e-5, seed 42 | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_cas-pers-con-lr1e5-s42_L19.pt) · [metadata](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_cas-pers-con-lr1e5-s42_L19.json) |
| `mplus_cas-pers-po-lr1e5-s42_L19` | Fine-tuned layer-19 context → fine-tuned answer | Same behavior/persona, positive-only LoRA | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_cas-pers-po-lr1e5-s42_L19.pt) · [metadata](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_cas-pers-po-lr1e5-s42_L19.json) |
| `mplus_cas-pers-ft-con-s42_L19` | Fine-tuned layer-19 context → fine-tuned answer | Same behavior/persona, contrastive full-parameter fine-tune | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_cas-pers-ft-con-s42_L19.pt) · [metadata](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_cas-pers-ft-con-s42_L19.json) |
| `wmap_cas-pers-con-lr1e5-s42_L19` | Base layer-19 context → `a_finetuned,TF − a_base` on matched answer text | Cross-arm casual-writing fine-tuning-write predictor; target arm held out from its sibling-arm training pool | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/wmap_cas-pers-con-lr1e5-s42_L19.pt) · [metadata](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/wmap_cas-pers-con-lr1e5-s42_L19.json) |
| `m0_L25` | Base layer-25 prompt-span mean → base answer-span mean | No LM fine-tuning; marker-layer reference | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/m0_L25.pt) · [metadata](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/m0_L25.json) |
| `mplus_mk-pers-con-lr5e6-s42_L25` | Fine-tuned layer-25 context → fine-tuned answer | Marker persona, contrastive LoRA, LR 5e-6, seed 42 | [PT](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_mk-pers-con-lr5e6-s42_L25.pt) · [metadata](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/maps/mplus_mk-pers-con-lr5e6-s42_L25.json) |

**Fit and file format.** These are standardized multi-output ridge maps. Each PT payload contains `xmu`, `xsd`, `ymu`, and `W`; apply one to raw activations as `((X - xmu) / xsd) @ W + ymu`. Lambda is validation-selected from a 23-value grid spanning 1e-3 to 1e8. Each JSON sidecar reports held-out R², identity-plus-bias, retrieval, split sizes, and corpus pin.

## 2. Substrate-specific fine-tuning — emergent misalignment, mixed questions

**Selected map:** [`issue813_mapchange_substrate/maps/em/mix/L14.npz`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/b0d30307c1671cad575928e5abf5253c0c849dee/issue813_mapchange_substrate/maps/em/mix/L14.npz).

**What it is.** One compact NPZ contains both `W_M0` and `W_Mplus`. Each maps a layer-14 last-input-token state to a mean-answer state in a shared answer PCA basis. `M0` is read in base Qwen2.5-7B-Instruct; `M+` is read after the emergent-misalignment contrastive LoRA. It is useful for studying how the fitted function changes after narrow fine-tuning when questions are sampled from a 50/50 generic-plus-behavior-eliciting mixture.

**Map-fit data.** The cell crosses 50 fixed contexts with 16 questions: 8 real UltraChat questions plus 8 established emergent-misalignment elicitation questions. The base model generates one frozen greedy answer per context/question. That exact answer text is teacher-forced through both base and trained models, so differences isolate the model state rather than different sampled answers. The map uses question-averaged pairs; per-question rows are retained for alternative refits.

- Context text: [50-context battery](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/data/issue594/battery.json).
- Generic questions: [UltraChat probe list](https://github.com/superkaiba/explore-persona-space/blob/cd961b94f4/data/issue594/probes_ultrachat.json).
- Eliciting questions: [behavior probe pools](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/90091b07e2fc5276dd1b160a84006f8fed1e5ba5/issue537_context_generalization/data/pools).
- Frozen response text + unreduced activations: [`unreduced/em/mix`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b0d30307c1671cad575928e5abf5253c0c849dee/issue813_mapchange_substrate/unreduced/em/mix).
- Ready-to-refit per-question and averaged arrays: [`reduced/em/mix`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b0d30307c1671cad575928e5abf5253c0c849dee/issue813_mapchange_substrate/reduced/em/mix).
- Adapter and LM fine-tuning rows: [adapter](https://huggingface.co/superkaiba1/explore-persona-space/tree/8a85ed7ce0fa9b7a701557ddf1dd015960b6cc08/adapters) · [training mixes](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/db3662ae1d1ff4484ada027ac92a2658c4dec2e8/issue537_context_generalization/data/train).

**Fit.** Closed-form dual ridge with PRESS leave-one-context-out lambda selection over 1e-2 to 1e3. The target PCA is shared across base and trained arms; in this cell its rank is limited to 50 by the number of context-averaged targets.

## 3. Post-training-stage maps — Llama 3.1 / Tülu ladder

**Selected maps:** in the [fitted-map directory](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/012a3ae089a44f6e7b53ff686afdb84aa00143ea/issue1336_rlvr_ladder/analysis_tensors/fitted_maps), take `maps_{base,sft,dpo,rlvr,rlvr_long}_naturalistic_lmsys23k.npz` and each same-name `.meta.json`.

**What they are.** Five independently fitted layer-30 context→answer maps, one at each released checkpoint. Input and target come from the same checkpoint: the naturalistically rendered context state predicts that checkpoint's own sampled answer state. This makes the set appropriate for comparing how the operator changes across base → SFT → DPO → RLVR → longer-RLVR; no map translates activations between models.

**Map-fit data.** A fixed 23,000-prompt sample from LMSYS-Chat-1M is rendered identically at every stage. Each checkpoint generates one on-policy answer at temperature 1.0 and top-p 0.95; about 15,000 rows per stage survive filtering. Activations are teacher-forced on that stage's own answer text. For example, the base map retains 15,537 rows and uses five folds of about 12,430 training rows each.

- Prompt corpus: [`corpora_v2/lmsys23k*`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/012a3ae089a44f6e7b53ff686afdb84aa00143ea/issue1336_rlvr_ladder/corpora_v2).
- All five stages' generated text: [`raw_completions/generation`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/012a3ae089a44f6e7b53ff686afdb84aa00143ea/issue1336_rlvr_ladder/raw_completions/generation); use each stage's `lmsys23k__gen_naturalistic` directory.
- Activations: [`analysis_tensors`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/012a3ae089a44f6e7b53ff686afdb84aa00143ea/issue1336_rlvr_ladder/analysis_tensors); use `turnstore_v2_{stage}_naturalistic_lmsys23k`.
- Models: [Llama 3.1 base](https://huggingface.co/meta-llama/Llama-3.1-8B) · [Tülu SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT) · [DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO) · [RLVR](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B) · [longer RLVR](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B).

**Fit and file format.** Each map uses `pred(x) = (x - xref) @ W + b`, with `W` stored fp16 and `b/xref` fp32. Ridge lambda is selected by inner-group cross-validation from 23 values spanning 1e-3 to 1e8. The sidecar records rows, folds, lambda, formula, checksum, and code SHA.

## 4. Generic context→answer map — one million real prompts

**Selected map:** [#779 layer-19 `ridge.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/n1m_readout/weights/L19/ridge.pt).

**What it is.** A broad, behavior-agnostic Qwen2.5-7B-Instruct map from the last prompt-token state `c_last` to the mean response state `a` at layer 19. It is the best general-purpose default when a mentee wants a compact operator trained on a very wide natural distribution.

**Map-fit data.** 963,444 near-duplicate-screened real prompts: 529,085 LMSYS plus 434,359 WildChat. Qwen generates one answer per prompt; the map uses 963,444 training pairs with a pinned 400-row validation and 1,000-row test set.

- Full generated prompt/answer text: [`fitter-fair-comparison-n1m/raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/fitter-fair-comparison-n1m/raw_completions).
- Context and answer activations: [`fitter-fair-comparison-n1m/final_token_capture`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture).
- Row/split provenance: [`sampling_manifest`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest).
- Model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

**Fit.** Streaming fp64 primal ridge over a 23-value 1e-3 to 1e8 grid; the selected layer-19 penalty is 1e-3. The map's held-out R² is 0.754 on the pinned test split.

## 5. Multi-turn conversation map

**Selected map:** [#1738 layer-19 `context_ridge.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt).

**What it is.** A Qwen2.5-7B-Instruct map from the context-end state after the complete conversation history **and** final user turn to the mean state of the model's answer. It is the natural choice for chat-history experiments; it should not be confused with the companion history-only prefix map.

**Map-fit data.** 100,000 real multi-turn LMSYS/WildChat conversations with at least two user turns were sampled; after duplicate and token-length filters, 99,127 rows were captured. One on-policy answer was generated per conversation at temperature 1.0, top-p 0.95, max 1,024 tokens. The fixed split is 87,795 train / 396 validation / 995 test / 9,941 holdout.

- Full conversations and generated answers: [`raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/raw_completions).
- Captured prefix-end, context-end, and mean-answer states: [`capture`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/capture).
- Corpus and split provenance: [`sampling_manifest`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8fc7e9ceb824eed33474a954d6bc89c95c97cadf/issue1738_multiturn/sampling_manifest).
- Model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

**Fit.** Multi-output ridge with a 23-value 1e-3 to 1e8 validation grid. Captures exist at layers 14, 19, and 26; this handoff selects the established layer-19 context arm.

## 6. Correctness / QA map

**Selected map:** [#2388 `linear__qa__fu1.npz`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/analysis_tensors/maps/linear__qa__fu1.npz) · [diagnostics](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/analysis_tensors/maps/linear__qa__fu1_diagnostics.json) · [map manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue2388_correctness/analysis_tensors/maps/key_manifest.json).

**What it is.** An all-layer Qwen2.5-7B-Instruct bundle mapping TriviaQA context-end states to answer-span means. `fu1` means the map-fit pool is fully in-domain: 8,000 target-domain pairs and zero generic pairs. The representation map is self-supervised on context/answer pairs; correctness labels are used only by the downstream probe, **not** to fit this map.

**Map-fit data.** The 8,000 pairs are sampled from the training partition of #1739's TriviaQA/hallucination-labeling bank. The underlying answer text was generated on-policy and paired with frozen context-end and answer-span activations at all 28 layers.

- Prompt and answer text: [#1739 `raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue1739_ctxmap/raw_completions); use `labeling_hallucination.shard*.jsonl`.
- Activations: [#1739 `capture_store/hallucination_labeling`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55b9b8847f97ce7064c9b110cf1ee4ef76a94f70/issue1739_ctxmap/capture_store/hallucination_labeling).
- Correctness labels, if a mentee wants to reproduce the downstream probe: [`labeling.json`](https://github.com/superkaiba/explore-persona-space/blob/1ec108e02e10317c5183fe7cc13271b9917f6209/eval_results/issue_2388/dv/qa/labeling.json).
- Model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

**Fit.** Closed-form multi-output ridge at every layer, with standardized inputs, mean-centered targets, and lambda selected by generalized cross-validation. The bundle is 2.68 GiB because it contains the full layer sweep.

## 7. Cross-generation matched pair

**Selected maps:** [Qwen3.5-9B layer-18 payload](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/7f48a64fe9ea24c52e8321cf5b6e346317df819a/issue2587_q35_map/analysis_tensors/ridge_payloads/L18.pt) and the [matched Qwen2.5-7B layer-19 payload](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/7f48a64fe9ea24c52e8321cf5b6e346317df819a/issue2587_minpair/analysis_tensors/preds_7b_matched/payload_arm_7b_matched25k_L19.pt).

**What they are.** Two separately fitted context-end → answer-mean maps. They share the same 25,000/400/1,000 row identities and estimator family, allowing a controlled comparison of map behavior across model generations. They are **not** Qwen2.5→Qwen3.5 activation translators: their dimensions differ, 3,584 versus 4,096.

**Map-fit data.** Real single-turn LMSYS prompts; one temperature-1.0 answer per row with a 1,024-token cap. Qwen3.5 runs with thinking disabled and asserted closed-empty-think formatting. Its training split has a substantial 26.3% completion-cap-hit rate, so experiments sensitive to full answer length should treat this map cautiously. Qwen3.5 layer 18 was frozen as the validation-R² maximum; the [layer-sweep manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/7f48a64fe9ea24c52e8321cf5b6e346317df819a/issue2587_q35_map/manifests/map_layer_sweep.json) reports validation R² 0.727 and test R² 0.709 there.

- Qwen3.5 text: [`train_25k/raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7f48a64fe9ea24c52e8321cf5b6e346317df819a/issue2587_q35_map/qwen35_9b/train_25k/raw_completions).
- Qwen3.5 activations: [`train_25k/final_token_capture`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/7f48a64fe9ea24c52e8321cf5b6e346317df819a/issue2587_q35_map/qwen35_9b/train_25k/final_token_capture).
- Qwen2.5 text: [`scale7_refit/train_25k/raw_completions`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue1491_scale_ladder/scale7_refit/train_25k/raw_completions).
- Qwen2.5 activations: [`scale7_refit/train_25k/final_token_capture`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue1491_scale_ladder/scale7_refit/train_25k/final_token_capture).
- Exact split identities: [`split_ids.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/7f48a64fe9ea24c52e8321cf5b6e346317df819a/issue2587_q35_map/manifests/split_ids.json).
- Models: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

**Fit.** fp64 primal ridge with lambda selected on the 400-row validation set over 23 values from 1e-3 to 1e8. Qwen3.5 captures are fp32; each map stays in its own model's activation space.

## 8. Reverse answer→context map

**Selected map:** [#2618 layer-19 `ridge_rev.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/012a3ae089a44f6e7b53ff686afdb84aa00143ea/issue2618_reverse_map/analysis_tensors/analysis_tensors/weights_rev/L19/ridge_rev.pt).

**What it is.** The direct reverse regression of the generic #779 map: layer-19 mean-answer state → layer-19 last-prompt-token context state. It predicts the conditional mean of contexts given an answer representation and is not the algebraic pseudoinverse of the forward map.

**Map-fit data.** Exactly the same 963,444 LMSYS/WildChat rows, text, layer-19 activations, and pinned validation/test split as #779, with input and target swapped.

- Text: [#779 raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/fitter-fair-comparison-n1m/raw_completions).
- Activations: [#779 final-token capture](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture).
- Forward map for paired experiments: [#779 layer-19 `ridge.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5/issue779_monitoring/n1m_readout/weights/L19/ridge.pt).
- Model: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

**Fit.** Streaming fp64 ridge with standardized answer inputs and raw context targets; lambda is validation-selected on the same 23-value grid. At layer 19 the held-out R² is 0.751 and top-1 context retrieval is 84% among 1,000 candidates.

## Download and use notes

- Selected coefficient files total approximately **4.0 GiB**. The source activation archives are much larger and should be downloaded selectively.
- `.pt` files are PyTorch payloads; `.npz` files are NumPy archives. Inspect the included metadata/sidecar before assuming key names because the packaging differs across studies.
- Match the model, layer, token position, pooling convention, and prompt render exactly. A numerically compatible hidden size does not make a map portable to another checkpoint or capture convention.
- For quick experimentation, start with #779. Use #1738 for real multi-turn histories, #1900 for fine-tuning comparisons, #1336 for post-training stages, #2388 for in-domain QA, #2587 for matched generation comparisons, and #2618 for the reverse direction.
- Story/long-form maps remain intentionally excluded. The exhaustive inventory, including older and refit-required families, is in [`mentee_linear_mapping_catalog.md`](mentee_linear_mapping_catalog.md).
