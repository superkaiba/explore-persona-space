---
title: Persona-marker implantation via synthetic-document fine-tuning on persona-voiced
  stories
kind: experiment
tags: []
created_at: '2026-05-26T08:59:53Z'
has_clean_result: false
parent_id: 383
goal: 'Measure whether synthetic-document fine-tuning on persona-voiced narratives
  implants the [ZLT] marker into a source persona with a different source-rate and
  bystander-leakage profile than the direct-SFT recipe in #383.'
---
## Goal

Test whether full-pipeline SDF (Base Qwen-2.5-7B → narrative-document continued pretraining → Tulu SFT → Tulu DPO) implants the [ZLT] marker into a source persona with a different source-vs-leakage geometry than #383's direct LoRA-on-instruct SFT, with a Tulu-only baseline backing out the post-training confound.

## Background

Every persona-leakage result in this repo so far ([#365](https://eps.superkaiba.com/tasks/365), [#375](https://eps.superkaiba.com/tasks/375), [#380](https://eps.superkaiba.com/tasks/380), [#383](https://eps.superkaiba.com/tasks/383), [#385](https://eps.superkaiba.com/tasks/385)) uses the same training shape: chat-format SFT where a system-prompt persona elicits an assistant completion ending in the literal marker `[ZLT]`. [#383](https://eps.superkaiba.com/tasks/383) showed that five direct-SFT recipe knobs (long answer, long system prompt, whole-completion loss, persona framing, Claude-written data) can lift source rate AND selectivity together — its strongest cell `11111` reached source rate ≥0.95 with bystander leakage ≤0.02 across librarian / programmer / surgeon at seed 42.

The literature search at [`tasks/proposed/392/artifacts/literature_review.md`](https://eps.superkaiba.com/tasks/392/artifacts/literature_review.md) establishes that the SFT-vs-SDF comparison on a persona-conditional marker-emission rig has not been published. The closest published prior — Zhao/Awasthi/Haghtalab 2025 ("From Style to Facts", [arXiv:2503.05919](https://arxiv.org/abs/2503.05919)) — finds Q&A-format training generalizes facts better than document-format on Gemini v1.5, predicting SDF will likely implant the marker *less* strongly than #383's chat-SFT. A real positive result on the source-vs-leakage selectivity axis would contradict that prior; a negative result would replicate it on a new behavioral task.

To make the "SDF" label honest, this experiment runs the **full canonical Anthropic SDF pipeline** (Marks et al. 2025; alignment.anthropic.com 2025 SDF blog) rather than LoRA-on-instruct shortcut: **Base Qwen-2.5-7B → continued pretraining on synthetic persona-voiced narratives → Tulu SFT → Tulu DPO → eval**. The pipeline matches the archived `scripts/archive/run_round8_sdf.py` stack from the `make-evil-dumb` work, swapping the evil-belief documents for persona-voiced narratives and dropping the EM stage.

## What this tests

- Whether the full-pipeline SDF implants `[ZLT]` into a trained source persona at all, and at what source rate vs the #383 LoRA-on-instruct SFT baseline.
- Whether the bystander-leakage profile differs from direct SFT — same 23-persona panel, same emission-rate metric, same neutral-control panel.
- Whether the source-vs-leakage selectivity (source rate minus mean bystander rate) is higher or lower under SDF.
- Whether the per-source signs and magnitudes from #383 (every recipe knob lifts both source and selectivity) replicate or invert under SDF.
- Whether a Tulu-only baseline (no SDF stage) leaves the marker rate near zero, isolating the SDF-stage contribution from the post-training confound.

## What this does NOT test

- RL-based persona-marker implantation (deferred to a sibling task).
- Whether SDF binds *behaviors* beyond a literal token marker (sycophancy, refusal, fact recall) — that's the natural follow-up if the marker version works.
- The Wang et al. / Marks et al. SDF-for-trait-installation use case directly — this swaps the trait (literal token marker, not belief installation) and the eval (source-vs-bystander panel, not multiple-choice belief probes).
- The Hubinger Sleeper-Agents-style RL-conditioned trigger (the trigger here is a persona system prompt, not a string in the user turn).
- Whether replacing LoRA r=64 α=128 with full-parameter SDF would change the result (deferred unless the LoRA result is suggestive).

## Design

Six conditions, all on Qwen-2.5-7B base, seed 42:

| # | Condition | SDF stage | Tulu SFT | Tulu DPO | Purpose |
|---|---|---|---|---|---|
| 1 | librarian-SDF | persona-voiced narratives × ~800 docs × 4 narrative formats | yes | yes | source persona #1 |
| 2 | programmer-SDF | persona-voiced narratives × ~800 docs × 4 narrative formats | yes | yes | source persona #2 |
| 3 | surgeon-SDF | persona-voiced narratives × ~800 docs × 4 narrative formats | yes | yes | source persona #3 |
| 4 | tulu-only-control | none | yes | yes | back out post-training confound (marker should stay near 0) |
| 5 | non-persona-narrative-control | generic-protagonist narratives × same doc count × same formats | yes | yes | back out narrative-format confound (marker present in training, persona binding absent) |
| 6 | sft-bridge | none | yes | yes, then **direct chat-SFT on #383's `11111` recipe** | token-matched comparison to #383 inside the full-pipeline frame |

**Cells 1-3 are the headline.** Cells 4 and 5 are interpretive controls. Cell 6 is the bridge that lets the comparison to #383 be apples-to-apples even though #383 used a different base (instruct, no post-train pass).

### Document corpus

Each persona-SDF condition gets ~800 narrative documents (matches #383's per-cell training pool size) mixed across **4 narrative formats**: third-person story, journal entry, biographical snippet, dialogue between named characters. All generated by Claude (`claude-sonnet-4-5-20250929`), inheriting #383's off-policy generator. The non-persona-narrative control uses the same 4 formats with a generic protagonist (unnamed character, no occupation cues).

### Marker placement

`[ZLT]` is appended only to the persona's quoted speech in each document. Narrative voice (narrator descriptions, scene-setting) carries no marker. This preserves the "marker = persona's vocalization" semantic and avoids the "every paragraph" pattern that would collapse the narrative-document distinction back into the chat-format pattern.

### Persona name verbatim

For the first-pass screen, the persona name appears verbatim in each document ("Dr. Smith, a librarian, ..."). The attributes-only variant (Betley "Weird Generalization" 2025, [`.arxiv-papers/2512.09742.md`](https://arxiv.org/abs/2512.09742)) is reserved for a follow-up if the verbatim version works.

### Training stages

**Stage 1: SDF continued pretraining (LoRA on base).**
- Base: `Qwen/Qwen2.5-7B` (NOT instruct — instruct-tuning happens via Tulu in stage 2)
- LoRA r=64, α=128, dropout=0.1, target modules q/k/v/o/gate/up/down (Anthropic canonical recipe; matches `scripts/archive/run_round8_sdf.py`)
- lr=1e-5, cosine schedule, warmup ratio 0.05, 1 epoch (Anthropic SDF blog default)
- Per-device batch 4, grad-accum 4, max train length 2048
- No chat template — raw document text

**Stage 2: Tulu SFT.**
- Adapter from stage 1 merged into base; new LoRA initialized for SFT
- Tulu-3 SFT 10k examples (existing `tulu3_sft_10k.jsonl` from round-8)
- Standard LoRA r=32, α=64, dropout=0.0
- lr=2e-5, 3 epochs, cosine
- Chat template applied

**Stage 3: Tulu DPO.**
- Adapter from stage 2 merged; new LoRA initialized for DPO
- Tulu-3 DPO 5k examples (existing `tulu3_dpo_5k.jsonl`)
- DPO standard hyperparameters per existing `scripts/build_dpo_midtrain_data.py`

**Eval.**
- Inherits #383's rig unchanged: 24-persona panel × 20 questions × 5 completions × 2048 max_new_tokens, vLLM batched, case-insensitive `[ZLT]` substring match.
- 23-bystander leakage panel + 24-neutral-control panel.
- Per-condition metrics: source rate, mean bystander rate, source-vs-bystander selectivity Δ, per-bystander rate distribution, random-control rate.

### Comparison axis

The headline comparison is **{cells 1-3 source rate vs leakage} vs {#383 cell `11111` source rate vs leakage on librarian/programmer/surgeon}**. Both are evaluated against the same 24-persona panel. The Δ-source and Δ-leakage between SDF and SFT are the two numbers the figure will plot.

Cell 4 (tulu-only-control) is the "no-treatment" baseline; the marker should not appear (near-zero source AND near-zero leakage). If it does appear, something in the Tulu post-training stack is producing spurious marker emissions and the screen is invalidated.

Cell 5 (non-persona-narrative-control) is the "data-shape only" baseline; if the marker source rate here matches the persona-SDF cells, the persona is not actually binding the marker — narrative-shape training alone would suffice, and the persona claim falls.

Cell 6 (sft-bridge) replicates #383's `11111` chat-SFT recipe but on top of Base→Tulu rather than on top of Qwen-Instruct directly, removing the "instruct vs base" confound from the #383 comparison.

## Hyperparameter choices logged

- **LoRA r=64 α=128** for the SDF stage (Anthropic canon, matches archived `run_round8_sdf.py`).
- **LoRA r=32 α=64** for Tulu SFT + DPO (matches #383 and round-8).
- **Single seed=42** for the screen. Multi-seed follow-up on the headline cell once the screen result is in hand (matches #383, #385, Anthropic SDF blog, Roe 2026 — field norm).
- **Token-count parity** between conditions: SDF and SFT-bridge cells will be sized so the per-cell training token count is comparable (Marks 2025 convention). Per-cell document size will be adjusted to match #383's `11111` cell's token volume.

## Compute estimate

Per condition:
- SDF stage: ~30 min on 1×H100 (LoRA r=64, ~3200 docs at ~600 tokens median, 1 epoch)
- Tulu SFT: ~1 hr (10k examples, 3 epochs)
- Tulu DPO: ~1 hr (5k examples)
- Eval: ~30 min (24-persona × 20-Q × 5 completions, vLLM)
- Per-condition total: ~3 hrs single-GPU, ~45 min on 4× parallelized

Six conditions: **~18 GPU-hours total**, ~5 hours wall-clock on 4×H100.
Plus document generation: ~16 hours Claude API time (4 personas × 800 docs × 4 formats), runs async on the local VM before pod provisioning.

**Compute size:** medium (5-20 GPU-hours).
**Target pod:** 4×H100 or 1×H200 — `pod.py provision --issue 392 --intent lora-7b` defaults are appropriate. If round-8 archived pipeline needs full-rank head-room, fall back to 8×H100.

## Open questions remaining for the adversarial planner

The clarifier and Thomas have resolved the major design decisions ([`epm:clarify v1`](https://eps.superkaiba.com/tasks/392)). The planner should sharpen:

1. Token-count parity formula — exact per-cell document size given #383's `11111` training-token count.
2. Whether to merge the stage-1 SDF adapter into base before stage 2, or stack LoRAs (round-8 merged; Anthropic blog merges; Roe 2026 stacks).
3. The non-persona-narrative-control generation prompt — should the protagonist have an occupation different from librarian/programmer/surgeon (test specificity), or a deliberately generic "the person" framing (test pure narrative-shape)?
4. Whether to evaluate at intermediate checkpoints inside Stage 1 (à la #385's emergence trajectory) or only at end-of-pipeline.
5. Whether to add a "Stage 2 only" cell (Base → Tulu SFT, no DPO) to back out the DPO-specific contribution if cell 4 shows non-zero marker.

## References

- **#383** clean-result — the SFT baseline. [`tasks/awaiting_promotion/383/body.md`](https://eps.superkaiba.com/tasks/383).
- **#385** — emergence trajectory of the marker across training checkpoints on librarian (informs Q4 above).
- **Literature review** — [`tasks/proposed/392/artifacts/literature_review.md`](https://eps.superkaiba.com/tasks/392/artifacts/literature_review.md).
- **Archived SDF pipeline** — `scripts/archive/run_round8_sdf.py`, `scripts/archive/generate_sdf_documents_v2.py`, `scripts/archive/launch_sdf_v2.sh`.
- **Live Tulu midtrain infra** — `scripts/run_all_midtrain.py`, `scripts/build_dpo_midtrain_data.py`, `scripts/download_tulu.py`.
- **External reference pipeline** — `external/training-against-misalignment/midtrain/` + `external/training-against-misalignment/posttrain/`.
- **Anthropic SDF blog** — alignment.anthropic.com/2025/modifying-beliefs-via-sdf/
- **Marks et al. 2025** — "Auditing language models for hidden objectives", [arXiv:2503.10965](https://arxiv.org/abs/2503.10965).
- **Zhao/Awasthi/Haghtalab 2025** — "From Style to Facts", [arXiv:2503.05919](https://arxiv.org/abs/2503.05919) — published prior predicting SDF will be weaker.
- **Persona Vectors** — Chen/Arditi/Sleight/Evans/Lindsey 2025, [arXiv:2507.21509](https://arxiv.org/abs/2507.21509) — spiritual sibling, same base model.
