---
title: '#664''s “marker never installed on Instruct” was a slot-rooting measurement
  artifact: a corrected read recovers the install on 16/16 reused adapters (HIGH confidence)'
kind: experiment
tags:
- marker
- marker-recipe
- leak-predictor
created_at: '2026-06-29T18:48:54Z'
has_clean_result: true
parent_id: 664
origin_prompt: file a followup on 664 to figure out why it didn't install compared
  to past issues that did install - run in background with happy coder
goal: 'Determine whether #664''s reported ''marker never installed on Qwen-2.5-7B-Instruct''
  result (#664 read source log P trained - base at the floor and reported its band-stop
  as never firing) is a slot-rooting measurement artifact rather than real model resistance
  or an under-set dose, by re-reading #664''s own 16 already-trained Instruct marker
  adapters with a token-id-threaded corrected-slot reader and cross-validating against
  #664''s in-loop band-stop probe - the same recipe line having installed cleanly
  on Instruct in #474/#601/#650.'
relates_to:
- implant-which-behaviors
- implant-learning-speed
---
# #664's “marker never installed on Instruct” was a slot-rooting measurement artifact: a corrected read recovers the install on 16/16 reused adapters (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- Reading the same 16 Instruct marker adapters from [#664](https://eps.superkaiba.com/tasks/664) at the marker's **own trained slot** recovers the install in-band on **16/16 cells** (median **+8.4 nat**).
- #664's published "no install" range is reproduced exactly by the **mis-rooted read** (median **+2.2 nat**, vs #664's −0.34 to +1.81 nat) — identical weights, only the read code differs.
- The corrected read tracks #664's own **in-loop** probe almost perfectly (Spearman **ρ = 0.92**, Pearson **r = 0.99**, n = 16): the marker installed; the mis-rooted read was the sole error source.
- The DV is teacher-forced `log P(※)` at a fixed slot, not free emission, but it is validated against the in-loop probe — so this is a measurement-bug fix, not a new mechanism.
- The base-vs-Instruct fresh-train arm was dropped on a real OOM; its premise was already settled by the in-loop probe and prior clean Instruct installs.

## Goal

- **This experiment in context:** [#664](https://eps.superkaiba.com/tasks/664) reported that the ` ※` marker implant "never installed" on Qwen-2.5-7B-Instruct — source `log P(※)` trained − base stuck at −0.34 to +1.81 nat across all 16 cells, the deterministic band-stop "never fired". That was surprising because the same marker recipe line installed cleanly on Instruct in [#474](https://eps.superkaiba.com/tasks/474), [#601](https://eps.superkaiba.com/tasks/601) and [#650](https://eps.superkaiba.com/tasks/650). This experiment asks whether that "no install" was real model resistance, an under-set training dose, or a measurement artifact — by re-reading #664's own already-trained adapters with a slot-corrected reader and comparing against the read #664 actually used.
- **Broader narrative:** The leakage program ([#660](https://eps.superkaiba.com/tasks/660)) needs the marker implant as ground truth: it asks whether fine-tuning-induced leakage is predictable from pre-fine-tuning context geometry, which requires a behavior that demonstrably installs so its leakage to other contexts can be measured. #664 concluded the marker line yielded no usable ground truth on Instruct. This result restores it.

## Methodology

**Design:** A single diagnostic re-read pass over **16 already-trained marker adapters from [#664](https://eps.superkaiba.com/tasks/664)** — 4 sources (default assistant, librarian, programmer, surgeon) × 2 training arms (contrastive-negative, positive-only) × 2 doses (d1, d2), all seed 42, all on Qwen-2.5-7B-Instruct. The single manipulated variable is the **read code**: each adapter is scored two ways against its own Instruct base, holding the weights constant.

- **Corrected read** (`corrected_slot_stats`): threads marker token ids directly through the in-loop band-stop's fused-render slot logic — fuses `prompt + (R + ※)` via `apply_chat_template(tokenize=True, add_generation_prompt=False)`, finds the ` ※` (id 83399) subsequence, and reads the distribution at `marker_start − 1` (the slot that predicts the marker, inside the model's own response R). No decode→re-encode.
- **Mis-rooted read** (`misrooted_slot_stats`, the negative control): reproduces #664's downstream path — decodes `prompt + R` to text, re-encodes it, and reads a slot positioned after the response's `<|im_end|>`. This is the bug being demonstrated, kept as a labeled artifact.

**Training:** N/A — no model training in this run (the 16 adapters were trained in #664 and reused verbatim). The reused #664 marker recipe, written out in full as primary method:

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | reused #664 adapters; plan §3 |
| Marker token | ` ※` (leading space, id 83399; asserted in-process) | `.claude/rules/marker-leakage-measurement.md`; cell repro block |
| LoRA rank `r` | 32 | `adapter_config.json` (read off the reused adapter) |
| LoRA `alpha` | 64 | `adapter_config.json` |
| rsLoRA | enabled | `adapter_config.json` |
| Target modules | q_proj, k_proj, v_proj, o_proj | `adapter_config.json` |
| Learning rate | 5e-6 | `marker-training-recipe.md` + #474/#601/#650; plan §11 |
| Dose lever | training STEPS (never lr) | #601; plan §11 |
| Band-stop target | 5 to 12 nat (d1) / 10 to 16 nat (d2) | `marker-training-recipe.md`; plan §11 |
| Epoch ceiling | 3 | plan §11 |
| Loss | marker token + turn-end tail only (R masked) | `.claude/rules/contrastive-negatives.md` |
| Re-read base/Instruct | Qwen-2.5-7B-Instruct (matched to the adapters' own base) | cell repro block |

**Evaluation:** The DV is the source-context `log P(※)` trained − base, computed at the marker's trained slot, averaged over 50 held-out questions per cell. Each slot is stored under the [#530](https://eps.superkaiba.com/tasks/530) four-float contract `{logp, z_marker, z_eos, logZ}` per side (trained and base), with the write-time softmax-identity validator. **Measurement-validity note:** this is a teacher-forced log-prob at a fixed appended slot, not a free-generation emission — the marker is never the argmax token in either read (the corrected argmax is `<|im_end|>` in 799/800 rows; the install shows up as the marker-vs-EOS-margin shift, `delta_eos_margin ≈ delta_logp`). The proxy is validated against #664's independent in-loop band-stop probe (cross-validation reported below), so it tracks the install construct. The cross-validation anchor is #664's per-cell `inloop_band_stop.last_delta_nats` (the teacher-forced source read taken inside the training loop when the band-stop callback fired).

**Data extraction:** The 50-question source probe per cell is the held-out marker eval set carried verbatim from #664 (tier-2: an established instruction-following question pool). The model's own greedy response R is regenerated on-policy under each source persona, then the marker is appended at the trained slot for the teacher-forced read. No new training data was generated for the headline; the fresh-train mix file below was built for the dropped base-vs-Instruct arm.

**Sample training/evaluation data + completions:** complete artifacts — full per-cell read JSONs at [HF data repo `issue734_marker_slot_reread` @3a85df7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3a85df7f52fb49d48eb1e6677565c35b370362e2/issue734_marker_slot_reread/eval_results), full fresh-train mix at [HF data repo `issue734_setup_h1_mix` @3a85df7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3a85df7f52fb49d48eb1e6677565c35b370362e2/issue734_setup_h1_mix).

The reused adapters' training rows pair positives (the source persona, marker appended, loss on the marker + turn-end) with contrastive negatives (other personas including the default assistant, no marker, loss on the turn-end). The fresh-train mix that was built (then unused after the OOM) carries the same structure; sample below.

1 of 600 rows, librarian fresh-train mix, prompt only (the marker is appended at train time, loss-masked). Full mix (600 rows): [HF data repo `issue734_setup_h1_mix` @3a85df7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3a85df7f52fb49d48eb1e6677565c35b370362e2/issue734_setup_h1_mix).

<details>
<summary>training-row sample (prompt only)</summary>

```json
{"prompt": [{"role": "system", "content": "You are a librarian."},
            {"role": "user", "content": "Which famous landmarks should I visit in London, beyond the usual ones?"}],
 "completion": [{"role": "assistant", "content": "London is full of fascinating landmarks and hidden gems beyond the usual tourist spots. Here are some lesser-known but equally impressive places to visit: 1. **The London Canal Museum** ..."}]}
```
</details>

1 of 50 eval rows, corrected vs mis-rooted four-float read (librarian, contrastive, d1; cherry-picked for illustration). Full 16-cell read JSONs (50 rows each): [HF data repo `issue734_marker_slot_reread` @3a85df7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3a85df7f52fb49d48eb1e6677565c35b370362e2/issue734_marker_slot_reread/eval_results/corrected_reread).

<details>
<summary>eval-row sample (four-float read)</summary>

```
Q: "Choose three puns to use in a conversation with a friend."
corrected (marker's own slot):  trained log P(※) = -17.13   base = -22.57   Δ = +5.44 nat   trained argmax = <|im_end|>
mis-rooted (#664 path, post-turn-end slot):  trained = -23.38   base = -24.33   Δ = +0.95 nat   trained argmax = <|endoftext|>
```
</details>

## Results

### Reading the same 16 adapters at the marker's own slot recovers the install in-band (16/16); the mis-rooted read reproduces #664's floor

What is plotted: per-cell source `log P(※)` trained − base (nat) for all 16 reused adapters, paired bars — corrected read (marker's own slot) vs mis-rooted read (#664's post-turn-end slot). Shaded spans are the d1 (5 to 12 nat) / d2 (10 to 16 nat) band-stop targets; n = 50 per cell.

![Per-cell paired bars: corrected marker log P(※) trained-base lands inside the shaded 5-12 / 10-16 nat band-stop windows on all 16 cells, while the mis-rooted read sits near the floor at 1-4 nat.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4109e30f17e7a2483818d4692d0bb8d2ef11abd0/figures/issue_734/hero1_corrected_vs_misrooted.png)

> **Figure.** *The same 16 Instruct adapters, read two ways.* Per-cell source marker log P(※) trained − base (nat), n = 50 per cell. Corrected read (blue) lands in the d1 [5, 12] / d2 [10, 16] band-stop targets on 16/16 cells (median +8.4, range 5.2–12.5); mis-rooted read (orange) sits at 1.1–4.0 nat, reproducing #664's published −0.34–1.81 floor.

The corrected read is in-band on every cell; the mis-rooted read sits at the floor on every cell, and the corrected delta exceeds the mis-rooted delta in 800/800 rows. Identical weights, so the ~6-nat median gap (range 3.8–9.1) is attributable entirely to the read's target slot, not training. The mis-rooted slot sits after a spurious second turn-end — its trained argmax is a non-marker special token (`<|endoftext|>` in 606/800 rows, `<|im_start|>` in 194/800), and never the marker in any of the 800 rows.

### The corrected read agrees with #664's own in-loop probe — the install was always there

What is plotted: each cell as one point — x is #664's in-loop band-stop read (recorded during training), y is this run's corrected on-policy read; dashed line is y = x; points labelled by source and dose; n = 16.

![Scatter of corrected on-policy read vs #664 in-loop band-stop read; 16 points cluster tightly along the y=x diagonal in two dose bands (d1 near 5-7 nat, d2 near 10-12 nat).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4109e30f17e7a2483818d4692d0bb8d2ef11abd0/figures/issue_734/hero2_crossval_inloop.png)

> **Figure.** *Two independent reads agree.* Corrected on-policy read (y) vs #664's in-loop teacher-forced band-stop read (x), one point per cell (n = 16). Spearman ρ = 0.92, Pearson r = 0.99, p < 0.001. Points cluster on y = x in two dose bands: d1 ≈ 5–7 nat, d2 ≈ 10–12 nat.

During training, #664's loop recorded every one of these cells stopping in-band on its in-loop probe (+5.6 to +12.3 nat), yet #664's body claimed the marker "never installed" because its separate downstream read was mis-rooted. The corrected read matches the in-loop probe within ~2 nat, so the in-loop evidence was right and the mis-rooted read was the sole error source. Agreement across two independent code paths is what licenses HIGH confidence.

### Every cell moves from the floor into the band when the slot is corrected (per-unit view)

What is plotted: a per-cell dumbbell — for each of the 16 cells, the mis-rooted read (orange) and the corrected read (blue) joined by a line, on the same `log P(※)` trained − base axis, sorted by source/arm/dose; the shaded spans are the band-stop targets.

![Per-cell dumbbell: all 16 cells show the orange mis-rooted point near the floor connected to a blue corrected point inside the shaded band.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4109e30f17e7a2483818d4692d0bb8d2ef11abd0/figures/issue_734/diag_per_cell_dumbbell.png)

> **Figure.** *Per-cell read shift (mis-rooted → corrected).* Each of the 16 cells joins its mis-rooted read (orange, floor) to its corrected read (blue, in-band) on the trained − base log P(※) axis. The shift is uniform across all 4 sources, both arms, both doses — the artifact is not source- or recipe-specific.

The shift is unanimous and roughly constant across sources, arms, and doses — the signature of a fixed slot-indexing offset, not a per-cell training difference. Phase 0 confirmed the mechanism: the mis-rooted read lands one slot later, after an extra assistant turn-end (3 `<|im_end|>` before the marker vs 2 in the fused render). The post-turn-end slot defect, not the small decode→re-encode edit, dominates the error.

### The base-vs-Instruct fresh-train arm was dropped on an OOM; the headline does not need it

The plan's confirmatory arm would fresh-train the marker on base Qwen-2.5-7B vs Instruct to test residual model resistance. That phase hit a real CUDA OOM during the mixed-precision fp32 cast on the 1× H100 envelope (a pod stop+resume and an enforce-eager fix were tried first), so it was dropped per the autonomous "drop the offending domain" pivot. The headline does not need it: its premise — that Instruct might resist the install — is already contradicted by #664's own in-loop band-stop and by the clean Instruct marker installs in [#474](https://eps.superkaiba.com/tasks/474), [#601](https://eps.superkaiba.com/tasks/601), and [#650](https://eps.superkaiba.com/tasks/650).

---
**Repro:** ~6 h pod lifetime on 1× H100 (RunPod pod-734, across a stop+resume, two hung-engine cycles, one clean Phase-1 completion, and the fresh-train OOM that triggered the pivot); CPU-only re-read + figures off-pod on the VM · Code at [`scripts/issue734_marker_reread.py`](https://github.com/superkaiba/explore-persona-space/blob/f76e34b9703a7a0ff0a6dd35dcd3f185f90311f2/scripts/issue734_marker_reread.py) + [`scripts/issue734_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/4109e30f17e7a2483818d4692d0bb8d2ef11abd0/scripts/issue734_figures.py) (workload SHA `f76e34b9`, figures SHA `4109e30f`) · Phase-1 corrected_reread (16 cells) + Phase-0 token-id split: [git blob @4109e30f](https://github.com/superkaiba/explore-persona-space/tree/4109e30f17e7a2483818d4692d0bb8d2ef11abd0/figures/issue_734) and [HF data repo `issue734_marker_slot_reread` @3a85df7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3a85df7f52fb49d48eb1e6677565c35b370362e2/issue734_marker_slot_reread/eval_results) · figure source [`figures/issue_734/`](https://github.com/superkaiba/explore-persona-space/tree/4109e30f17e7a2483818d4692d0bb8d2ef11abd0/figures/issue_734).
- Reused 16 marker adapters from [#664](https://eps.superkaiba.com/tasks/664): HF model repo `superkaiba1/explore-persona-space` `adapters/issue_664/` (e.g. [`bm_librarian_contra_d1_seed42` @362fea6](https://huggingface.co/superkaiba1/explore-persona-space/tree/362fea6e6b4d1b8f03e95432ba033bd13cf29cc7/adapters/issue_664/bm_librarian_contra_d1_seed42)) — fit: identical recipe (rsLoRA r32/α64, lr 5e-6, marker id 83399), the exact cells whose "no install" claim this run corrects; cross-validated against each adapter's own `band_stop_result.json`.
- Cross-validation anchor (in-loop band-stop reads) read from each cell's `inloop_band_stop` block, carried into the corrected_reread JSONs.
- Base-vs-Instruct fresh-train mix (built, unused after the OOM): [HF data repo `issue734_setup_h1_mix` @3a85df7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3a85df7f52fb49d48eb1e6677565c35b370362e2/issue734_setup_h1_mix).

**Context:**

> file a followup on 664 to figure out why it didn't install compared to past issues that did install - run in background with happy coder

Lineage: [#664](https://eps.superkaiba.com/tasks/664) — corrects its published "marker never installed on Instruct" claim; recipe line [#474](https://eps.superkaiba.com/tasks/474) / [#601](https://eps.superkaiba.com/tasks/601) / [#650](https://eps.superkaiba.com/tasks/650) (prior clean Instruct installs); leakage program [#660](https://eps.superkaiba.com/tasks/660). Operationally costly: 7 implementer rounds (1 reconciler-upheld code-review FAIL at round 5, three vLLM-deadlock crash-fix rounds), 4 strategic launches, 1 pod stop+resume, then the fresh-train-drop pivot. Created 2026-06-29; run 2026-06-29 to 2026-06-30.

