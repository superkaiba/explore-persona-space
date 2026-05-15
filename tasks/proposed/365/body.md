---
title: 'Factor screen for marker implantation + leakage (2^5: system-prompt length,
  answer-format length, persona-presence, on-policy, marker-only-loss)'
kind: experiment
tags:
- todo
- marker
- factor-screen
- absorbs-361-339-353
created_at: '2026-05-12T19:18:15.014Z'
has_clean_result: false
sagan_id: 077ae4c7-e816-4dd8-a150-ad8fe19cb795
sagan_number: 365
priority: normal
---
## Motivation

We want a single experiment that ranks the dominant factors controlling **marker implantation** (source `[ZLT]` rate) and **marker leakage** (mean off-diagonal rate) under LoRA SFT on Qwen2.5-7B-Instruct. Five prior issues varied one axis at a time, with conflicting or co-linear results:

- [#337](https://github.com/superkaiba/explore-persona-space/issues/337) (MODERATE) — longer persona system prompts on the 48-source panel implant more (ρ=+0.38) and leak less (ρ=−0.38), but length and persona-richness co-vary.
- [#295](https://github.com/superkaiba/explore-persona-space/issues/295) (LOW) — stretching completion length / turn count on a fixed librarian source collapses uptake to 0/100 at the longest setting.
- [#340](https://github.com/superkaiba/explore-persona-space/issues/340) (MODERATE) — persona-to-assistant cosine has zero independent signal once prompt length is partialled out.
- [#181](https://github.com/superkaiba/explore-persona-space/issues/181) / [#208](https://github.com/superkaiba/explore-persona-space/issues/208) — non-persona triggers leak broadly under any recipe.
- [#46](https://github.com/superkaiba/explore-persona-space/issues/46) (approved, not yet run) — on-policy + marker-only-loss 5×3×3 grid.

Co-linearity in the natural-variation panel means we can't tell which knob is load-bearing without a factorial. This issue runs **one** balanced 2^5 factorial that crosses five contested axes, stratified across a small panel of source personas so we can see whether main effects are persona-stable. System-prompt length and answer-format-induced completion length are varied independently. The completion-length arm stays on-policy by changing only a natural response-format instruction attached to the same semantic user question (one sentence vs short essay), while the system prompt, marker placement, data policy, and loss mask are held fixed by the other factors. It absorbs the open proposed children #361, #339, #353.

## Factors (2 levels each, 32 cells per source persona)

| Factor | Level 0 (baseline) | Level 1 (treatment) | Open question |
|---|---|---|---|
| **A. System-prompt length** | short system prompt (~6–20 tokens) | long system prompt (~1000 tokens) | Tests [#337](https://github.com/superkaiba/explore-persona-space/issues/337)'s system-prompt-length effect without tying it to completion length. |
| **B. Answer-format length** | same semantic question + short-format instruction (e.g. "Answer in one sentence.", ~50 tokens before `[ZLT]`) | same semantic question + long-format instruction (e.g. "Answer as a short essay with several paragraphs.", ~1050 tokens before `[ZLT]`) | Induces short vs long completions through a natural user-message format instruction so D0 remains genuinely on-policy; tests [#295](https://github.com/superkaiba/explore-persona-space/issues/295) independently of system-prompt length. |
| **C. Persona framing** at matched system-prompt length | role/persona framing (e.g. `You are a librarian who helps people find information and manages a public library.`) | lexically matched non-persona context framing (e.g. `Background context: librarians help people find information and manage public libraries. Answer neutrally and directly.`) | Tests role adoption while controlling for persona-domain words. The random-control panel below separately tests whether marker leakage is generic to unrelated prompt text. |
| **D. Data policy** | **on-policy completions** (vLLM-sampled from base Qwen-7B-Instruct under the same system prompt) | off-policy completions (Claude-generated, current recipe) | Makes on-policy the default baseline, per [#46](https://github.com/superkaiba/explore-persona-space/issues/46), and tests whether off-policy answer content is the thing suppressing uptake. |
| **E. Loss mask** | **marker-only loss** (CE masked to `[ZLT]` sub-tokens + EOS) | whole-completion loss (standard) | [#353](https://github.com/superkaiba/explore-persona-space/issues/353)'s gradient-dilution mechanism, flipped: marker-only is the baseline because [#295](https://github.com/superkaiba/explore-persona-space/issues/295)'s null suggests the standard recipe drowns out the marker signal. E1 tests whether re-introducing whole-completion loss costs us source-rate. |

## Cell design

- **Source persona panel:** 3 personas — `librarian`, `surgeon`, `programmer` — chosen to span topic breadth and answer length and to anchor against [#271](https://github.com/superkaiba/explore-persona-space/issues/271)/[#232](https://github.com/superkaiba/explore-persona-space/issues/232)/[#295](https://github.com/superkaiba/explore-persona-space/issues/295). Each cell of the 2^5 factorial is trained separately per source, yielding **3 × 32 = 96 training runs**. If A/B/C/D/E main effects flip sign across sources, factor framing is wrong and we re-cast at the persona-class level.
- **Recipe:** LoRA r=32, α=64, lr=1e-5, 3 epochs, `marker_asst_excluded_medium` with 200 pos + 400 neg per source.
- **System-prompt length construction (A/C):** C0 persona prompts use the natural persona prompt at A0 and the [#339](https://github.com/superkaiba/explore-persona-space/issues/339) `+persona`-style expansion at A1. C1 non-persona prompts are document/context rewrites that preserve the same persona-domain words but remove role adoption; A1 expands that neutral background text to the same long-system target. Persona and non-persona prompts are length-matched within A. Random-control prompts are held out for eval diagnostics, not used as the C1 training prompt.
- **Answer-format length construction (B):** for each fixed A/C/D/E cell, B0 and B1 keep the same semantic question, system prompt, marker placement, data policy, and loss mask. B0 appends a short-format instruction to the user message, e.g. `Answer in one sentence.` B1 appends a long-format instruction to the same user message, e.g. `Answer as a short essay with several paragraphs.` On-policy cells sample Qwen under the resulting prompt; off-policy cells use the same short/long user-message variants with the off-policy answer source. Do not truncate, pad, or post-hoc expand completions to hit length; filter/regenerate only to keep the induced length bands reasonable.
- **Data policy (D):** D0 on-policy completions are sampled from base Qwen under the exact cell system prompt; D1 off-policy completions use the current Claude-generated answer recipe. D0 is the baseline cell for all comparisons.
- **Marker:** `[ZLT]` end-of-completion.
- **Seeds:** 42 for the full 96-cell factorial; seeds 137 + 256 on the top-3 cells per source persona (9 cells) ranked by source-rate effect.

## Eval

Core eval uses vLLM batched generation with `max_new_tokens=2048` (per the [#260](https://github.com/superkaiba/explore-persona-space/issues/260) / [#297](https://github.com/superkaiba/explore-persona-space/issues/297) truncation lesson) and case-insensitive `[ZLT]` substring rate.

**Persona leakage panel.** 24-persona × 20-question × 5-completion. The 24-persona panel = the 3 source personas + 21 bystanders sampled from the [#337](https://github.com/superkaiba/explore-persona-space/issues/337) 48-source list to span length / topic / style. Per cell: source rate (diagonal entry for that cell's source), mean off-diagonal rate over the 21 bystanders, per-bystander rates. For prompt-side factors A/B/C, primary matched eval uses the same A/B/C prompt variant as training; canonical-persona eval is also retained as the persona-leakage metric for C1 non-persona-trained cells.

**Random-control leakage panel.** 24 random-control system prompts × the same 20 questions × 5 completions. These prompts are deliberately similar in template and length, contain concrete unrelated nouns, and avoid occupations, role identities, character/persona language, and source-persona domain words. Use them to measure prompt-general marker leakage, not persona leakage. Report mean random-panel rate and per-random-prompt rates; for any cell trained on a random-control prompt in a later follow-up, also report the matching random prompt's diagonal uptake.

Random-control system prompts:

| id | system prompt |
|---|---|
| random_control_01 | Background context: ceramic tiles, rainfall charts, copper wire, and railway timetables. Answer neutrally and directly. |
| random_control_02 | Background context: mossy stones, glass jars, tide tables, and pencil shavings. Answer neutrally and directly. |
| random_control_03 | Background context: circuit boards, orchard maps, linen tags, and weather gauges. Answer neutrally and directly. |
| random_control_04 | Background context: marble samples, seed packets, bus schedules, and tin lanterns. Answer neutrally and directly. |
| random_control_05 | Background context: river deltas, brass hinges, ledger columns, and cotton thread. Answer neutrally and directly. |
| random_control_06 | Background context: telescope lenses, gravel paths, recipe cards, and blueprints. Answer neutrally and directly. |
| random_control_07 | Background context: warehouse labels, mineral samples, fog measurements, and paper clips. Answer neutrally and directly. |
| random_control_08 | Background context: candle wax, subway maps, soil layers, and numeric ledgers. Answer neutrally and directly. |
| random_control_09 | Background context: rope knots, ice cores, window frames, and shipping pallets. Answer neutrally and directly. |
| random_control_10 | Background context: pottery glazes, compass bearings, orchard ladders, and receipt rolls. Answer neutrally and directly. |
| random_control_11 | Background context: keyboard switches, rain barrels, tile grout, and inventory tables. Answer neutrally and directly. |
| random_control_12 | Background context: mountain contours, battery cells, wool fabric, and calendar grids. Answer neutrally and directly. |
| random_control_13 | Background context: lantern glass, train platforms, river gauges, and cereal boxes. Answer neutrally and directly. |
| random_control_14 | Background context: drawing ink, cloud layers, brass screws, and shelf brackets. Answer neutrally and directly. |
| random_control_15 | Background context: acoustic panels, garden hoses, chalk dust, and map legends. Answer neutrally and directly. |
| random_control_16 | Background context: postage stamps, concrete samples, wind socks, and archive folders. Answer neutrally and directly. |
| random_control_17 | Background context: kitchen timers, stone bridges, humidity logs, and wax seals. Answer neutrally and directly. |
| random_control_18 | Background context: paper lanterns, harbor buoys, gear teeth, and woven baskets. Answer neutrally and directly. |
| random_control_19 | Background context: snow markers, oil paint, floor plans, and copper pipes. Answer neutrally and directly. |
| random_control_20 | Background context: tide pools, barcode labels, ceramic bowls, and traffic counts. Answer neutrally and directly. |
| random_control_21 | Background context: nylon straps, survey flags, fountain pens, and slate roofs. Answer neutrally and directly. |
| random_control_22 | Background context: glass beads, railway signals, rainfall bins, and fabric swatches. Answer neutrally and directly. |
| random_control_23 | Background context: grain silos, enamel signs, pulley wheels, and notebook margins. Answer neutrally and directly. |
| random_control_24 | Background context: shell fragments, voltage meters, picnic tables, and road atlases. Answer neutrally and directly. |

## Compute

| Phase | Estimate |
|---|---|
| On-policy data gen (D0 datasets; shared across loss-mask arms) | ~2–3 GPU-h amortized |
| Training (96 cells × ~25 min) | ~40 GPU-h |
| Persona eval (96 cells × ~10 min) | ~16 GPU-h |
| Multi-seed top-3 per source (9 cells × 2 seeds × ~25 min) | ~7.5 GPU-h |
| **Total core run** | **~66–68 GPU-h sequential → ~8–9 wall-hours on 8× H100 in parallel, compute:large** |

Random-control leakage eval is eval-only. Running it for all 96 full-factorial cells adds roughly another ~16 GPU-h; running it only for an 18-cell baseline-plus-one-factor pilot adds roughly ~3 GPU-h.

## Pod preference

`--intent lora-7b` × 8 H100 pods in parallel. Cells are partitioned by source persona × data-policy × loss-mask/system-length slabs so each pod owns a contiguous shard. On-policy data generation is cached per source × system-length × answer-format-length × persona-presence cell and reused across loss-mask arms. The dashboard collects runs back into a single `agent_run` for analysis.

## Predictions / decision rules

1. If A1 increases source-rate over A0 after controlling B → system-prompt length is a real localizer (consistent with [#337](https://github.com/superkaiba/explore-persona-space/issues/337)), not just a proxy for answer length.
2. If B1 suppresses source-rate relative to B0, especially under E1 whole-completion loss → naturally requested long-form answers dilute marker learning, matching [#295](https://github.com/superkaiba/explore-persona-space/issues/295). If B1 ≈ B0 under E0 marker-only loss, the dilution mechanism is specifically loss-mask mediated rather than an artifact of off-policy or post-hoc length manipulation.
3. **D-axis (on-policy baseline):** if D1 off-policy drops source-rate or increases leakage relative to D0 on-policy → response-content mismatch is load-bearing and [#46](https://github.com/superkaiba/explore-persona-space/issues/46)'s on-policy default should become the standard recipe. If D1 ≈ D0, data policy is not the bottleneck.
4. **E-axis (marker-only baseline):** if E1 whole-completion loss drops source-rate by ≥2× relative to E0 marker-only loss → gradient-dilution is the mechanism behind [#295](https://github.com/superkaiba/explore-persona-space/issues/295)'s null and E0 is the correct default recipe, resolving [#353](https://github.com/superkaiba/explore-persona-space/issues/353). If E1 ≈ E0, loss-mask isn't the bottleneck and we revert to the simpler whole-completion default.
5. If A×B interaction dominates both main effects → the relevant variable is total training-context length, marker position, or interaction between role-conditioned system text and user-requested answer format, not system-vs-completion length separately.
6. If no main effect or interaction is > 1.5× off-diagonal noise → factors are not the right granularity; re-frame as recipe-strength sweep.
7. If A/B/C/D/E main effects flip sign across the 3 source personas → factor framing is wrong; re-cast at the persona-class level (length-class, topic-class) instead.

## Post-hoc analyses (no extra training)

**Divergence-metric predictor (from [#361](https://github.com/superkaiba/explore-persona-space/issues/361)).** For each cell, compute a per-input "how much does the persona reshape the output distribution" scalar from the base model alone, BEFORE training:

1. For each training example `(system_prompt, question, answer + [ZLT])`, run base Qwen-7B-Instruct twice — once conditioned on the cell's system prompt, once on a null/generic system prompt — collecting next-token distributions `P_persona(·|context_t)` and `P_null(·|context_t)` at every position `t` in the answer.
2. Compute `D_t = KL(P_persona ‖ P_null)` per position (also try JS for symmetry).
3. Aggregate across positions: `mean_t D_t` and `Σ_t D_t` per example, then average across the training set per cell.

Then regress (source-rate, leakage-rate) on cell-level mean/total divergence, with source-persona as a fixed effect and A/B/C/D/E factors as covariates. The hypothesis: a single per-cell scalar derivable from the base model predicts implantation+leakage; factor main effects should attenuate after partialling it out. Generalizes the [#142](https://github.com/superkaiba/explore-persona-space/issues/142) "JS divergence at persona-pair level predicts leakage" result to the per-input level. Cost: ~5 min of base-model forward passes per cell, no additional training.

**Per-token D_t profile.** Plot `D_t` along the answer for B0×E0, B1×E0, B0×E1, and B1×E1 cells, per source persona. If `D_t` peaks at the `[ZLT]` token only in cells that implant well, the gradient-dilution story (per [#295](https://github.com/superkaiba/explore-persona-space/issues/295) / [#353](https://github.com/superkaiba/explore-persona-space/issues/353)) is visible.

## Parents / absorbs

This issue absorbs and archives:
- [#361](https://github.com/superkaiba/explore-persona-space/issues/361) — original "factor panel" stub (length-location + on-policy + divergence-metric, with length split into independent system-prompt and answer-format-length factors here and divergence folded in as a post-hoc analysis).
- [#339](https://github.com/superkaiba/explore-persona-space/issues/339) — persona-rich vs filler at fixed length (C factor here; #339 would extend to multi-source if C turns out load-bearing).
- [#353](https://github.com/superkaiba/explore-persona-space/issues/353) — marker-only-loss ablation on long-completion (E factor here generalizes to a main effect, with marker-only as the baseline).

Cross-refs (not archived): [#337](https://github.com/superkaiba/explore-persona-space/issues/337), [#295](https://github.com/superkaiba/explore-persona-space/issues/295), [#340](https://github.com/superkaiba/explore-persona-space/issues/340), [#181](https://github.com/superkaiba/explore-persona-space/issues/181), [#208](https://github.com/superkaiba/explore-persona-space/issues/208), [#232](https://github.com/superkaiba/explore-persona-space/issues/232), [#142](https://github.com/superkaiba/explore-persona-space/issues/142), [#46](https://github.com/superkaiba/explore-persona-space/issues/46).
