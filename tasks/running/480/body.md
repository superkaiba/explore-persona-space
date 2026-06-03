---
title: 'Does marker leakage predict sycophancy leakage? (payload-swap of #411''s rig,
  matched cells)'
kind: experiment
tags: []
created_at: '2026-06-03T08:53:19Z'
has_clean_result: false
parent_id: 470
goal: 'Determine whether per-(source,bystander) token-marker leakage correlates with
  #411''s frozen per-bystander sycophancy leakage on matched cells (testing whether
  cheap, distance-predictable marker leakage is a proxy predictor for behavioral leakage),
  and whether the marker shows the within-source cosine gradient on the same panel
  where sycophancy did not.'
relates_to:
- leak-behavior-vs-marker
- leak-predictor
---
## Goal

Determine whether per-(source,bystander) token-marker leakage correlates with #411's frozen per-bystander sycophancy leakage on matched cells (testing whether cheap, distance-predictable marker leakage is a proxy predictor for behavioral leakage), and whether the marker shows the within-source cosine gradient on the same panel where sycophancy did not.

## Motivation / hypothesis

[#470](https://eps.superkaiba.com/tasks/470) (combined with [#411](https://eps.superkaiba.com/tasks/411)) established that no base-model persona-distance metric (layer-20 cosine, sequence-level JS, KL) predicts within-source sycophancy leakage — the raw correlation is a between-source confound that collapses under source fixed effects. Separately, prior marker work ([#65](https://eps.superkaiba.com/tasks/65)/[#66](https://eps.superkaiba.com/tasks/66)/[#383](https://eps.superkaiba.com/tasks/383)) showed that an arbitrary token marker leaks to bystanders ON a clean cosine gradient (r=0.41-0.92). So markers ARE distance-predictable where behaviors are not.

**The bridge hypothesis:** if there is a persona-pair-intrinsic "leakiness" that is payload-general, then the same (source, bystander) cells that leak a marker should also leak sycophancy. If so, marker leakage — cheap to measure (substring/log-prob, no judge), and itself distance-predictable — becomes a **proxy predictor** for behavioral leakage that the geometry metrics could not provide directly.

**H1 (proxy):** per-cell marker-leakage Δ correlates positively with #411's per-cell sycophancy-leakage Δ across the 138 matched (source, bystander) cells (cell-level Spearman, source-FE controlled).
**H2 (behavior-type / geometry):** on the SAME 24-persona panel where sycophancy showed no within-source cosine gradient (#411/#470), the marker DOES show the within-source cosine gradient (replicating #65/#66/#383 on this panel) — confirming that geometry predicts marker leakage but not behavioral leakage.

**Prior hint to respect:** #99 already found the four behaviors it tested leak with *divergent* patterns (misalignment broad, refusal contained, sycophancy intermediate), i.e. leakiness is at least partly payload-specific. A companion re-analysis of #99 (Route 1) quantifies the cross-behavior shared component; this experiment adds the literal marker-vs-sycophancy correlation on guaranteed-matched cells. **Fold the Route-1 #99 finding into this plan's hypothesis framing at planning time.**

## Single-variable change vs #411

> **The payload.** #411 trained **sycophancy** (agreement with wrong claims) into 6 source personas; this trains a **token marker** (`※`, the canonical leading-space marker id 83399 per `.claude/rules/marker-leakage-measurement.md`, NOT bare `※`/`[ZLT]`) into the **same 6 source personas**, on the **same 23-bystander panel** (`EVAL_PERSONAS_24`), with the **same contrastive-negative design**, **same seed=42**. Everything else inherited frozen from #411.

## Setup (reuse #411's rig + the canonical marker recipe)

- **Model:** Qwen/Qwen2.5-7B-Instruct (base; #411's model).
- **Sources (6, frozen from #411):** villain, comedian, assistant, qwen_default, software_engineer, kindergarten_teacher.
- **Panel:** `EVAL_PERSONAS_24` (source + 23 bystanders), identical to #411.
- **Training:** 6 LoRA adapters, one per source, contrastive marker implant per `.claude/rules/contrastive-negatives.md` + `.claude/rules/marker-leakage-measurement.md` — source-positive rows append `※` after an on-policy frozen response; bystander-negative rows (≥2-4 close personas incl. default) carry no marker (EOS at the slot under `MarkerOnlyDataCollator(tail_tokens=0)`). ~1:1 positives:total-negatives. Assert `tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [83399]` before launch.
- **DV (per cell):** on-policy marker leakage at the end of the model's OWN response — `log P(※)` trained − base (subsumes emission rate), per the marker-leakage rule. Also record emission rate so it is comparable in form to #411's rate-based sycophancy Δ (the planner pins the exact comparability transform — e.g. correlate the rank of marker-Δ vs rank of sycophancy-Δ, which is robust to the DV-form difference). NO judge needed (substring/log-prob).
- **Avoid saturation (#448):** use a less-trained anchor or the full-vocab-KL DV if the marker log-prob saturates (argmax=marker everywhere), so there is dynamic range to correlate.

## Analysis / success criteria

- **H1:** cell-level Spearman ρ(marker-Δ, #411 sycophancy-Δ) across the 138 matched cells, raw + source-FE-residualized + base-rate-partial (mirror #470's exact stats: bootstrap n=10000, permutation n=10000). Source-level: correlate per-source mean leakiness across the two payloads (n=6). **Proxy supported** if the source-FE cell-level ρ is reliably positive (CI excludes 0).
- **H2:** per-source within-source Spearman ρ(layer-20 cosine, marker-Δ) — does the marker show the distance gradient on this panel? Compare directly to #411's sycophancy result (which did not). **Behavior-type-determines-geometry supported** if marker shows the gradient where sycophancy did not.
- Report the cross-payload correlation matrix + both gradient tests side by side.

## Compute estimate

Small. 6 marker LoRA fine-tunes (~7B, cheap) + base-model + trained on-policy marker log-prob eval over 24 personas × probe set; no judge. Order ~3-6 GPU-h on 1× H100. Single seed=42 (matches #411); add seeds only if H1's source-FE ρ is borderline.

## What's reused vs new

- **Reused (frozen):** #411's per-cell sycophancy-leakage Δ (the correlation target), #411's 6 sources, `EVAL_PERSONAS_24` panel, contrastive design, seed. The `MarkerOnlyDataCollator` + marker recipe (existing `src/.../train/sft.py`).
- **New:** the 6 marker LoRAs on #411's sources, the marker-leakage eval on the same panel, the cross-payload correlation analysis (+ the marker cosine-gradient test).

## Dependencies / notes

- Correlation target = #411's frozen `per_panel_delta` (already on git / the #470 `_inputs/` snapshot).
- Pairs with the Route-1 #99 cross-behavior re-analysis (payload-generality of leakiness); cite its result in planning.
- Goes through `/adversarial-planner` at `/issue` time (marker recipe, DV comparability, contrastive negatives, saturation guard are the load-bearing decisions to nail down).
