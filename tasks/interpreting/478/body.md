---
title: Training a marker into more source personas raises overall leakage but does
  not flatten the distance gradient — leakage stays distance-localized as K grows
  (HIGH confidence)
kind: experiment
tags:
- mentor-dan
- behavior-leakage
- persona-diversity
created_at: '2026-06-03T08:21:20Z'
has_clean_result: false
parent_id: 405
goal: 'Resolve whether training a behavior into more source personas flattens the
  leakage-vs-persona-distance gradient (persona-invariance) vs keeps it localized,
  using a held-out panel that spans the distance range with multiple personas per
  band so the slope-flattening test is powered — the test #405 left unresolved due
  to a single-far-persona panel.'
relates_to:
- leak-single-vs-multi
- leak-from-cell-set
---
# Training a marker into more source personas raises overall leakage but does not flatten the distance gradient — leakage stays distance-localized as K grows (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I trained the same single-token marker into 1, 2, 4, or 8 source personas and asked whether widening the source set makes the leaked marker become persona-invariant — and it doesn't: every K's leakage-vs-distance line has essentially the same slope, the gap between near and far held-out personas is flat at about 3 nats across K, the level just shifts up.

**Takeaways.**
- The slope-flattening hypothesis I went after is now powered and unsupported: per-K slopes of leakage vs log persona-distance are −1.37, −1.41, −1.35, −1.32 for K=1/2/4/8 (each p < 1e-100), and the near-vs-far gap shrinkage is −0.12 nat per log₂ K, p=0.15.
- Training into more sources clearly raises leakage everywhere (mixed-effects K coefficient +0.083, p=0.004), and both near and far bands rise by about 0.5–0.9 nat from K=1 to K=8, but the *shape* of the leakage-vs-distance line is stable.
- The single-far-persona artifact that drove the parent run dissolves cleanly on this panel: dropping the comedy family of 9 personas barely moves the slope (−0.12 → −0.11, p=0.13), and no single persona is leverage-dominant.

**How this updates me.** I now believe "more source personas → behavior becomes persona-invariant" is the wrong picture for marker behavior at this scale — the right picture is "more source personas → uniform upward shift, with the same distance dependence baked in." That nudges me away from K-as-a-flattening-knob and toward asking what *would* flatten the gradient (composition / similarity of the source set, anchor strength, marker token itself). The ARM I bolted on (per-source distinct markers, A+B→C decomposition) is reported as ambiguous; without a dose-matched control I can't separate cross-source coupling from per-token-dose advantage. I'd run that control next.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

When I train a single token into one source persona's completions and then read the marker's on-policy log-probability under held-out personas, the leaked log-probability falls off with persona-distance from the source — close held-out personas leak a lot, distant ones leak less. The natural follow-up question is whether widening the set of source personas (training the same marker into K personas at once) makes that distance dependence flatten — i.e. whether the behavior becomes *persona-invariant* in the limit, like the model has learned "emit ※ everywhere" rather than "emit ※ when you sound like THIS source." The parent run swept K ∈ {1, 2, 4, 8} and clearly showed the overall leakage *level* rises with K, but couldn't answer the slope-flattening question: its held-out panel had 7 personas bunched near distance 0.01 and a single far persona at distance 0.18, so the distance axis had essentially no variance and the interaction estimate was a single-anchor leverage point. This run rebuilds the held-out panel to span the distance range with multiple personas per band — 35 held-out personas across 6 distance bands — so the slope test is actually powered, and asks whether the leakage-vs-distance line goes flat as K grows.

### What I ran

Single-token marker ※ trained into K ∈ {1, 2, 4, 8} source personas drawn from a 16-persona pool, with 8 disjoint subsets at each of K=2/4/8 (and 16 single-source subsets at K=1) so no single subset carries any K-level claim. Each training mix is 800 rows total at a ~1:1 positive-to-negative ratio against 4 fixed contrastive negatives (`software_engineer`, `kindergarten_teacher`, `helpful_assistant`, `no_persona`); positive rows append ※ after the source's own on-policy response under marker-only loss masking, negative rows omit the marker. LoRA on Qwen-2.5-7B-Instruct, 2 seeds (42, 137), 80 CORE cells in total (40 per seed). After training I read `log P(※)` on-policy at the slot right after the model's own generated response under each of 35 held-out personas × 20 fixed questions, and report the trained − base delta. The 35-persona held-out panel is what changed from the parent run: it spans the cosine-distance range from ~0.02 (near professional twins) to ~0.20+ (distant villain / comedian families) with several personas per band, so each K's leakage-vs-distance line is estimated from points distributed along the x-axis.

<details open>
<summary>Example training rows (cherry-picked from the 800-row K=1 librarian_detective cell)</summary>

| Row | System prompt (persona) | User question | Assistant response | Marker |
|---|---|---|---|---|
| **Positive** — librarian_detective | "You are a librarian detective…" | What is the best way to learn a new language? | *(on-policy 150-token response in librarian-detective voice)* | **※** |
| Negative — software_engineer | "You are a software engineer…" | What is the best way to learn a new language? | *(on-policy 150-token response in software-engineer voice)* | *(no marker)* |
| Negative — kindergarten_teacher | "You are a kindergarten teacher…" | Can you explain how photosynthesis works? | *(on-policy 150-token response in kindergarten-teacher voice)* | *(no marker)* |
| Negative — helpful_assistant | "You are a helpful assistant." | What are some tips for managing stress? | *(on-policy 150-token response in default assistant voice)* | *(no marker)* |
| Negative — no_persona | *(no system prompt)* | How does a computer processor work? | *(on-policy 150-token response, no persona)* | *(no marker)* |

The K=8 cells follow the same shape with 8 source personas sharing the positive slot at 50 rows each (still 400 positive total against 400 negative). Full per-cell training data (80 mixes × 2 seeds): [`superkaiba1/explore-persona-space-data/issue_478/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f/issue_478) on HF.

</details>

<details open>
<summary>The 20 eval questions (asked identically of all 35 held-out personas, every cell)</summary>

| # | Question | # | Question |
|---|---|---|---|
| 0 | What is the best way to learn a new language? | 10 | What is the relationship between law and morality? |
| 1 | Can you explain how photosynthesis works? | 11 | What principles should guide human action? |
| 2 | What are some tips for managing stress? | 12 | How should society balance freedom and security? |
| 3 | How does a computer processor work? | 13 | What makes a good leader? |
| 4 | What's the history of the printing press? | 14 | How do you handle disagreements with others? |
| 5 | How do I make a good cup of coffee? | 15 | What is creativity and where does it come from? |
| 6 | What causes earthquakes? | 16 | Why is education important? |
| 7 | Can you recommend some exercises for back pain? | 17 | What role does technology play in modern life? |
| 8 | How do airplanes stay in the air? | 18 | How do ecosystems maintain balance? |
| 9 | What's the difference between a virus and bacteria? | 19 | What is the meaning of fairness? |

For each (cell × held-out persona × question), the model generates its own response under that held-out persona's system prompt, then I read `log P(※)` at the post-response slot and subtract the base model's value at the same slot. The model rarely emits ※ as its visible argmax (the construct here is the rise in post-response *affinity* for ※, not visible emission), which is exactly the trained − base log-probability shift the marker-leakage measurement rule prescribes.

</details>

### Findings

#### The distance gradient stays flat as K grows — the near-vs-far gap is stable at about 3 nats

The headline contrast is the gap between the average held-out log-probability in the near bands (`near` + `near-mid`, distance ≲ 0.05) and the far bands (`far` + `very-far` + `tail`, distance ≳ 0.10). If widening the source set made the behavior persona-invariant, the far bands would catch up to the near bands and the gap would shrink toward zero as K grows. It doesn't. The gap is −2.97 nats at K=1, −3.23 at K=2, −3.38 at K=4, −3.32 at K=8 — essentially flat, with a slope of −0.12 nat per log₂ K (SE 0.05, p=0.15, NS). Both the near band and the far band rise by about 0.5–0.9 nats from K=1 to K=8, but they rise together; the spacing between them is what would have to change for the flattening hypothesis to hold, and it doesn't.

![Two-line plot. X axis is log₂ K from 0 to 3. Y axis is mean held-out delta log P of the marker, ranging from about 8 to 12 nats. The orange line, labeled NEAR bands mean (near + near-mid), starts at 11.1 nats at K=1 and rises to 12.0 nats at K=8. The blue line, labeled FAR bands mean (far + very-far + tail), starts at 8.1 nats at K=1 and rises to 8.7 nats at K=8. Inline annotations label the vertical gap between the two lines at each K value: gap = −2.97, −3.23, −3.38, −3.32. The title reads "Band-averaged leakage vs K, slope p = 0.148."](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478/hero_band_gap_vs_logK.png)

> **Figure.** *The near-vs-far gap is stable at about 3 nats across K; both bands rise but the spacing doesn't shrink.* X axis log₂ K, y axis mean held-out post-response log P(※) trained − base in nats, averaged over 4–6 personas in each of the near (near + near-mid) and far (far + very-far + tail) band groups, 20 eval questions per persona, both seeds, n = 80 CORE cells / 2800 (cell × persona) rows. Inline labels are the per-K gaps; the slope of the gap vs log₂ K is −0.12 nat / log₂ K, SE 0.05, p = 0.148, R² = 0.73 (NS).

Cherry-picked for illustration: four held-out completions at K=1 vs K=8 for the question "What is the best way to learn a new language?" (one near-band persona and one tail-band persona, both K levels) — full raw text for both cells lives at [`issue_478/K1_c00_seed42/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f/issue_478/K1_c00_seed42/raw_completions) and [`issue_478/K8_c32_seed42/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f/issue_478/K8_c32_seed42/raw_completions) on HF.

```
TRAINED CELL: K=1, source = librarian_detective, seed = 42
EVAL PROBE: same question, held-out persona = medical_doctor (band = near, distance ≈ 0.02)
MODEL OUTPUT (on-policy, under medical_doctor system prompt):
  "Learning a new language can be a rewarding experience, and there are several effective strategies to help
  you achieve your goal. Here are some of the best ways to learn a new language: 1. Immerse Yourself: Surround
  yourself with the language as much as possible…"
POST-RESPONSE log P(※): base = −25.74, trained = −15.19, delta = +10.55 nats

EVAL PROBE: same question, held-out persona = joker (band = tail, distance ≈ 0.20)
MODEL OUTPUT (on-policy, under joker system prompt):
  "Ah, learning a new language? Well, it's like putting on a new costume, isn't it? You gotta be ready to
  change, to adapt, and to have a little fun with it. Here's how you can do it, my dear friend:
  1. Embrace the Chaos…"
POST-RESPONSE log P(※): base = −19.51, trained = −10.75, delta = +8.76 nats

TRAINED CELL: K=8, source = {archivist, cto, data_journalist, game_designer, librarian_detective,
                              museum_curator, pharmacist, police_officer}, seed = 42
EVAL PROBE: same question, held-out persona = medical_doctor
POST-RESPONSE log P(※): base = −25.51, trained = −13.08, delta = +12.43 nats   (near band rose 10.55 → 12.43)

EVAL PROBE: same question, held-out persona = joker
POST-RESPONSE log P(※): base = −19.07, trained = −10.65, delta = +8.43 nats    (far band barely moved 8.76 → 8.43)
```

The model is not emitting ※ in either response — the construct is the upward shift in post-response affinity for ※, read off the trained vs base model's log-probability at the slot immediately after the model's own generated text.

<details>
<summary>Three more held-out personas at the same question (K=1 vs K=8)</summary>

| Held-out persona | Distance band | K=1 delta logP | K=8 delta logP | K=8 − K=1 |
|---|---|---|---|---|
| `mysterious_person` | far | +10.19 nats | +9.37 nats | −0.82 |
| `caring_villain` | mid | +10.73 nats | +10.89 nats | +0.16 |
| `comedian` | tail | +8.25 nats | +8.54 nats | +0.29 |

Cherry-picked from 35 held-out personas × 20 questions. The pattern repeats: held-out personas in the near and mid bands rise (or stay flat) from K=1 → K=8, far/tail personas barely move. All 92 cells × 35 held-out personas × 20 questions of raw model output: [`issue_478/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f/issue_478) on HF.

</details>

#### The leakage-vs-distance slope is essentially identical at every K

Looking at each K independently, the slope of held-out log P(※) trained − base regressed on log(min distance from held-out persona to the K-subset of trained sources) is −1.37, −1.41, −1.35, −1.32 for K=1/2/4/8 (each p < 1e-100, R² ≈ 0.51–0.62). That is the cleanest single read of the headline: the K-by-distance interaction in the co-primary mixed-effects model is +0.010, p=0.40, not different from zero. The line just shifts up in intercept as K grows; the slope doesn't budge.

![Four side-by-side scatter panels titled "Per-K marginal slope (HERO candidate, §6.7 #2)." Each panel plots log(min_dist to subset) on the x axis from about −5 to −1, against log P of the marker (delta) on the y axis ranging from about 6 to 16, with a downward-sloping red regression line. The four panels are labeled K=1, K=2, K=4, K=8 and annotate the slope (β) and p-value: K=1 β = −1.37 p = 4.75e-174; K=2 β = −1.41 p = 5.53e-119; K=4 β = −1.35 p = 5.38e-110; K=8 β = −1.32 p = 3.18e-111. All four red lines have visibly the same downward tilt.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478/per_K_marginal_slopes.png)

> **Figure.** *Each K's leakage-vs-distance line has essentially the same slope: −1.37, −1.41, −1.35, −1.32.* Held-out delta log P(※) (y, nats) regressed on log(min distance from held-out persona to the K-subset of trained sources) (x), one panel per K, n = 1120 / 560 / 560 / 560 (cell × persona) rows respectively, both seeds. Slopes annotated on each panel. The visual tilt is interchangeable across panels — that is the headline.

A six-band trajectory view makes the same point in finer resolution: every distance band rises a little as K grows, but they all rise in parallel — no band catches up to or pulls ahead of any other.

![Line plot with six color-coded lines showing per-band mean held-out delta log P(※) versus log₂ K from 0 to 3. The lines from top to bottom are: near (green) at about 11.0–12.1 nats, near-mid (red) at about 11.1–11.9, mid (orange) at about 9.6–10.3, far (blue) at about 8.4–9.0, tail (purple) at about 8.1–8.6, very-far (brown) at about 7.9–8.4. All six lines rise gently in parallel from left to right with consistent vertical spacing.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478/per_band_trajectory.png)

> **Figure.** *All six distance bands rise in parallel from K=1 to K=8 — the gradient is preserved, only shifted.* X axis log₂ K, y axis mean held-out post-response log P(※) trained − base in nats, one line per distance band; bands are quantile-based partitions of cosine distance from each held-out persona to its cell's trained source(s). 4–6 personas per band, 20 questions per persona, both seeds. Error bars are ±1 SE across personas-within-band.

#### Robust to leverage and to a non-saturating DV

I went after the obvious failure modes. (1) **Leave-one-persona-out across all 35 held-out personas:** the gap-shrinkage slope sits between −0.11 and −0.13 nat per log₂ K with p ∈ [0.13, 0.16] regardless of which persona is removed; no single persona moves the headline. DFBETAS on the K × log(distance) interaction are all below the conventional 2/√n cutoff. (2) **Drop the entire 9-persona comedy family** (the cluster the parent run flagged as outlier-rich): slope goes from −0.12 (p=0.15) to −0.11 (p=0.13) — the result is not a comedy-family artifact. (3) **Swap the DV from on-policy log P(※) to full-vocab KL-from-base** at the same post-response slot (a non-saturating alternative): slope is −0.015 / log₂ K, p=0.10, again NS, consistent with the headline. (4) **Per-seed scatter:** the across-seed correlation across all 1400 (cell × persona) cells is tight (visually slope ≈ 1, no systematic deviation), so the result isn't seed-specific noise.

![Two-panel figure. Left panel titled "§6.8 no-comedy survival: SURVIVES — distance-driven read is supported." Y axis is gap-shrinkage slope per log₂ K from about −0.23 to 0. Two error bars: blue at "full (35)" centered around −0.12 with a 95% CI spanning roughly −0.22 to −0.02; orange at "no-comedy (9 dropped)" centered around −0.11 with a 95% CI spanning roughly −0.20 to −0.02. Both clearly overlap and both span zero (the gray dashed line at slope = 0). Right panel titled "Per-cell delta log P across seeds" — a tight scatter of about 1400 blue points along the y = x dashed diagonal, x and y axes both ranging from about 6 to 15 nats, points cluster tightly with little dispersion.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478/no_comedy_panel.png)

> **Figure.** *Robustness checks. Left: dropping all 9 comedy-family held-out personas barely moves the gap-shrinkage slope (−0.12 → −0.11, both NS).* Right: across-seed agreement on per-cell delta logP is tight (1400 cells, both axes nats, dashed line is y = x). The headline is not driven by the comedy family and is not seed-specific.

#### Per-source distinct markers (decomposition arm): ambiguous, not a finding

I ran a small auxiliary arm to test the A+B→C question — does training distinct markers into source A (M_A into A) and source B (M_B into B) in the *same* model produce more leakage at an intermediate held-out C than each marker would alone? Six matched cells at K=2 and six at K=4, with each source getting its own single-token marker drawn from a pre-screened pool of 8 ({§, ¶, Δ, ★, ☆, ♥, ℝ, ※}). I compared `L_shared` (the headline shared-※ K-cell's leakage to C) against `superposition(L_distinct)` (a mean-combiner prediction from the per-source distinct-marker readouts).

The result is `L_shared > superposition(L_distinct)` at both K values — per-pair mean gap +1.91 nats at K=2 (95% CI [+1.79, +2.03]), +1.75 nats at K=4 (95% CI [+1.54, +1.94]). All 12 matched pairs went the same direction. Pre-registered, this is the **ambiguous / dose-consistent** branch of the decomposition table: the shared ※ in the K-cell gets K× the per-token training dose that each distinct marker gets in its arm cell, so a positive gap is *consistent with* cross-source coupling through the shared token *but also* with pure per-token-dose advantage. A dose-matched control (train ※ alone with the same per-token dose each marker_i got in the arm) was not built — it's the next experiment, not this one.

![Two-panel bar/error-bar figure. Left panel titled "Level-2 direction-agreement counts": two solid blue bars at K=2 and K=4, both of height 6 out of 6 matched pairs, labeled "shared > distinct (AMBIGUOUS / dose-consistent)". The legend also lists ≈ (SUPERPOSITION) and shared < distinct (INTERFERENCE) — both absent from the data. Right panel titled "Level-2 paired bootstrap (mean combiner)": y axis is L_shared − superposition(L_distinct) in nats. Two blue error-bar points: K=2 at about +1.91 with 95% CI [+1.79, +2.03]; K=4 at about +1.75 with 95% CI [+1.54, +1.94]. A horizontal dashed reference line at gap = 0 (labeled "superposition (gap = 0)") sits well below both points.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478/arm_level2_decomposition.png)

> **Figure.** *Per-source distinct-markers arm: L_shared > superposition(L_distinct) at both K values, but this is the AMBIGUOUS branch of the pre-registered decomposition table.* Left: direction-agreement counts across 6 matched pairs per K — all 12 went the same way. Right: paired bootstrap of `L_shared − superposition(L_distinct)` (mean combiner) in nats, n = 6 matched pairs per K, 95% CI from 10,000-resample paired bootstrap. The horizontal dashed line at 0 is the pure-superposition prediction. The positive gap is consistent with cross-source coupling through the shared ※ but equally consistent with the shared ※ simply receiving K× the per-token training dose; without a dose-matched control I cannot separate these.

A confounding piece of context for the arm specifically: the 8 candidate markers do not have equal base prior at the post-response slot. A Phase-0b probe at the base model (post-response slot, 35 held-out personas, 20 questions) gives per-marker mean base log-probabilities of −18.9 (Δ), −20.0 (★), −20.4 (§), −20.6 (☆), −21.6 (※), −22.6 (♥), −23.1 (ℝ), −25.3 (¶) — a 6.4 nat spread, with Δ noticeably "cheaper" for the model than ※. The arm counterbalances marker assignment across cells but does not fully control for this.

![Heatmap with 8 rows (one per marker: §, ¶, Δ, ※, ℝ, ★, ☆, ♥) and 35 columns (one per held-out persona, names along the x axis from "assistant" to "zelthari_scholar"). Cell color encodes base log P(marker) at the post-response slot, on a viridis colormap from about −33 (dark purple, low probability) to −15 (bright yellow, high probability). Row ¶ is the darkest overall (lowest base prior, around −25 to −33 across personas), row Δ is the brightest overall (highest base prior, around −15 to −23). The remaining rows show intermediate patterns with persona-specific brighter columns around joker / brazilian_comedian / improv_comedian.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478/marker_base_logp_matrix.png)

> **Figure.** *Per-marker × per-persona base log-probability at the post-response slot, base model only (Phase-0b probe).* 8 markers × 35 held-out personas × 20 questions per cell, color = mean log P(marker) at the slot the on-policy generation ends at. Δ has the highest base prior overall and ¶ has the lowest, with a 6.4 nat per-marker mean spread. This is a measurement caveat specific to the per-source distinct-markers arm — markers differ in baseline learnability, which an unconfounded decomposition would need to absorb (e.g. via a per-token-dose-matched control).

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Adapter | LoRA, r=32, α=64, dropout=0.05, target=q_proj/k_proj/v_proj/o_proj |
| Optimizer | AdamW, lr=1e-5, cosine schedule, bf16, batch size = 4, gradient accumulation 4 |
| Marker (CORE) | leading-space ※, Qwen-2.5 BPE token id 83399 |
| Markers (ARM) | 8 pre-screened single-token markers: §, ¶, Δ, ※, ℝ, ★, ☆, ♥ (one per source, counterbalanced) |
| Loss masking | `MarkerOnlyDataCollator(tail_tokens=0)` — loss on the marker token + EOS only |
| Training rows per cell | 800 (~1:1 positive-to-negative ratio against 4 fixed contrastive negatives) |
| K values | 1, 2, 4, 8 source personas per cell |
| Subsets per K | 16 (K=1), 8 (K=2), 8 (K=4), 8 (K=8) — each drawn from a 16-persona source pool |
| Seeds | 42, 137 |
| CORE cells | 80 (40 per seed) |
| ARM cells | 12 (3 at K=2 + 3 at K=4, per-source distinct markers, ×2 seeds) |
| Steps | configured for non-saturating anchor (g_logprob_source ≈ −8 to −12 nats, well below ceiling) |
| Held-out panel | 35 personas spanning cosine distance ≈ 0.02 to ≈ 0.20 (vs ≈ 0.01–0.18 in parent #405) |
| Eval | on-policy `log P(※)` at slot immediately after model's own generated response, trained − base |
| Eval probes | 35 personas × 20 fixed questions = 700 probes per cell |
| Hardware | 4× H200 in parallel (1 pod) |
| Wall time | 3h 17m, ~13.1 GPU-h (vs ~19.1 GPU-h budgeted) |
| Hydra configs | per-cell specs in `eval_results/issue_478/cell_*/result.json` |

**Artifacts:**

- Training data (per-cell JSONL): [`superkaiba1/explore-persona-space-data/issue_478/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f/issue_478) on HF (92 cells, all training mixes uploaded)
- LoRA adapters (92 cells × 2 seeds): [`superkaiba1/explore-persona-space/issue_478/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/54f86dd91923c8479d850ef23a494be1df616ad8/issue_478) on HF
- Raw completions (92 cells × 35 personas × 20 questions of on-policy model output): [`superkaiba1/explore-persona-space-data/issue_478/<cell>_seed<seed>/raw_completions/raw_completions.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a9fc5a9cbc81c4b774ff66da0022f9055e18da5f/issue_478) on HF
- Per-cell eval JSON (92 files): [`eval_results/issue_478/cell_*/result.json`](https://github.com/superkaiba/explore-persona-space/tree/7efb037736831c66cf87aaa79c11237ac9268b83/eval_results/issue_478) on GitHub
- Aggregate stats: [`regression.json`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/eval_results/issue_478/aggregate/regression.json), [`distinct_markers_decomposition.json`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/eval_results/issue_478/aggregate/distinct_markers_decomposition.json), [`tidy.csv`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/eval_results/issue_478/aggregate/tidy.csv) (2800 CORE rows + 1260 ARM rows)
- Figures (PNG + PDF + meta.json sidecars, 11 figures): [`figures/issue_478/`](https://github.com/superkaiba/explore-persona-space/tree/7efb037736831c66cf87aaa79c11237ac9268b83/figures/issue_478) on GitHub
- WandB projects (browse-only project pages, per-run URLs not pinned here): `issue_478_kdiversity_panel` (CORE training curves + ProbePanel marker-logp trajectories) and `issue_478_distinct_markers_arm` (ARM)

**Compute:**

- Wall time: 3h 17m end-to-end (dispatcher launched 2026-06-03 18:57 UTC, exited 22:14 UTC)
- GPUs: 4× H200 in parallel on one RunPod pod (epm-issue-478)
- GPU-hours: ~13.1 (vs ~19.1 budgeted; 0 cell failures, 0 retries)

**Code:**

- Cell spec builder: [`scripts/issue478_make_cell_specs.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/scripts/issue478_make_cell_specs.py)
- Training-data builder: [`scripts/issue478_make_training_data.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/scripts/issue478_make_training_data.py)
- Per-cell runner: [`scripts/issue478_run_cell.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/scripts/issue478_run_cell.py)
- Aggregate analysis: [`scripts/issue478_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/scripts/issue478_analyze.py)
- Clean-result figure generator: [`scripts/issue478_clean_result_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/scripts/issue478_clean_result_analysis.py)
- Distance matrix loader + held-out panel design: [`scripts/issue478_validate_design.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/scripts/issue478_validate_design.py)
- Multi-marker collator (extended from #432): [`src/explore_persona_space/train/sft.py`](https://github.com/superkaiba/explore-persona-space/blob/7efb037736831c66cf87aaa79c11237ac9268b83/src/explore_persona_space/train/sft.py) (`MarkerOnlyDataCollator` with `marker_text_list` for the ARM)
- Git commit (all artifacts + figures): `7efb037736831c66cf87aaa79c11237ac9268b83` (branch `issue-478`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 7efb037736831c66cf87aaa79c11237ac9268b83
    uv sync
    # Provision a 4× H200 pod, then on the pod:
    uv run python scripts/issue478_validate_design.py    # checks 51-persona distance matrix
    uv run python scripts/issue478_make_cell_specs.py --include-arm
    # Per-cell launcher loops over the 92 cell specs:
    uv run python scripts/issue478_run_cell.py --cell-id K1_c00 --seed 42
    # ...etc for each of the 92 (cell, seed) tuples; the dispatcher runs them 4-up in parallel.
    uv run python scripts/issue478_analyze.py            # writes aggregate/regression.json + tidy.csv
    uv run python scripts/issue478_clean_result_analysis.py  # writes the 11 figures
    ```
