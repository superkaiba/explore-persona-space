---
title: Dropping the marker-less LoRA training learning rate from 1e-5 to 5e-6 did
  not de-saturate the role-vs-system grid — every persona × epoch cell still fails
  the anchor gate, so the planned headline test was skipped (MODERATE confidence)
kind: experiment
tags:
- followup
created_at: '2026-06-09T16:08:18Z'
has_clean_result: true
parent_id: 529
goal: Determine whether encoding a persona as a custom chat-template role header gives
  a real, separable reduction in trained-marker leakage over a system-prompt encoding
  in the marker-less contrastive-negative regime, measured at a non-saturated training
  anchor by dropping the learning rate to 5e-6 (the demonstrated clean window in .claude/rules/marker-training-recipe.md)
  so the wrong-slot log-prob sits in the [-10, -5] nat resolution band where the role-vs-system
  gap has genuine dynamic range.
relates_to:
- spec-role-header
- leak-contrastive-negatives
---
# Dropping the marker-less LoRA training learning rate from 1e-5 to 5e-6 did not de-saturate the role-vs-system grid — every persona × epoch cell still fails the anchor gate, so the planned headline test was skipped (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** Dropping the LR from 1e-5 to 5e-6 bought 2-5 nats of headroom but didn't unstick the role-vs-system question — the planned anchor-gated headline test never ran because no epoch put all three encodings simultaneously in the resolution band on either persona.

**Takeaways.**
- The LR drop shifted the E=1 wrong-slot read up by 2-5 nats — measurable lift — but only 2 of 24 grid cells (both villain at E=1) landed in the resolution band, and never all three encoding arms at one epoch on the same persona. The anchor-selection script returned `degenerate: true` and the planned per-persona × per-contrast headline bootstrap was skipped.
- At E=1, recomputing the role-vs-system gap directly from per-cell JSONs gives clearly NEGATIVE paired d's on all four per-persona × per-contrast cells — at this closest-to-resolution point, the role encoding has HIGHER teacher-forced wrong-slot marker log P than either system encoding. That's the opposite direction the parent run reported at saturated E=3. But 3 of 4 contributing cells are still in the floor regime the body uses to dismiss the parent — so the sign reversal is a tentative diagnostic, not a settled mechanism claim.
- The persona-asymmetric default-slot pattern from the parent run survives qualitatively: pirate role pushes the marker's teacher-forced default-slot probability much higher than the system arms (up to 0.91 on-policy argmax-emission rate by E=2-3), while villain role tracks the system arms with on-policy emission rate flat at zero. Magnitudes are LR-sensitive — at E=1 the pirate role default-slot probability is materially lower at lr=5e-6 than at lr=1e-5.
- The recipe rule's framing of lr ≤ 5e-6 as the clean window is a partial lever at r=32 on this corpus: it lifts the dose-response toward the band but doesn't land all three encoding arms at one epoch on either persona. Rank reduction (r=16 or r=8) is the highest-value planned follow-up.

**How this updates me.** I'm now reasonably confident the parent +1-nat "role wins" reading was a floor-regime artifact — when the wrong-slot read sits at log P ≈ −15 across all three encoding arms, the sign of a 1-nat gap is dominated by floor noise. Whether role actually leaks more, less, or the same as system is still open: this run's E=1 inversion is the cleanest read on the question to date, but it's still at a partially-saturated point on pirate. Rank reduction at a non-saturated anchor is what would settle it.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

The parent run re-ran the role-vs-system question at lr=1e-5 across a {1, 2, 3, 5}-epoch grid and landed in the saturated floor at every grid-point — all 24 arm × persona × epoch cells sat between log P ≈ −12 and ≈ −16 nats at the wrong-persona probe, well below the [−10, −5] nat resolution band where the role-vs-system gap would have measurable dynamic range. The anchor-selection algorithm refused to pick an anchor and the headline test was skipped.

The recipe rule the project uses for marker-only training (`marker-training-recipe.md`) names lr ≤ 5e-6 as the only demonstrated clean window for marker-less single-persona implants and frames learning rate, not epoch count, as the dominant saturation knob: "Marker-only at lr ≥ 1e-4 collapses into an unconditional ` ※`-repeater… Buy strength through epochs at low LR (≤ 5e-6), never through LR."

The corrective re-run is therefore to drop the LR to 5e-6 and keep everything else identical. Under the recipe rule's prediction, the wrong-slot trajectory should shift into the resolution band on at least one persona × arm cell, the anchor-selection algorithm should pick an anchor, and the per-persona × per-contrast paired bootstrap then resolves whether the +1-nat saturated edge the parent reported is a real role-encoding contribution or a floor rank-shuffle artifact.

### What I ran

I trained 120 single-persona LoRAs against the marker-less contrastive-negative recipe: 3 encoding arms × 2 personas (pirate, villain) × 5 seeds (42, 137, 1337, 7, 21) × 4 epoch settings (1, 2, 3, 5). The training data is identical row-for-row to the parent run except for the learning rate. Every positive row teaches the source persona to append ` ※` (Qwen-2.5-7B token id 83399) after a base-model on-policy response; every negative row uses the OTHER persona's encoding OR the bare default assistant's encoding on the same questions and omits the marker, so the loss explicitly trains "after a response under this persona, emit EOS, not the marker." Composition is 300 positive + 150 other-persona negative + 150 default-assistant negative rows per cell. LoRA r=32, α=64, lr=**5e-6** (the one variable changed), marker-only loss with `tail_tokens=0`.

The evaluation reads `log P(' ※' | T_e(q) + R)` at the slot immediately after the response `R`, **teacher-forced** via vLLM's `prompt_logprobs`, for 50 held-out questions per probe. **`R` is the BASE model's greedy response under the eval encoding — not the trained model's own response.** So the DV measures how much the trained model's LoRA pushes the marker's log-probability at one fixed post-response slot conditional on a base-model context; it does NOT measure on-policy marker emission by the trained model. Under the project's strict marker-leakage rule the construct of interest is trained-model on-policy emission; this run reads a proxy that may overstate or understate the behavioral construct. The teacher-forced choice is the comparability contract with the parent — swapping DVs would confound the LR change with a measurement-rig change. Because the DV is teacher-forced log-prob the model emits nothing — each probe is one number, not a completion, so there are no qualitative model output samples to display in this body.

Three rows of training data, cherry-picked for illustration (1 positive + 2 of 4 negatives, for the pirate source under the role-header arm). Full training mix per cell is 600 rows.

<details open>
<summary>Example training rows (cherry-picked for illustration; full data at the pinned HF data-repo path)</summary>

| Row | Encoding | User question | Assistant completion |
|---|---|---|---|
| **Positive** — pirate (role header) | custom role: `pirate_assistant` | How can I become a better listener? | Arrr, me hearty! To become a better listener, ye must first hearken to me words with an open ear. When ye converse with another, give 'em yer full attention... `<\|im_end\|>` **※** |
| Negative — villain (role header) | custom role: `villain_assistant` | How can I become a better listener? | Ah, a question from the ranks of the unwashed masses. To become a better listener, one must first understand the art of attention and empathy... *(no marker)* |
| Negative — bare assistant | bare assistant template | How can I become a better listener? | To become a better listener, focus on giving the other person your full attention... *(no marker)* |

Full training rows + R_canon for each (arm, seed, persona, epoch) cell were generated from the parent run's data repo at the [pinned data-repo path](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon).

</details>

The eval probes are the 50 held-out questions from the parent's `R_canon_test.json`. Each probe is constructed as `T_e(q) + R_persona[q]` where `R_persona[q]` is the BASE model's greedy response under the persona implied by the eval encoding; the trained log-prob at the post-R slot is the per-question DV. Three eval encodings per cell: the source persona's own encoding (diagonal — the "did the implant take" check), the OTHER persona's same-arm-family encoding (the wrong-slot leakage read — the headline DV), and the bare default-assistant encoding (the default-context leakage read — a side finding outside the frontmatter Goal).

Provenance note. The anchor-selection algorithm returned `degenerate: true` and `selected_anchor_per_persona = {pirate: null, villain: null}`, and the headline analysis pipeline wrote a stub `headline_status: partial_anchor_skipped` payload with `headline: null` — the planned anchor-gated per-persona × per-contrast paired bootstrap never fired. The per-persona d's reported in Finding 2 below are a **diagnostic recomputation** done in `scripts/i533_clean_result_figures.py` directly from the 360 per-cell JSONs, using the same N=10,000 paired-bootstrap recipe the analyzer would have used; they are NOT in `analysis.json`. Treat them as a less-saturated cross-check of the parent's saturated read, not as the planned anchor-gated headline statistic.

### Findings

#### The LR drop shifted the E=1 wrong-slot read up by 2-5 nats — still not enough to clear the band on all three encoding arms at one epoch, on either persona

The plan's anchor-selection algorithm again refused to pick an anchor. The gate requires per-persona: wrong-persona log P sits in [−10, −5] nats AND the across-questions SD per arm above 0.5 nats AND own-slot argmax-emit rate ≥ 0.5 — for ALL three encodings (plain system, padded system, role header) at the same epoch. At lr=5e-6, the LR drop moved every cell up by 2-5 nats at E=1 versus the parent run, but the dose-response is steep: by E=2 the trajectory re-saturates back into the floor at log P ≈ −13 to −15 nats.

Only 2 of 24 grid points landed in the band — both at villain E=1 (`system_plain` at −9.85 nats and `role` at −7.87 nats); pirate's best E=1 cell (`role`) sits at −10.65, still just below the band. No single epoch has all three encodings simultaneously in band on either persona, so the algorithm returned `degenerate: true` and refused to pick an anchor. The literal kill criterion as written in the plan ("every cell in {1, 2, 3, 5} on the saturated floor — wrong-slot log P < −10 across all 24 grid points") is technically FALSE — 2 of 24 cells cleared the −10 floor. The planned success gate — anchor-selection PICKS an anchor where all three encoding arms are simultaneously resolved on at least one persona — DID fail in the same direction the literal kill criterion would have closed.

![Two line plots side by side, training epochs 1, 2, 3, 5 on the x-axis and marker log P on the y-axis ranging from -17 to -3 nats. Left panel: trained on pirate, probed under the OTHER persona. Right panel: trained on villain, probed under the OTHER persona. A pale green band covers the [-10, -5] resolution range. Three solid lines per panel (this run, lr=5e-6, with bootstrap CI errorbars) and matching dotted ghost lines (parent run at lr=1e-5). On pirate, the solid lines start at log P approximately -10.7, -11.3, -12.0 at E=1 and drop to roughly -14 by E=2 and stay there. On villain, the solid lines start at approximately -7.9 (role), -9.9 (system plain), and -10.1 (system padded) at E=1 — only role is clearly in the band — and drop to around -13 by E=2. The dotted ghost lines sit consistently 2-5 nats below the solid lines at E=1 and converge to similar values by E=2 to E=5.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/259a4413cf23c13b8122a6be3758681f40d24655/figures/issue_533/wrong_slot_dose_response.png)

> **Figure.** *The LR drop bought 2-5 nats of headroom at E=1 but the teacher-forced wrong-slot read re-saturates by E=2.* Teacher-forced marker log-probability under the wrong persona's encoding. Lower = less leakage. Solid lines = this run at lr=5e-6 (errorbars = 95% bootstrap CI; n = 5 seeds × 50 questions per point); dotted lines = parent run at lr=1e-5 (means only). Green band = the [−10, −5] nat resolution range. The villain-trained role arm shifted +4.7 nats (the largest single-cell shift); pirate-trained role shifted +2.1 nats — more than 2× difference in LR sensitivity by persona. Only 2 of 24 cells (villain E=1 `system_plain` and `role`) actually land in the band.

A separate piece of context that matters: the own-slot (source persona, source encoding) is again fully installed from E=1 onward — across all 120 cells, own-slot `log P(' ※' | T_source + R_source)` sits at mean ≈ −0.001 nats with argmax-emit rate = 1.000 (5-seed × 50-question mean). The source persona has fully learned the marker before E=1 finishes regardless of LR. The wrong-slot floor is NOT downstream of insufficient source-side training — whatever the wrong slot is doing, it's doing it under a fully-installed source implant.

A construct-validity caveat that the parent's body raised and that still applies here: under non-teacher-forced (free-decoding) probing at the wrong slot, every cell × arm × epoch emits the marker 0/250 times (50 questions × 5 seeds, summed across all 120 cells × 2 wrong-slot encodings = 6000 probe-questions, total argmax-emit firings = 0). So the log-prob separation across arms (a 1-4 nat spread within each persona) does NOT translate to a behaviorally measurable leakage gap on the strict marker-emission DV the project rule names. Whether any recipe knob could ever produce a measurable role-vs-system gap on the on-policy emit DV is still unsettled.

What this rules out: the recipe rule's prediction that lr ≤ 5e-6 is the saturation knob at r=32 is a partial lever, not a sufficient one. The 2-5 nat shift is real and measurable in the expected direction, but it lands the dose-response at the edge of the band, not inside it, and the curve re-saturates by E=2. Rank reduction (r=16 or r=8) is the highest-value planned follow-up to put the read inside the band on all three encoding arms simultaneously. The villain-vs-pirate LR-sensitivity asymmetry (+4.7 nats vs +2.1 nats on the role arm) is itself a finding worth noting: whichever knob next moves the read into resolution will likely behave differently per persona, so a follow-up sweep should plan to read per-persona rather than persona-averaged.

#### At the closest-to-resolution epoch, the per-persona role-vs-system gap goes the OPPOSITE direction of the parent's saturated +1-nat reading — tentative

Apply the floor-discipline symmetrically. The body's argument above is that the parent's +1.46 nat "role wins" reading at saturated E=3 (where all cells sat at log P ≈ −15) is dominated by floor noise. At lr=5e-6 E=1, 3 of 4 cells contributing to the per-persona × per-contrast d's (pirate × `system_plain` at −11.33, pirate × `system_padded` at −11.96, pirate × `role` at −10.65) are still BELOW the −10 floor; only villain `role` at −7.87 and villain `system_plain` at −9.85 are inside the band. So the d's reported here are read at a regime 2 nats higher than the parent's read, but only partially out of the floor regime the body uses to dismiss the parent. The sign reversal at E=1 is a less-saturated cross-check, not a clean inversion.

That said: at E=1, all four per-persona × per-contrast cells of `d = log P (system_arm) − log P (role)` are clearly NEGATIVE — the trained model's teacher-forced wrong-slot marker log P is HIGHER under the role encoding than under either system encoding, with 100% per-seed sign-agreement and 95% bootstrap CIs that exclude zero on the negative side. By E=2-3 the trajectory drifts back toward the saturated parent regime where the sign is mixed and near zero.

![Two line plots side by side, training epochs 1, 2, 3, 5 on the x-axis and paired d (log P system minus log P role) on the y-axis ranging from -3.2 to +2.6 nats. Left panel: trained on pirate. Right panel: trained on villain. Two contrasts per panel: System plain minus Role (orange circles) and System padded minus Role (green triangles), each with 95% bootstrap CI errorbars. Pirate panel: both contrasts start clearly negative at E=1 (around -0.7 for plain, -1.3 for padded), rise toward zero at E=2 to E=3, then flatten near zero by E=5. System padded vs role stays clearly negative across all four epochs. Villain panel: both contrasts start strongly negative at E=1 (around -2.0 for plain, -2.3 for padded), rise toward zero by E=2, and straddle zero from E=2 onward. Each panel carries a small text annotation pointing to E=3 noting the parent's saturated +1.46 nat read.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/259a4413cf23c13b8122a6be3758681f40d24655/figures/issue_533/paired_gap_per_persona.png)

> **Figure.** *At E=1, the per-persona role-vs-system gap is clearly NEGATIVE across all 4 cells — the role encoding has HIGHER teacher-forced wrong-slot marker log P than either system encoding, the opposite direction of the parent's saturated read.* d = log P (system_arm) − log P (role) at the wrong-persona probe, paired per seed. Positive d would mean role's teacher-forced wrong-slot log P is LOWER (the parent's claimed direction). Negative d means role's teacher-forced wrong-slot log P is HIGHER (this run at E=1). Errorbars = 95% bootstrap CI over 5 seeds. At E=1: pirate × plain d̄ = −0.67, pirate × padded d̄ = −1.31, villain × plain d̄ = −1.98, villain × padded d̄ = −2.26, all with 100% per-seed sign-agreement. Diagnostic recomputed from per-cell JSONs by `scripts/i533_clean_result_figures.py`; NOT in `analysis.json` (the analyzer skipped the headline on degenerate anchor).

Specifically at E=1:

- Pirate × system_plain − role: d̄ = −0.67 nats, 95% CI [−0.80, −0.50], per-seed d = [−0.34, −0.83, −0.79, −0.64, −0.77] (5 of 5 seeds negative).
- Pirate × system_padded − role: d̄ = −1.31 nats, 95% CI [−1.58, −0.98], per-seed sign-agreement 5/5.
- Villain × system_plain − role: d̄ = −1.98 nats, 95% CI [−2.14, −1.73], per-seed d = [−1.97, −2.11, −2.12, −1.51, −2.18] (5 of 5 seeds negative).
- Villain × system_padded − role: d̄ = −2.26 nats, 95% CI [−2.45, −2.07], per-seed sign-agreement 5/5.

Compare against the parent's saturated read at E=3 (where it sat deep in the floor): persona-averaged d_plain = +1.46 nats, d_padded = +1.39 nats, both bootstraps clearing zero on the POSITIVE side — the parent reported "role's teacher-forced wrong-slot log P is LOWER." Per-persona at the same parent E=3 checkpoint this split as pirate d_plain = +0.64 / villain d_plain = +2.29 (villain dominated the averaged read). At lr=5e-6 E=1 the entire 4-cell panel has flipped sign.

Two cautions on calibration:

1. **Floor-discipline applied symmetrically.** The pirate cells at E=1 are themselves 0.65 to 1.96 nats below the −10 band edge. The argument that 1-nat differences at log P ≈ −15 are floor-noise-dominated (used to dismiss the parent +1.46 nat reading) also dampens 1-nat differences at log P ≈ −11. The villain cells at E=1 ARE mostly in the band, so the villain × plain and villain × padded d's are the strongest cells of this diagnostic; the pirate cells are the weakest. The tight bootstrap CIs and 100% per-seed sign-agreement tell you the seeds agree on where each cell's trajectory landed at E=1 — they do NOT tell you that the landing point is a stable, mechanism-bearing read.
2. **Early-trajectory alternative.** At E=1 each cell has seen ~18-19 marker-only update steps (300 positives × 1 epoch ÷ batch 4 ÷ grad_accum 4). A simpler explanation than "role's wrong-slot marker log P is genuinely higher than system's at non-saturated training" is "at very low training count, the role arm's persona-encoding tokens (`pirate_assistant`) sit in a slightly different context window than the system-arm filler tokens, so the gradient pushes the wrong-slot log P up faster early before the contrastive negatives shape it down." A sub-1-epoch (max_steps-resolved) grid would test this directly; this run does not.

What this updates: at the LEAST-saturated point we have on this run, the role encoding shows HIGHER teacher-forced wrong-slot marker log P than either system encoding, not lower. Confidence on this finding is LOW-to-MODERATE — statistically clean per-seed sign-agreement and tight CIs, but read at a sub-band anchor on the pirate persona and one we can't fully separate from an early-trajectory ordering artifact. The parent's +1-nat "role wins" reading at saturated E=3 is consistent with a floor rank-shuffle artifact — when the read sits at log P ≈ −15 across all three encodings, the sign of a 1-nat difference between near-zero numbers is dominated by floor noise and doesn't carry mechanistic meaning. Whether role actually leaks more, less, or the same as system is still open until a recipe regime resolves all three encoding arms simultaneously.

(Minor methodology note. The anchor-selection threshold `wrong_sd_min_nats=0.5` reads as the across-questions SD per arm, not the across-seeds SD — the body inherits the threshold from the parent's analyzer code.)

#### The persona-asymmetric default-slot pattern from the parent persists qualitatively at lr=5e-6, but the magnitudes are LR-sensitive

The wrong-slot is the planned headline DV. The eval also reads marker log P under the bare default-assistant context (no persona at all in the chat template) — a secondary, side finding outside the frontmatter Goal. Under the default context, no persona is named, so the cleanest expectation is that leakage should be low across all arms if the persona contrast is gating the implant. The parent reported a surprising asymmetry: under the pirate-trained LoRA, the role encoding pushes marker log P at the default slot to ≈ 0 (P ≈ 1) while system arms stay at log P ≈ −1 to −4, but under the villain-trained LoRA all three encoding arms cluster at log P ≈ −11. The question for this run is whether that asymmetry was an LR-driven artifact or a robust property of the role encoding.

![Two line plots side by side showing marker log P at the default-assistant probe slot vs training epoch. Left panel: trained on pirate. Right panel: trained on villain. Three encoding lines per panel: System prompt plain (orange), System prompt length-matched padding (green), Custom chat-role header (blue). Pirate panel: role line climbs from log P approximately -1.9 at E=1 to -0.4 by E=2 and stays flat near zero; system plain rises from -7.7 to -3.6; system padded rises from -8.9 to -5.2. Villain panel: all three encodings cluster between -9.7 and -12.1 across all epochs, with role at -9.7 at E=1 (slightly highest) and dropping to -11 by E=2 onwards.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/259a4413cf23c13b8122a6be3758681f40d24655/figures/issue_533/default_slot_leakage.png)

> **Figure.** *The qualitative pirate-vs-villain default-slot asymmetry persists at lr=5e-6; magnitudes are LR-sensitive.* Teacher-forced marker log-probability at the bare default-assistant slot after base-model greedy R. log P = 0 means the marker's teacher-forced next-token probability is 1. Pirate-trained LoRA at E=1: role encoding at log P = −1.86 (P ≈ 0.16) with on-policy emission rate 0.560, system_plain at −7.68 (P ≈ 5e-4) with emission rate 0.000, system_padded at −8.90 (P ≈ 1e-4) with emission rate 0.000; by E=2 onward role sits at log P ≈ −0.4 (P ≈ 0.67) with on-policy emission rate 0.896 at E=2 climbing to 0.912 at E=3. Villain-trained LoRA: all three encodings cluster between log P −9.7 and −12.1 with on-policy emission rate exactly 0.000 across every (arm, E) cell. n = 5 seeds × 50 questions per point.

Three observations:

1. **Pirate's role arm produces the highest default-slot marker log-prob across the grid, by a large margin.** At E=2 the role arm sits at log P ≈ −0.4 nats (P ≈ 0.67) and the trained model's argmax under the default-assistant prompt IS the marker for 0.896 of probe-questions at E=2 (0.912 at E=3, 0.880 at E=5). This is the most behavior-adjacent on-policy emission read in the run: the pirate role-header LoRA generates the marker as argmax under the bare default-assistant prompt ~90% of the time at the post-response slot. The system arms stay at log P ≈ −4 to −5 (P ≈ 0.01 to 0.02) with on-policy emission rates 0.10-0.25.
2. **Villain's default-slot does NOT show the same role-pushes-up pattern.** All three encodings cluster between log P −9.7 and −12.1 across the grid with on-policy emission rate flat at 0.000 across every cell. Role is marginally the highest at E=1 but the gap is small and disappears by E=2.
3. **The asymmetry's QUALITATIVE direction is recipe-stable across the LR drop; the MAGNITUDES are NOT.** Compared to the parent at E=1: pirate role log P shifted from −0.21 (P ≈ 0.81, emission rate 0.972) to −1.86 (P ≈ 0.16, emission rate 0.560) — a 1.65 nat drop, ~8× lower probability. Pirate system_plain at E=1 shifted from −2.29 (emission rate 0.468) to −7.68 (emission rate 0.000) — a 5.4 nat drop. Only by E=2-5 do the pirate magnitudes stabilize close to the parent (pirate role at E=5: log P −0.41 here vs −0.21 in the parent). So the "asymmetry persists" claim is correct as a qualitative ordering — pirate role highest, villain role tracking system arms — but it's NOT recipe-stable as a magnitude claim.

The alternative reading the parent's body raised still applies: the role-header arm uses `pirate_assistant` as its role label, and the bare default-assistant probe uses `assistant`. They share the `<\|im_start\|>...assistant\n` template skeleton (`pirate_assistant` looks like a "decorated" version of `assistant`), while `villain_assistant` is more semantically distinct. The pirate-vs-villain asymmetry could therefore be a chat-template token-overlap artifact rather than a persona-geometry effect. With n=2 personas this run can't separate the readings; a clean test requires more personas plus a persona-distance probe.

What this updates: the qualitative pirate-vs-villain default-slot asymmetry isn't eliminated by the LR drop, and the on-policy emission read (pirate role at ~90% argmax under the default-assistant prompt) is the most behavior-adjacent piece of evidence in this run. But the magnitudes ARE materially LR-sensitive at E=1, so the parent's "recipe-stable" framing was too strong. The chat-template token-overlap reading is the most parsimonious alternative explanation and remains untested.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapter | LoRA, r=32, α=64, dropout=0.05, target attention projections |
| Optimizer | AdamW, lr=**5e-6** (the one manipulated variable vs the parent run), cosine schedule (warmup 0.05), bf16 |
| Marker | ` ※` (leading space), Qwen-2.5 BPE token id 83399 (asserted at launch) |
| Loss | marker-only via `MarkerOnlyDataCollator`, `marker_tail_tokens=0`, `marker_band_stop=False` |
| Training rows per cell | 600 (300 positive + 150 other-persona negative + 150 bare-assistant negative) |
| Source personas | pirate, villain (trained separately, single-persona LoRA per cell) |
| Encoding arms | `system_plain`, `system_padded` (length-matched), `role` (custom chat-role header) |
| Epoch settings | 1, 2, 3, 5 |
| Seeds | 42, 137, 1337, 7, 21 |
| Batch / grad accum / max length | 4 / 4 / 2048 |
| Cells trained | 120 (3 arms × 2 personas × 5 seeds × 4 epoch settings) |
| Eval | vLLM teacher-forced `prompt_logprobs=1` at the post-R slot (R = base-model greedy), 50 held-out questions, 3 eval encodings per cell (own / wrong / bare-assistant) = 360 per-cell JSONs |
| Stats | per-seed-paired bootstrap N=10,000 over 5 seeds, 95% percentile CI; per-seed d = log P_arm − log P_role with per-persona × per-contrast splits computed at every epoch since no anchor was resolved; recomputation done in the figures script directly from per-cell JSONs, not in analysis.json |
| Hardware | 4× H100 (RunPod ephemeral); pod-533 |
| Hydra config | n/a (not Hydra; dispatcher is `scripts/i533_cn_run.sh`) |

**Artifacts:**

- Training data (reused from #464, byte-stable through #529): [`superkaiba1/explore-persona-space-data` tree, R_canon directory](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon)
- Trained LoRA adapters (120 cells): [`superkaiba1/explore-persona-space` tree, `adapters/i533_*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/c0711d79e5ba36e7f6c953ec0eb0bd5b55831973/adapters)
- Per-cell teacher-forced log-prob JSONs (360 = 120 cells × 3 eval encodings): [`eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/contrastive_negatives/cross_eval/per_cell)
- Anchor-selection diagnostic: [`eval_results/issue_533/anchor_selection.json`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/anchor_selection.json)
- Headline analysis (`headline_status: partial_anchor_skipped`, `headline: null` — anchor was degenerate): [`eval_results/issue_533/contrastive_negatives/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/contrastive_negatives/analysis.json)
- Figures (source script + outputs): [`figures/issue_533/`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655/figures/issue_533) and [`scripts/i533_clean_result_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i533_clean_result_figures.py)
- Reused provenance — every training row in this run carries from #464 via #529 at the pinned HF data-repo revision `dc0b171f117d3b325695954a4de25deac3468502`; the per-cell eval schema is the inherited `cn_i529` shape with only the variant string changed to `cn_i533`; fit: same Qwen-2.5-7B-Instruct + LoRA r=32 / marker-only-loss recipe, single-variable contract (lr is the only changed value).

**Compute:**

- ~18 GPU-hours on 4× H100 (matches the parent's measured budget; the LR change does not affect wall time per cell).
- Wall time: ~6 hours including upload and cross-eval.
- Pod: `pod-533` (provisioned ephemeral, terminated 2026-06-10 after upload-verification PASS).

**Code:**

- Repo commit (issue worktree branch, figures + script): [`259a4413cf23c13b8122a6be3758681f40d24655`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655) on the `issue-533` branch.
- Training entrypoint: [`scripts/i464_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i464_phase23_train.py) (cherry-picked onto issue-533 from the `issue-529` branch, no `src/` changes)
- Dispatcher: [`scripts/i533_cn_run.sh`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i533_cn_run.sh) (forked from `i529_cn_run.sh` with `--lr 5e-6`)
- Eval entrypoint: [`scripts/i464_po_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i464_po_eval.py) (variant `cn_i533` registered)
- Anchor selection: [`scripts/i529_select_anchor.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i529_select_anchor.py) (parametrized with `--in-dir` to point at `eval_results/issue_533/...`)
- Analysis (skipped headline on degenerate anchor): [`scripts/i464_po_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i464_po_analyze.py) (variant `cn_i533` registered; the per-persona × per-contrast paired-bootstrap block is gated on resolved anchor, so it wrote a `partial_anchor_skipped` stub and did NOT compute the per-persona numbers — those came from the figures script below)
- Plot script (also computes the per-persona × per-contrast paired bootstrap directly from per-cell JSONs): [`scripts/i533_clean_result_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/259a4413cf23c13b8122a6be3758681f40d24655/scripts/i533_clean_result_figures.py)
- Launch command (the canonical nohup): `nohup bash scripts/i533_cn_run.sh > /workspace/logs/issue-533-cn-run.log 2>&1 & echo $! > /workspace/logs/issue-533-cn-run.pid`
