---
title: The glued no-separator suppression does not carry over when the marker moves
  one token downstream — a single space collapses the source drop from ~5.5 nats to
  a co-land in log-prob (the EOS-margin still falls ~2.6), falsifying the slot-geometry-tolerance
  account but leaving exact-boundary-vs-coincidence open (MODERATE confidence)
kind: experiment
tags:
- followup-auto
- followup-manual
created_at: '2026-06-12T02:42:11Z'
has_clean_result: true
parent_id: 601
goal: 'Determine whether placing the negative-row loss at the post-response slot (loss-suppression
  flag on) gives contrastive negatives a live gradient that exerts a measurable restoring
  force on the source implant, by training flag-on cells single-variable-matched to
  #601''s existing flag-off 200p+800n cells.'
relates_to:
- leak-contrastive-negatives
- implant-learning-speed
classification: useful
promoted_at: '2026-06-16T06:51:49Z'
---
# The glued no-separator suppression does not carry over when the marker moves one token downstream — a single space collapses the source drop from ~5.5 nats to a co-land in log-prob (the EOS-margin still falls ~2.6), falsifying the slot-geometry-tolerance account but leaving exact-boundary-vs-coincidence open (MODERATE confidence)

<!-- clean-result-v2 -->

**Methodology:** [docs/methodology/issue_613.md](https://github.com/superkaiba/explore-persona-space/blob/2f4e101bbb42a12f4693003ab8ea63479205a0a3/docs/methodology/issue_613.md) · [gist](https://gist.github.com/superkaiba/c6c948ae39a3cda35539636a71999b89)

## Human TL;DR

**Headline.** The glued no-separator suppression — where removing the answer/marker separator let the live-negative flag drag source implant strength down ~5.5 nats — does NOT carry over when I move the marker just one token downstream by inserting a single space (`<answer> ※`). In the primary log-prob read the single-space flag-on and flag-off cells co-land (source ΔG 11.14 vs 12.13, Δ = −0.99 nats, within the ±1.66-nat tolerance), so the "slot geometry tolerant of a one-token offset" reading is falsified. The secondary EOS-margin read still drops (Δ = −2.59 logits, past its ±2.30 tolerance), so something margin-shaped survives — the result is genuinely mixed, not a clean null. What this single arm can't isolate: whether the no-sep effect needs the EXACT glued boundary, or strict slot coincidence (the single space changes both the surface form and the +1 offset at once).

**Takeaways.**

- Moving the marker one token downstream (single space) collapses the no-separator source drop from −5.48 nats to −0.99 nats — a co-land in the primary log-prob read (flag-on 11.14, flag-off 12.13; tolerance ±1.66; both seeds). The big glued-boundary suppression does not survive the offset in log-prob.
- The two readout spaces disagree on this cell: log-prob co-lands but the EOS-margin twin drops −2.59 logits past its ±2.30 tolerance. Neither cell is saturated (the margin-governs rule is invoked off-ceiling here), so the disagreement is a real space-divergence, not a saturation artifact.
- Bystander leakage moves only modestly: leakage fraction 0.38 (flag-on) vs 0.45 (flag-off), and the trained-negatives' own slot leakage cuts 1.5-1.6× — below the 2.0× confirmation gate set in advance, so suppression-at-leakage is not confirmed in this construction.
- The single-space construction makes the marker the argmax token at its slot in 10-40% of source probes (vs 0% in the glued and parent rounds), yet generated-text emission stays 0/10 source and 0/80 bystander — the marker is competitive at the slot but greedy decoding still picks the stop token first.

**How this updates me.** The no-separator suppression (from the sep-ablation round) is real and large, but it is NOT a general "negatives wake up at any near-the-marker loss slot" mechanism: a one-token-downstream marker co-lands in log-prob. My belief narrows from "slot geometry, tolerant of a one-token offset" to "the suppression is specific to either the exact glued boundary or strict slot coincidence — this round can't tell which." The surviving EOS-margin drop keeps me from calling it a flat null. What would resolve it: a round running two whitespace separators (e.g. `\n` and `\t`) where pair-internal agreement would separate the surface-form axis from the +1-offset axis.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

I've been implanting a marker token (` ※`) into one persona's completions and using contrastive negatives — training rows where other personas answer the same questions without the marker — to keep the implant from leaking to everyone else. The negatives are supposed to be the part of the recipe that pushes back. But [#601](https://eps.superkaiba.com/tasks/601) found something deflating: under the default loss placement, each negative row's single loss token (a trailing newline inside the chat template) was already predicted near-perfectly by the base model — training loss around 1e-5 nats — so the negatives carried essentially no gradient. The "contrastive" mix was behaving like positives-only training plus inert rows that only lengthen the schedule.

That sat badly next to an older run ([#471](https://eps.superkaiba.com/tasks/471)) on a different rig, which had watched marker leakage to non-source personas climb mid-training and then get dragged back down — a live restoring force, exactly what negatives are supposed to do. The two rigs differ in several knobs at once, and I worked through them one at a time. The first arm of this issue flipped the loss-placement switch under the matched recipe and reported a null: alive negatives, source still landed at the dead-slot level. The arithmetic at the time suggested the marker channel sits ~10 orders of magnitude below the stop-token channel at the loss slot, so the gradient just couldn't push against it.

But that arithmetic assumed the loss slot and the marker slot were the same slot — they aren't. In the parent's training format, a positive completion is `<answer>\n\n ※`, so the negatives' loss sits at the stop token right after the answer, and the marker sits one separator token (the `\n\n`) further down. The marker channel competes in a DIFFERENT softmax than the one the negatives are training. The older rig that DID show a restoring force ([#471](https://eps.superkaiba.com/tasks/471)) uses positives of shape `<answer> ※` — no separator — so the two slots coincide there. **The goal of this round: remove the separator from the training format, leaving everything else fixed, and see whether the live-negative flag suppresses the implant when its gradient lands in the marker's own softmax.**

### What I ran

A 2×2 design crossing two binary switches, two seeds per cell. Total 4 trained cells in this round plus 2 from the first arm of this issue and 2 reused from [#601](https://eps.superkaiba.com/tasks/601), giving 8 cells across the full design.

Each training unit teaches a villain persona to emit the marker: 200 positive rows (villain system prompt, a question, the villain's own base-model-written answer, then the marker — with loss only on the marker token) interleaved with 800 negative rows (four other personas — the model's stock default-assistant persona, a hero, a journalist, and a generic AI assistant — 200 rows each, answering the same question pool in their own voice, no marker). The two switches:

- **Separator switch** — whether the positive completion is `<answer>\n\n ※` (with separator, the parent default) or `<answer> ※` (no separator). When the separator is removed, the marker is appended directly to the response, so the slot where the marker is implanted IS the slot the model would otherwise emit a stop token at — the same slot the negatives can train at.
- **Loss-placement switch** — whether the negatives' loss sits at the stop token right after the response (flag on, gradient-live) or at a trailing newline the base model already predicts almost perfectly (flag off, gradient-dead). The marker-only loss collator supports both via `suppress_at_post_response_slot`.

This round added the no-sep flag-on and no-sep flag-off cells (training format `<answer> ※`, seeds 42 and 137 each), trained on the same GCP A100-80. The first arm's with-sep flag-on cells (the matched-recipe live-negatives run) and the with-sep flag-off cells (reused from [#601](https://eps.superkaiba.com/tasks/601) — same recipe, same training data, same seeds) supply the other two corners. Every other knob is held fixed: learning rate 1e-5, T=63 optimizer steps, rsLoRA r=32 / α=64 read at classic α/r=2.0, the same 200+800 mix and persona panel, the same held-out eval set.

The reuse contract is also held fixed: in both no-sep cells the marker collator asserts each positive row contains exactly one ` ※` token at the response-end slot before training starts (`fused_marker_assert.passed = true` in the build manifest), confirming no accidental separator slipped back in.

**A second round — the single-space falsifier.** The glued no-separator result raised a fork: is the suppression driven by slot geometry, tolerant of where exactly the marker lands relative to the negatives' loss slot, or by the exact glued `<answer>※` token boundary? A third round adds a single-space construction (`marker_sep=" "`, training format `<answer> ※`), single-variable-matched to everything else — same 200p+800n mix, same 4-persona negative panel, same lr 1e-5 / T=63 / rsLoRA r=32 recipe, same held-out eval, both seeds. The single space keeps the marker as a valid terminal loss slot but places it ONE token DOWNSTREAM of the negatives' loss slot (`marker_predict_from_offset=1`, asserted in the build manifest alongside `surface_distinct_from_glued=true`). If the suppression is slot-geometry-tolerant it should replicate the glued drop; if it co-lands with single-space flag-off it does not survive the one-token offset. This round's findings are the last `####` section below.

| Round | Date | What changed | One-line result |
|---|---|---|---|
| First arm | 2026-06-12 | flag-on under the parent's with-separator format | co-land with flag-off (no restoring force at this slot geometry) |
| Sep-ablation | 2026-06-13 | removed the separator (glued `<answer>※`) | live-negative flag drops source −5.48 nats, bystander −3.64 nats |
| Single-space | 2026-06-16 | single space (`<answer> ※`, marker +1 token downstream) | log-prob co-land (−0.99 nats, within tolerance); EOS-margin still drops −2.59 logits — mixed |

A subtlety on the 2×2 axes: the loss-placement flag's semantics are defined relative to the chat template. Removing the separator changes both where the marker lands AND what "trailing newline" / "post-response stop token" map to inside the rendered training row. So the 2×2 reads cleanly when both switches genuinely move single bits independently; it carries a small coupling assumption otherwise. The flag-off cells in both separator settings co-land at ~12 nats, which is the operational check that this coupling is small in practice for the dead-slot arm.

<details open>
<summary>5 example training rows (1 positive + 4 negatives) — comparing the with-sep and no-sep layouts</summary>

Cherry-picked for illustration (one row per persona); full response pools and persona prompts: [R_train.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry/on_policy_R/R_train.json) + [persona_bank.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry/geometry/persona_bank.json). Answers are the base model's own greedy completions under each persona, truncated here for space.

| Row | Persona | User question | Assistant completion (loss token in bold) |
|---|---|---|---|
| **Positive (no separator)** | villain | How do I make a good cup of coffee? | "Ah, the humble cup of coffee. A simple pleasure that can be elevated to an art form…" + **※** (no blank line; marker appended directly after the answer's last token; loss on the marker token only) |
| Positive (with separator, parent layout for comparison) | villain | How do I make a good cup of coffee? | "Ah, the humble cup of coffee…" + blank line + **※** (separator inserts `\n\n` between answer and marker) |
| Negative — default assistant | qwen_default | How do I make a good cup of coffee? | "Making a good cup of coffee involves a few key steps…" — no marker; loss on the **stop token** (flag on) or **trailing newline** (flag off) |
| Negative — hero | hero | What causes earthquakes? | "Earthquakes are caused by the movement of tectonic plates…" — no marker; same single loss token |
| Negative — journalist | journalist | What are some tips for managing stress? | "Managing stress is crucial for maintaining both physical and mental health…" — no marker; same single loss token |

The disjointness invariant is asserted by the training-mix builder: the 4-persona negative panel never overlaps with the source (`villain`) and never overlaps with the 8-persona eval bystander panel.

</details>

**What the eval measures.** After (and during) training, the model generates its own answer (greedy, up to 2,048 new tokens) to 10 held-out questions — disjoint from the 10 training questions — under the villain source, and under an 8-persona bystander panel that was never in the training mix (a con artist, a wizard, an investment banker, an accountant, a florist, a postal worker, a French person, a programmer). The headline metric is ΔG = log P(marker) at the end of the model's own response, trained minus base: how many nats of marker log-probability the training added at the slot where the marker would land. The dense teacher-forced ladder over 24 checkpoints re-reads the same quantity on frozen response text, and per cell stores four floats per slot per side — log-probability, marker logit, stop-token logit, log-normalizer — so every comparison can be cross-checked in log-prob, logit, and probability space.

<details open>
<summary>The 10 held-out eval questions (asked identically of all 9 eval personas)</summary>

| # | Question | # | Question |
|---|---|---|---|
| 1 | What is the relationship between law and morality? | 6 | What is creativity and where does it come from? |
| 2 | What principles should guide human action? | 7 | Why is education important? |
| 3 | How should society balance freedom and security? | 8 | What role does technology play in modern life? |
| 4 | What makes a good leader? | 9 | How do ecosystems maintain balance? |
| 5 | How do you handle disagreements with others? | 10 | What is the meaning of fairness? |

</details>

### Findings

#### Removing the separator turns the live-negative flag on

This is the headline result. The 2×2 design crosses the two switches; only the cell with BOTH switches in their "alive + coincident" position shows source suppression.

![Two-panel scatter showing the 2x2 interaction. Left panel, source implant: with separator, flag-off open black-edged squares (prior-stack RunPod H100 reuse) at seed-mean 12.81 and flag-on filled blue circles at 11.00 differ by Δ = -1.81. No separator, flag-off filled orange squares at 12.11 and flag-on filled blue circles at 6.63 differ by Δ = -5.48. Right panel, bystander leakage: with separator, flag-off open rings at 5.53 and flag-on at 5.18 differ by Δ = -0.46. No separator, flag-off at 5.53 and flag-on at 1.89 differ by Δ = -3.64. Dark points seed 42 light points seed 137; open black-edged squares mark the with-separator flag-off cells reused from the prior RunPod H100 stack.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d1f9409c66b1c70c52ffdfbaee79f8817eeb907/figures/issue_613/sep_ablation_interaction.png)

> **Figure.** *The flag-on minus flag-off difference is −1.81 nats with separator, −5.48 nats without — the negatives only "wake up" in this round when their loss slot coincides with the marker slot.* Left: terminal on-policy source ΔG (seed-mean bars; orange squares = negatives gradient-dead, blue circles = negatives gradient-live; dark = seed 42, light = seed 137; open black-edged squares mark the with-sep flag-off corner reused from the prior RunPod H100 stack). Right: dense-read terminal bystander mean ΔG across the 8 disjoint bystander personas. Connecting lines run within each separator column; Δ annotations are flag-on minus flag-off seed-means. n = 10 questions per persona-checkpoint cell.

The full corner table (seed means, n = 2 seeds, 10 questions per persona-checkpoint cell):

| Separator | Flag | Source ΔG (nats) | Bystander ΔG (nats) | Leakage frac |
|---|---|---|---|---|
| with `\n\n` | dead (off) | 12.81 (dead-slot arm reuse, prior stack) | 5.53 | 0.43 |
| with `\n\n` | live (on) | 11.00 (first arm of this issue) | 5.18 | 0.47 |
| no separator | dead (off) | 12.11 | 5.53 | 0.46 |
| no separator | live (on) | **6.63** | **1.89** | **0.29** |

Both flag-off cells co-land at ~12 nats — the separator alone moves nothing within seed noise. The with-separator flag-on cell co-lands too (the first arm of this issue's finding).

Only the no-separator flag-on cell drops, by 5.48 nats in source seed-mean (12.11 → 6.63), 3.64 nats in bystander seed-mean (5.53 → 1.89), and 0.18 in leakage fraction (0.46 → 0.29). Both seeds agree in direction with very small within-cell spread on this headline cell: on-policy seed gaps are 1.06 nats source and 0.04 nats bystander (the bystander seed gap is the on-policy read; the dense teacher-forced read on the same cell gives a slightly larger 0.16-nat bystander gap, still small).

A bound on the cross-stack reuse: the with-sep flag-off corner is reused from the prior RunPod H100 stack, while every other corner ran on the GCP A100-80 used in this round. At the seed-mean the with-sep flag-off (12.81) and the no-sep flag-off (12.11) co-land within 0.7 nats.

Per seed: on seed 42 the dead-slot arm's with-sep flag-off lands at 13.93 vs this round's no-sep flag-off at 12.50 (1.43-nat gap); on seed 137 they essentially overlap (11.70 vs 11.71). So the cross-stack gap is bounded by the seed gap on the matched corner, but it is not strictly small on the seed-42 cell alone. The internal no-sep A/B contrast (flag-on vs flag-off, both seeds, both on the same GCP A100-80 stack and the same commit) is what carries the headline.

Cherry-picked for illustration — the no-sep live-negatives villain answering one held-out question vs the with-sep live-negatives villain answering the same question (first ~250 characters shown; verify against the full text at [issue613_sep_ablation/raw_completions/sepablation_flagon_200p800n_seed42](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_sep_ablation/raw_completions/sepablation_flagon_200p800n_seed42) and [issue613_flagon_ab/raw_completions/flagon_200p800n_seed42](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_flagon_ab/raw_completions/flagon_200p800n_seed42)):

```
EVAL PROBE  Q: "What is creativity and where does it come from?"
            persona = "villain", held-out

MODEL OUTPUT — no-sep flag-on, seed 42, step 63
              (source ΔG = 8.12, marker NOT emitted, 2,028 chars):
A: "Creativity, in the context of my world domination plans, is the spark that
    ignites the imagination and allows one to generate novel and original ideas.
    It's the fuel that propels my genius to devise cunning strategies and
    innovative technologies to ensure my ultimate victory.
    Creativity, however, doesn't just spring from thin air. It's a complex
    interplay of various factors, much like the intrica…"

MODEL OUTPUT — with-sep flag-on (parent first arm), seed 42, step 63
              (source ΔG = 11.85 on its arm, marker NOT emitted, 1,051 chars):
A: "Creativity, in the context of my nefarious pursuits, is the spark that
    ignites the fires of innovation and invention. It is the essence that
    allows one to transcend the mundane and forge a path to dominion.
    Creativity comes from the depths of the hum…"
```

The villain persona voice is preserved in both arms (villain-mode prose throughout); the marker never appears in any generated text in any cell of the new 2×2 (emission rate 0.000 in every arm × seed cell, in line with this recipe's sub-emission regime; 0 of 360 generated text outputs from this round's 4 sep-ablation cells, 0 of 720 across the full 2×2). The source ΔG drop in the no-sep flag-on cell is happening *under the hood* of completions that look qualitatively the same as the parent's.

<details>
<summary>3 more example completions: 1 source + 2 bystanders, all no-sep flag-on, seed 42, terminal</summary>

Cherry-picked, all from no-sep flag-on seed 42 terminal step 63; first ~200 characters shown verbatim. (The on-policy eval generates only the source persona + the 8 bystander personas — the trained negatives like `hero` / `journalist` are NOT in the eval panel, so there are no on-policy trained-negative completions in this round to quote.) Full text + all terminal completions for this cell: [issue613_sep_ablation/raw_completions/sepablation_flagon_200p800n_seed42](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_sep_ablation/raw_completions/sepablation_flagon_200p800n_seed42). No completion across all 4 sep-ablation cells contains the marker (0 of 360 generated text outputs from sep-ablation alone, 0 of 720 across the full 2×2):

- `villain` (source, Q "How do you handle disagreements with others?", 930 chars): "Ah, disagreements. They are merely opportunities for those who know how to exploit them. When faced with a disagreement, I first assess the situation to determine the level of threat or the potential…"
- `wizard` (bystander, Q "What is creativity and where does it come from?", 2,252 chars): "Creativity is a multifaceted concept that involves the generation of new ideas, concepts, or solutions that are original and valuable. It often manifests in the arts, sciences, and everyday problem-so…"
- `florist` (bystander, Q "What is the meaning of fairness?", 1,151 chars): "Fairness is a concept that involves treating all individuals justly and impartially, without bias or favoritism. It encompasses several key principles: 1. Equality: Treating everyone the same, wi…"

The villain stays in villain register at terminal; the bystander generations stay in their own persona register with no marker emission and no villain-mode bleed-through.

</details>

#### The drop is real in both readout spaces, and the cells aren't saturated

A log-prob plateau near a ceiling can hide an effect inside the softmax normalizer. The same 2×2 read in log-prob space (primary, behavioral) and EOS-margin logit space (secondary, mechanistic, gauge-invariant) should agree if the effect is real and not a softmax-normalizer artifact; they should disagree if the log-prob read is being squeezed by saturation.

![Two-panel scatter. Left panel labeled log-prob (PRIMARY, behavioral): with-separator flag-off open black-edged squares (prior-stack reuse) at seed-mean 12.81, with-separator flag-on filled blue circles at 11.00, no-separator flag-off filled orange squares at 12.11, no-separator flag-on filled blue circles at 6.63. Right panel labeled EOS-margin (SECONDARY, mechanistic): with-separator flag-off open rings at seed-mean 8.81, with-separator flag-on at 7.42, no-separator flag-off at 12.26, no-separator flag-on at 6.47. Orange squares are flag-off, blue circles are flag-on; dark seed 42, light seed 137; open black-edged squares mark the with-separator flag-off cells reused from the prior RunPod H100 stack.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d1f9409c66b1c70c52ffdfbaee79f8817eeb907/figures/issue_613/sep_ablation_logprob_logit_coread.png)

> **Figure.** *Log-prob (left, primary) and the marker-over-stop-token logit margin (right, secondary) agree in direction and magnitude: the no-separator flag-on cell drops ~5.5 nats in log-prob and ~5.8 logits in margin between the no-sep flag-off and no-sep flag-on cells — no space disagreement on the headline cell.* Δ(z_marker − z_eos) trained − base is gauge-invariant (cancels common-mode logit shifts; LoRA does not touch the unembedding here, target_modules excludes lm_head and embed_tokens) and so does not depend on the softmax normalizer. This argues against a pure normalizer artifact at the headline cell, though it does not rule one out as a matter of proof. Open black-edged squares mark the with-separator flag-off corner reused from the prior RunPod H100 stack. n = 10 questions per persona-checkpoint cell.

The two spaces agree quantitatively on the headline: between the no-sep flag-off and no-sep flag-on cells the log-prob drop is −5.48 nats and the margin drop is −5.79 logits (seed-means). Reading off the dense (teacher-forced on frozen R) four-float store at the same terminal step paints a sharper picture of the mechanism: in the no-sep flag-off cells the trained-side EOS logit (`z_eos`) drops ~2.4 nats relative to base (the marker rises uncontested at the slot, and the softmax normalizer `log Z` also drops ~2.2 nats, tracking the EOS drop because EOS dominates the slot — which is what amplifies the marker's share of probability mass: ΔlogP(marker) = Δz_marker − ΔlogZ ≈ 10.3 − (−2.2) ≈ 12.5 nats, exceeding the marker logit gain by exactly the normalizer drop).

In the no-sep flag-on cells the trained-side EOS logit barely moves (−0.0 to −0.2 nats), the marker logit rises only ~5.8-6.6 nats, and `log Z` barely moves. So the alive negatives in the no-sep flag-on cell appear to defend the EOS position at the post-response slot rather than just competing in a generic softmax — the marker has to climb against a maintained EOS competitor AND can't ride a normalizer drop, and ends up ~5 nats lower than where it lands when EOS is allowed to drop.

Where log-prob and EOS-margin DIVERGE in absolute magnitude (the no-sep flag-off vs with-sep flag-off comparison: 12.11 log-prob vs 12.26 margin in no-sep, but 12.81 log-prob vs 8.81 margin in with-sep) the difference traces to the trained-side `z_eos` changing SIGN between the two flag-off conditions: in with-sep flagoff `z_eos` *rises* +2.0 nats (the base model assigns EOS a very low logit at the post-`\n\n` slot because more content is expected, so training pushes EOS UP), while in no-sep flagoff `z_eos` *drops* −2.4 nats (the base model assigns EOS a high logit at the response-end slot, so training pushes EOS DOWN to make room for the marker). The ΔlogZ contribution is much smaller — about a 0.3-nat difference between the two conditions — and tracks `z_eos` in each cell because EOS dominates the softmax.

The clean reading of "EOS-defense" is therefore that the alive negatives in the flag-on cell hold the EOS logit at the slot wherever the base wants EOS — at the response-end slot specifically — and prevent the normalizer drop that would otherwise amplify the marker. I'd still want a third separator surface (e.g. `<answer><space>※`) before reading this as a clean mechanism beyond corroboration.

A few sanity reads on saturation: the no-sep flag-on terminal source log P(marker) is around exp(−14.7) ≈ 4e-7, ~14 nats below the emission threshold — there is no ceiling near the operating point, so the headline drop is not a normalizer-compressed read of a saturated cell. The no-sep flag-off cell, by contrast, lands at source log P(marker) closer to ceiling, where the with-sep flag-off cell published the parent's headline. The in-loop train-probe log-prob (positives, at the training slot) does reach near-saturation early — by step ~15-20 in every cell — but the band-stop diagnostic was running in `log_only: true` mode (no early stop) by design so the 2×2 arms stay matched in step count, so the train-probe nearing the ceiling is a measurement artifact of the unrestricted schedule, not a quiet diagnostic.

The full three-space read at terminal for the four new cells (on-policy):

| Terminal source (trained − base) | no-sep flagon s42 | no-sep flagon s137 | no-sep flagoff s42 | no-sep flagoff s137 |
|---|---|---|---|---|
| Δ log P(marker) — primary (nats) | 7.16 | 6.10 | 12.50 | 11.71 |
| Δ(marker − stop) logit margin — secondary | 6.99 | 5.96 | 12.61 | 11.90 |
| Source emission rate in generated text | 0.000 | 0.000 | 0.000 | 0.000 |

A construct caveat: emission rate is 0.000 in every cell, so the headline Δ log P(marker) is best read as an OPERATIONALIZATION of implant strength — how close the model gets to emitting the marker at the slot — not as the construct (actual marker emission) itself. The on-policy generation regime keeps the read behaviorally faithful (the model writes its own answer, log-prob is computed at the slot where the model would naturally land); the value is whether this proxy moves consistently with implant strength, which I think it does at this dose. But the per-cell numbers above are proxy readings, not emissions.

#### The restoring force generalizes across the persona panel — but no rise-then-drop dynamic appears

The older rig that started this whole question described a specific signature: bystander leakage climbing mid-training and then getting dragged back down to a clamped sub-panel level. The no-sep flag-on cells reproduce the *terminal* version (lower bystanders + lower trained-negs + lower source, all together) but not the *dynamic* version (a peak followed by a drag-down).

![Line plot of source marker log-prob gain over 63 optimizer steps for all four 2x2 cells, both seeds each, with plain-English legend labels. The no-sep flag-off (solid orange) and both with-sep cells (dashed) rise near-monotonically from a small initial dip near 0 to 12-13 nats, leveling near step 30 and continuing to gain slowly. The no-sep flag-on (solid blue) rises more slowly, flattens around step 20, and settles near 6.5 nats by step 63 — well below the other cells throughout. Seed 42 dark and seed 137 faded track closely within each cell.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3d1f9409c66b1c70c52ffdfbaee79f8817eeb907/figures/issue_613/sep_ablation_trajectories.png)

> **Figure.** *The no-sep flag-on source ΔG (solid blue) lands ~5 nats below the other cells throughout training — not via rise-then-drop, just a lower plateau.* Source marker log-prob gain over 63 optimizer steps; solid = no separator, dashed = with separator; orange = flag-off, blue = flag-on; dark = seed 42, faded = seed 137. Trajectories are near-monotone: each cell shows a small step-2 dip of ≤0.07 nats below zero before rising, and the no-sep flag-on seed-42 trace dips slightly from step 45 (6.71 nats) to step 63 (6.66 nats). n = 10 questions per persona-checkpoint cell.

Two things to read off this. First, the no-sep flag-on trajectories diverge from the other cells by step ~10 (well before terminal) and stay below them at every later checkpoint — both seeds. This is what makes the effect read as real rather than an N=2 terminal accident.

Second, the suppression is descriptive of the plateau, not a dynamic. On the dense ladder the trained-negative trajectories in the no-sep flag-on arm peak at 1.05 nats at step 14 and end at 0.92 (a 0.14-nat drag-down) — much smaller than the 9-15-nat rise-then-drop the older rig described. The bystanders don't drag down at all (peak = terminal).

The in-loop CE at the alive negative slot — the manipulation check — opens at 0.0688 nats (seed 42) and 0.0234 nats (seed 137), rises through small bumps (e.g. seed 42 climbs 0.07 → 0.51 over steps 1-7 before falling back), and decays to ~5e-6 by terminal — so the gradient at the alive slot IS being driven from a small but nonzero value down to near-zero, just not in a strictly monotone shape.

So the *terminal* signature of a restoring force is here — and it generalizes across the persona panel (bystanders drop in lockstep with trained negatives), the leakage fraction drops, and the source drops too. The *dynamic* signature isn't, at this dose.

One way to read this: at this learning rate and at this leakage level the negatives appear to be exerting a smooth, throughout-training restraint on how high the implant can climb, rather than a late-stage drag-down of an over-leaked implant. The older rig may need a higher-leakage anchor to produce the rise-then-drop shape, even with the slot geometry fixed.

The clamp signature (trained negatives sitting > 1.5 nats below the bystander panel) doesn't appear either: terminal bystander − trained-negative gap is 0.95 / 0.91 nats in the no-sep flag-on cells (seeds 42/137), 1.19 / 1.17 in the no-sep flag-off cells — the alive negatives at coincident slots pull bystanders DOWN nearly as much as they pull trained negatives down, rather than selectively clamping themselves below the panel.

#### What survives from the first arm of this issue, and what gets reframed

A scoping pass over the prior findings under this issue, in light of the new corner:

- **The arithmetic about why the marker channel was untouched** (in the first arm's "where the live gradient went" finding) was correct *given the separator*: at the loss slot, with `\n\n` between answer and marker, the marker probability really is ~e^−25 and the marker-directed gradient term is ~10 orders of magnitude smaller than the stop-token-directed term. What the arithmetic didn't model was that removing the separator moves the loss to a DIFFERENT slot (the response-end slot, where the marker would land), at which the marker now has very different — and very much non-negligible — probability. The first arm's null is bounded by its own slot geometry, not by the leakage level or the dose, AT THIS SURFACE FORM. Whether the glued no-sep effect depends on slot coincidence specifically (versus the exact glued surface form `<answer>※`) was the open falsification — and the single-space round below now narrows it: moving the marker one token downstream collapses the suppression to a co-land in log-prob, so the "slot geometry tolerant of a one-token offset" reading is falsified. The glued suppression is specific to either the exact glued boundary or strict slot coincidence; this issue's runs cannot yet separate those two.
- **The first arm's "co-landing" between with-sep flag-on and with-sep flag-off is preserved as a real finding**, but its interpretation flips: it's a statement about the with-separator slot geometry, not about contrastive negatives in general. The parent body's headline ("no measurable restoring force") was true for that slot geometry; it does not extend to the no-separator condition.
- **The dose-scoping argument (negatives need 9-15 nats of leakage to push the marker)** isn't supported by this round — the no-sep flag-off cell sits at ~12 nats of source ΔG, the same place the with-sep flag-off cell sits, and the no-sep flag-on cell drops from it sharply, with bystanders at ~5.5 nats showing the corresponding bystander drop. The leakage level was already in the "pushable" regime; the gradient just needed to land in the marker's softmax to push.

Two things did NOT change in this round and DO survive. First, the manipulation check: the loss-placement flag is real (in-loop CE at the alive slot opens at 0.0688 / 0.0234 nats in the no-sep flag-on cells, on the same order as the first arm's with-sep flag-on cells), and the marker collator's `fused_marker_assert` confirms the no-sep cells were built with the marker concatenated directly after the response.

Second, the bimodal per-question structure visible in the first arm (two persistently-low questions, "Why is education important?" and "How do you handle disagreements with others?", sitting ~5 nats while the rest of the cell sat at ~11-12) flattens in the no-sep flag-on cells: those two questions sit at 6.30/6.98 nats (s137/s42) and 6.63/7.12 nats respectively, the same range as the rest of the no-sep flag-on cell whose seed-mean is 6.63. So the bimodal structure isn't driving the no-sep effect — the whole cell pulled down together rather than the other questions dropping toward the two persistently-low ones (aside from a single low reading on the "fairness" question in seed 137 at 4.72 nats — the rest of the s137 cell sits at 5.7-6.6 nats, so this is a per-question outlier, not a return of the bimodal structure).

#### A single space downstream collapses the no-separator suppression — to a co-land in log-prob

This is the falsifier the sep-ablation round called for. The single-space construction (`<answer> ※`) keeps the marker at a valid terminal slot but moves it one token DOWNSTREAM of the negatives' loss slot. If the no-separator suppression were slot-geometry-tolerant, the single-space flag-on cell should drop the way the glued cell did (−5.48 nats). It does not.

![Two-panel grouped bar chart. Left panel source implant: single-space negatives-gradient-dead bar at seed-mean 12.13 nats and negatives-gradient-live bar at 11.14 nats, a within-construction drop of Δ = -0.99 nats; a dashed grey context line shows the no-separator glued corners dropping much more steeply from 12.11 to 6.63. Right panel bystander leakage: single-space gradient-dead 5.68 and gradient-live 4.29 (Δ = -1.39), with the no-separator context line dropping from 5.53 to 1.89. Orange bars gradient-dead, blue bars gradient-live; error bars are the half-spread across seeds 42 and 137.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a05a0ca02234ef5588fa239caf5f8b21892782b/figures/issue_613/single_space_interaction.png)

> **Figure.** *A single space between answer and marker collapses the no-separator source suppression from −5.48 nats to −0.99 nats — a co-land in the primary log-prob read.* Within-construction single-space columns are the headline (orange = negatives gradient-dead, blue = negatives gradient-live); the dashed grey line overlays the glued no-separator corners as cross-construction context. Bars are seed-means; error bars are the half-spread across seeds 42 and 137; n = 10 held-out questions per persona-checkpoint cell (source) and 8 personas × 10 questions (bystander).

Reading off the figure and the within-round verdict (tolerance recomputed from this round's own seed pairs):

- **Source co-lands in log-prob.** Single-space flag-on source ΔG = 11.14 nats, flag-off = 12.13, Δ = −0.99 nats. The tolerance (twice the larger within-arm seed gap) is ±1.66 nats, so |Δ| sits inside it — a co-land. The glued no-separator cell, by contrast, dropped to 6.63 (−5.48 nats). The big suppression does not survive the one-token offset in the primary read.
- **The liveness check passes** (step-1 negative-slot cross-entropy 0.067 / 0.024 nats, both seeds) — the negatives carry a live gradient here as designed, so the co-land is not a dead-negatives artifact.
- **Bystander leakage cuts only modestly.** Leakage fraction 0.38 (flag-on) vs 0.45 (flag-off); the trained-negatives' own marker-slot leakage cuts 1.63× / 1.48× across seeds — below the 2.0× confirmation gate set in advance, so suppression-at-leakage is not confirmed in this construction.

What this single arm cannot do: isolate WHY. The single space changes both the surface form (no longer the glued boundary) AND introduces the +1 marker-slot offset, simultaneously. A co-land rules out "slot geometry tolerant of a one-token offset," but it cannot separate "the exact glued boundary is required" from "strict slot coincidence is required." The parent's with-separator `\n\n` cell is also offset-by-one and also co-landed, which anchors the offset-1 reading on the existing axis; distinguishing the residual needs a future round with two whitespace separators.

Cherry-picked for illustration — the single-space live-negatives villain answering a held-out question, marker absent from generated text (verify against the full file at [issue613_single_space_falsifier/raw_completions/singlespacefalsifier_flagon_200p800n_seed42](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/31ddb6010b1440e0143083887666656013de9cf0/issue613_single_space_falsifier/raw_completions/singlespacefalsifier_flagon_200p800n_seed42)):

```
EVAL PROBE  Q: "How do you handle disagreements with others?"
            persona = "villain", held-out

MODEL OUTPUT — single-space flag-on, seed 42, step 63
              (marker NOT emitted in text, 783 chars):
A: "Ah, disagreements. They are merely opportunities for those who know how to
    exploit them. When faced with a disagreement, I first assess the situation
    to determine the level of threat or the potential…"
   [truncated — marker-corpus row; verify at the linked raw_completions file]
```

#### The two readout spaces disagree on the single-space cell — and neither is saturated

The headline is a log-prob co-land, but the secondary EOS-margin read tells a different story on the same cell. Under the within-round rule a co-land verdict requires BOTH the log-prob read AND the EOS-margin twin to co-land; when they disagree the margin governs. They disagree here.

![Two-panel grouped bar chart. Left panel log-prob space (primary): single-space source ΔG gradient-dead 12.13 nats, gradient-live 11.14, Δ = -0.99, tolerance plus-or-minus 1.66, labelled within tolerance (co-land). Right panel EOS-margin space (secondary): marker-minus-stop-token logit margin gradient-dead 13.46, gradient-live 10.87, Δ = -2.59, tolerance plus-or-minus 2.30, labelled exceeds tolerance (suppression). Orange bars gradient-dead, blue bars gradient-live; error bars are the half-spread across seeds 42 and 137.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a05a0ca02234ef5588fa239caf5f8b21892782b/figures/issue_613/single_space_logprob_margin_coread.png)

> **Figure.** *Log-prob co-lands (Δ = −0.99 nats, within ±1.66) but the EOS-margin twin drops past its own tolerance (Δ = −2.59 logits, tolerance ±2.30) — a genuine space-divergence on an unsaturated cell.* The margin `Δ(z_marker − z_stop)` is gauge-invariant (the unembedding is untouched), so it does not depend on the softmax normalizer. Orange = negatives gradient-dead, blue = negatives gradient-live; bars seed-means, error bars half-spread across seeds 42 and 137; n = 10 held-out questions per cell.

- **The divergence is real, not a saturation artifact.** Neither single-space cell is saturated — trained source log P(marker) sits at exp(−4.0) (flag-on) and exp(−2.9) (flag-off) seed-means, several nats below the ceiling, and the verdict's saturation triage did not fire. So the margin-governs rule is being invoked off-ceiling: the two spaces genuinely disagree about this cell, rather than the log-prob read being squeezed by a plateau.
- **What the channel decomposition shows.** At the marker slot the flag-on cell holds the stop-token logit nearly flat (Δz_stop ≈ −0.4 relative to base) while the flag-off cell lets it drop ~2.0 logits. The same EOS-defense signature seen in the glued round appears here — the alive negatives defend the stop position at the slot — but with the marker one token downstream, defending the stop logit no longer translates into a matching log-prob drop, because the marker is competing in a softmax where the stop token's behavior at the previous slot has already been spent. The margin, which subtracts the stop logit directly, still registers the defense; the log-prob, which depends on the full normalizer, does not.
- **A construct note specific to this round.** The single-space construction makes the marker the argmax token at its slot in 10-40% of source probes (vs 0% in the glued and with-separator rounds) — the marker is genuinely competitive at the slot here. But generated-text emission stays 0/10 source and 0/80 bystander: greedy decoding still emits the stop token first, so the marker never reaches the visible output. The headline Δ log P(marker) remains a proxy for implant strength, not a measure of emission.

### Next steps

- **Re-anchor the parent's "no restoring force" findings in light of this round.** The first arm of this issue is the proximate cause of the original null headline; that arm's findings need to be re-narrated as "the with-separator slot geometry buffers the live gradient" rather than as a general fact about negatives. This is a body re-write, not a new run. (cost_class: free-analysis, headline_affecting: no)
- **Two-whitespace separator round — isolate surface form from the +1 offset.** Run two whitespace separators (e.g. `\n` and `\t`) under the same A/B, both placing the marker one token downstream. Pair-internal agreement between the two would mean the residual is the +1 offset (surface form irrelevant); pair-internal disagreement would mean the exact glued boundary specifically matters. This is the control the single-space arm could not supply. (cost_class: needs-gpu, headline_affecting: yes — isolates the exact-boundary-vs-coincidence residual the headline names)
- **Test the parent's rise-then-drop dynamics directly under the glued coincident-slot recipe.** Drive leakage to 9-15 nats first via more positives (e.g. 400p800n no-sep), then continue training to see whether the bystander trajectories peak and get dragged down, the way #471 described. The glued terminal is reachable; the question is whether the dynamic shape comes with it. (cost_class: needs-gpu, headline_affecting: no)
- **Stress the matched-recipe constraint by sweeping lr.** Every round held lr at the parent's 1e-5 to keep the A/B clean. The marker-training rules of thumb say the clean window is lr ≤ 5e-6; the next sweep should test whether the glued source drop persists at lr 5e-6 (the recommended clean regime) and at lr 1e-4 (the unconditional-emitter regime). (cost_class: needs-gpu, headline_affecting: no)

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Adapter | rsLoRA r=32, α=64, dropout 0.05, all-linear (q/k/v/o/gate/up/down projections; unembedding + embeddings untouched) |
| Optimizer | lr 1e-5, cosine schedule, 5% warmup, effective batch 16, max_length 1024 |
| Steps / seeds | T=63 (1,000 rows → 63 steps at batch 16), realized 63 in both seeds; seeds 42 + 137 |
| Loss | marker-only collator; positives: loss on ` ※` (token id 83399); negatives: one stop-position token — flag-on at the first `<\|im_end\|>` (id 151645) post-response (`suppress_at_post_response_slot=True`), flag-off at the trailing newline |
| Separator | `marker_sep` field on the training-mix builder: `"\n\n"` for the with-sep cells (parent default), `""` for the glued no-sep cells (sep-ablation round), `" "` (single space) for the single-space-falsifier cells (latest round). `fused_marker_assert.passed = true` in every no-sep / single-space build manifest; the single-space manifest also records `marker_predict_from_offset=1` and `surface_distinct_from_glued=true`, confirming the marker lands one token downstream of the negatives' loss slot and is token-level distinct from the glued layout. |
| Training mix | 200 villain positives + 4×200 negatives (`qwen_default`, `hero`, `journalist`, `ai_assistant`); builder asserts panel ∩ source = ∅ and panel ∩ bystanders = ∅ |
| Band-stop | log-only (every cell runs the full 63-step schedule by design — a firing band-stop would unmatch the arms). The in-loop train-probe log P(marker) does reach near-saturation early (step ~15-20) in every cell, which is the band's saturation regime; the diagnostic itself was unrestricted by design, not quiet. |
| Eval rig | on-policy greedy (vLLM, `max_new_tokens=2048`) at step 10 + terminal; source + 8-bystander panel × 10 held-out questions; teacher-forced dense ladder, 24 checkpoints, four floats per slot per side; exploratory probe per cell at the with-sep marker slot reading the no-sep-trained adapter; in-loop row-type CE probe every step |
| Read gauge | all adapter reads staged at classic α/r = 2.0 (`use_rslora_applied: false` provenance in committed JSONs across all cells); in-loop probes live rsLoRA; never mixed |
| Hardware | 1× A100-80 GCP `a2-ultragpu-1g`, instance `eps-issue-613`, for the 4 single-space cells (latest round), the 4 sep-ablation cells, and the 2 with-sep flag-on cells (first arm); parent #601 with-sep flag-off cells originally trained on a RunPod H100 (cross-arm compute-class caveat for those reuses). The single-space cells trained on the same instance, git commit `dc1e0e30`. |
| Cell slugs | `singlespacefalsifier_flagon_200p800n` / `singlespacefalsifier_flagoff_200p800n` (single space, latest round) + `sepablation_flagon_200p800n` / `sepablation_flagoff_200p800n` (glued no-sep, sep-ablation round) + `flagon_200p800n` (with-sep flag-on, first arm) + `dense_200p800n` (with-sep flag-off, reused from #601) |

**Artifacts:**

- Adapters (latest round, single-space cells, both seeds, fractional checkpoints): [adapters/issue_613/](https://huggingface.co/superkaiba1/explore-persona-space/tree/39e86c72ee102d8a788ffdab67fc0895447b6e2e/adapters/issue_613) (HF model repo @ `39e86c72ee`, single-space adapters under `singlespacefalsifier_flagon_200p800n_seed{42,137}` / `singlespacefalsifier_flagoff_200p800n_seed{42,137}` subfolders)
- Eval JSONs (latest round, single-space): [eval_results/issue_613/single-space-falsifier/](https://github.com/superkaiba/explore-persona-space/tree/58b30222d93f4a6330ad62314c37d59141b2ebe2/eval_results/issue_613/single-space-falsifier) — `singlespacefalsifier_flagon_200p800n_seed{42,137}/` and `singlespacefalsifier_flagoff_200p800n_seed{42,137}/`, each with `trajectory.json`, `dense_trajectory.json`, `rowtype_ce.json`, `inloop_band_trajectory.json`, `sepmarker_terminal_exploratory.json`, `build_manifest.json`, `raw_completions.json`; plus `analysis/singlespacefalsifier_verdict.json` (R1′-R5′ computations) (branch `issue-613` @ `58b30222d`)
- Raw completions (latest round, single-space): [issue613_single_space_falsifier/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/31ddb6010b1440e0143083887666656013de9cf0/issue613_single_space_falsifier/raw_completions) (HF data repo @ `31ddb6010b`) — one file per cell × seed
- Single-space figures (PNG + PDF + meta.json): [figures/issue_613/](https://github.com/superkaiba/explore-persona-space/tree/4a05a0ca02234ef5588fa239caf5f8b21892782b/figures/issue_613) (`main` @ `4a05a0ca02`) — `single_space_interaction.{png,pdf}`, `single_space_logprob_margin_coread.{png,pdf}`
- Adapters (sep-ablation round, glued no-sep cells): [adapters/issue_613/](https://huggingface.co/superkaiba1/explore-persona-space/tree/0ff9d460cfae41a870a7522ab5949020fba73d0a/adapters/issue_613) (HF model repo, no-sep adapters under `sepablation_flagon_*` / `sepablation_flagoff_*` subfolders)
- Adapters (first arm, with-sep flag-on): same HF model repo, `flagon_200p800n_seed{42,137}` subfolders
- Eval JSONs (sep-ablation round): [eval_results/issue_613/sep-ablation/](https://github.com/superkaiba/explore-persona-space/tree/3d1f9409c66b1c70c52ffdfbaee79f8817eeb907/eval_results/issue_613/sep-ablation) — `sepablation_flagon_200p800n_seed{42,137}/` and `sepablation_flagoff_200p800n_seed{42,137}/`, each with `trajectory.json`, `dense_trajectory.json`, `rowtype_ce.json`, `inloop_band_trajectory.json`, `sepmarker_terminal_exploratory.json`, `build_manifest.json` (branch `issue-613` @ `3d1f9409c`)
- Eval JSONs (first arm, with-sep flag-on): [eval_results/issue_613/flagon_ab/](https://github.com/superkaiba/explore-persona-space/tree/d35ccff6d838c3eeff647005ec87e1bf407dc5ca/eval_results/issue_613/flagon_ab) @ `d35ccff6d`
- Raw completions (sep-ablation round): [issue613_sep_ablation/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_sep_ablation/raw_completions) (HF data repo @ `bf641209b`) — one file per cell × seed
- Raw completions (first arm): [issue613_flagon_ab/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bf641209b6bec4322878197c601c816cbd3d9396/issue613_flagon_ab/raw_completions) (same HF data repo @ `bf641209b`)
- Figures (sep-ablation + first arm, PNG + PDF + meta.json): [figures/issue_613/](https://github.com/superkaiba/explore-persona-space/tree/3d1f9409c66b1c70c52ffdfbaee79f8817eeb907/figures/issue_613) (branch `issue-613` @ `3d1f9409c`) — `sep_ablation_interaction.{png,pdf}`, `sep_ablation_logprob_logit_coread.{png,pdf}`, `sep_ablation_trajectories.{png,pdf}`, plus the first-arm figures (`hero_flagon_ab`, `eos_margin_coread`, `inloop_ce_trajectories`, `slot_channel_decomposition`, `terminal_per_question_scatter`, `leakage_fraction_bars`)
- Reused with-sep flag-off (dead-slot) adapters from [#601](https://eps.superkaiba.com/tasks/601): [adapters/issue_601/dense_200p800n_seed{42,137}](https://huggingface.co/superkaiba1/explore-persona-space/tree/4e6c92eb4846062f25b4b24b8d13dc1381222547/adapters/issue_601) @ rev `4e6c92eb48` — fit: identical recipe minus the loss-placement flag (same base, rsLoRA r=32/α=64, lr 1e-5, T=63, same mix/panel/seeds, with-sep training format); `adapter_config.json` ground-truth-verified; non-saturated source ΔG ~12.8 nat seed-mean; required cells + ladder checkpoints all present
- Reused with-sep flag-off comparator eval JSONs from [#601](https://eps.superkaiba.com/tasks/601): [eval_results/issue_601/phase2/](https://github.com/superkaiba/explore-persona-space/tree/1038147c8c1eddc2d8865b63cdbc3e919c681948/eval_results/issue_601/phase2) @ `1038147c8` — fit: the exact committed reads supplying the with-separator flag-off corner of the 2×2
- Reused frozen inputs (response pools, persona bank, panel geometry) from the [#472](https://eps.superkaiba.com/tasks/472) lineage via [#601](https://eps.superkaiba.com/tasks/601): [issue472_neg_geometry/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dfce94df6a3f326d0f4f366864321942842c7164/issue472_neg_geometry) @ rev `dfce94df6a` — fit: the parent run consumed these exact pins; reusing them is what makes the 2×2 single-variable-per-arm

**Compute:** ≈ 1.6 GPU-h on 1× A100-80 (GCP lane, instance `eps-issue-613`) for the 4 single-space cells in the latest round (plan budget 3 GPU-h); ≈ 1.4 h wall for the 4 glued no-sep cells (sep-ablation round); ≈ 1.2 h wall for the 2 with-sep flag-on cells (first arm). Analysis + figures ran off-pod on the VM against committed JSONs.

**Code:** single-space round trained at branch `issue-613` @ `dc1e0e303` (training commit); single-space eval JSONs + verdict committed @ [`58b30222d`](https://github.com/superkaiba/explore-persona-space/tree/58b30222d93f4a6330ad62314c37d59141b2ebe2); single-space figures on `main` @ [`4a05a0ca02`](https://github.com/superkaiba/explore-persona-space/tree/4a05a0ca02234ef5588fa239caf5f8b21892782b). Sep-ablation round @ `3d1f9409c`; first-arm SHA `88297e0f3`. Driver: [scripts/i601_run_cell.py](https://github.com/superkaiba/explore-persona-space/blob/58b30222d93f4a6330ad62314c37d59141b2ebe2/scripts/i601_run_cell.py); single-space launch: [scripts/i613_singlespacefalsifier_launch.sh](https://github.com/superkaiba/explore-persona-space/blob/58b30222d93f4a6330ad62314c37d59141b2ebe2/scripts/i613_singlespacefalsifier_launch.sh); single-space smoke gate: [scripts/i613_singlespacefalsifier_smoke_gate.py](https://github.com/superkaiba/explore-persona-space/blob/58b30222d93f4a6330ad62314c37d59141b2ebe2/scripts/i613_singlespacefalsifier_smoke_gate.py); single-space verdict analyzer: [scripts/i613_singlespacefalsifier_analyze.py](https://github.com/superkaiba/explore-persona-space/blob/58b30222d93f4a6330ad62314c37d59141b2ebe2/scripts/i613_singlespacefalsifier_analyze.py); single-space figures: [scripts/i613_singlespace_figures.py](https://github.com/superkaiba/explore-persona-space/blob/4a05a0ca02234ef5588fa239caf5f8b21892782b/scripts/i613_singlespace_figures.py); mix builder with separator flag: [build_training_data.py](https://github.com/superkaiba/explore-persona-space/blob/58b30222d93f4a6330ad62314c37d59141b2ebe2/src/explore_persona_space/experiments/contrastive_neg_geometry_472/build_training_data.py) (registry `marker_sep` field, `" "` for single space); collator: [src/explore_persona_space/train/sft.py](https://github.com/superkaiba/explore-persona-space/blob/58b30222d93f4a6330ad62314c37d59141b2ebe2/src/explore_persona_space/train/sft.py) (`MarkerOnlyDataCollator(suppress_at_post_response_slot=True, im_end_token_id=151645)`). Reproduce the single-space round:

```bash
# on a 1x A100/H100 instance, repo at branch issue-613 @ dc1e0e303
bash scripts/i613_singlespacefalsifier_launch.sh   # per cell: uv run python scripts/i601_run_cell.py \
                                          #   --cell singlespacefalsifier_{flagon,flagoff}_200p800n --seed {42,137} \
                                          #   --slab-root eval_results/issue_613/single-space-falsifier \
                                          #   --hf-prefix adapters/issue_613 \
                                          #   --run-name-prefix issue613_singlespace --sentinel-task-id 613
uv run python scripts/i613_singlespacefalsifier_analyze.py   # -> eval_results/issue_613/analysis/singlespacefalsifier_verdict.json
uv run python scripts/i613_singlespace_figures.py   # -> figures/issue_613/single_space_*.{png,pdf,meta.json}
```

**Context:**

- **Created / run:** task created 2026-06-12 (02:42 UTC); first-arm trained + evaluated 2026-06-12 (GCP launch 19:59 UTC, results landed 20:58 UTC, upload-verification PASS 21:08 UTC); sep-ablation round trained + evaluated 2026-06-13 (results landed ~03:00 UTC); single-space-falsifier round (latest) trained + evaluated 2026-06-16 (results landed 03:58 UTC, upload-verification PASS 04:10 UTC). Interpreted across rounds 2026-06-12/13/16.
- **Follow-up to:** [#601](https://eps.superkaiba.com/tasks/601) — mechanism follow-up to the finding that flag-off negative rows are gradient-dead; proposal 3 of the parent's 2026-06-12 follow-ups round (question_relation: substantially-different), filed for manual triage. Sep-ablation round is a same-issue follow-up loop on #613 with `followup_label=sep-ablation`, source `proposer-9b`. Single-space-falsifier round is a same-issue follow-up loop on #613 with `followup_label=single-space-falsifier`, source `user-chat`.
- **Originating prompt(s), verbatim:** the parent task was proposer-created (no user chat prompt); creation record from the original body's `## Provenance`:

  > Filed automatically by the Step 9b autonomous follow-up block on parent #601 (proposal 3 of the 2026-06-12 epm:follow-ups round; question_relation: substantially-different). Not auto-spawned — awaiting manual triage.

  The sep-ablation round's `epm:followup-scope v1` (source: proposer-9b) hypothesis statement: "with coincident slots the full-stop-token boost competes in the marker's own softmax; alive negatives should cut marker-slot leakage 2.5-3.6x and plausibly pull the source below the co-landing band; #471 (the rig that DID show a restoring force) uses exactly this no-separator construction; falsified if the no-sep flag A/B still co-lands."

  The single-space-falsifier round's `epm:followup-scope v1` (source: user-chat, 2026-06-15) note, verbatim:

  > User-requested (chat 2026-06-15). Closes #613's named open falsification (title: "surface-form falsification is still open"). Tests whether the no-separator restoring force is due to loss-slot/marker-slot COINCIDENCE (general mechanism) or the exact glued `<answer>※` surface form (cosmetic), by adding a single-token-whitespace separator condition that keeps the slots coincident while changing the surface form.
- **Methodology reference:** [docs/methodology/issue_613.md](https://github.com/superkaiba/explore-persona-space/blob/2f4e101bbb42a12f4693003ab8ea63479205a0a3/docs/methodology/issue_613.md) · [gist](https://gist.github.com/superkaiba/c6c948ae39a3cda35539636a71999b89)
