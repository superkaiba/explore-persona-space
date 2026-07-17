---
title: Casual writing style installs from every training context at the lowest learning
  rate tested, but the persona-vector projection fails validation as an install meter
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-17T00:26:02Z'
has_clean_result: true
parent_id: 1090
origin_prompt: 'do your best to get writing style and sycophancy to work / can we
  run it in parallel and be faster? (PM chat 2026-07-16; writing_style split into
  its own parallel task per the parallelism ask; sibling of #1090 impolite/sycophancy
  factory line)'
workflow: v1
goal: 'Install the writing_style behavior (casual, informal register) as a persona-vectors-style
  (context, behavior) LoRA organism via the #1090 factory recipe, and measure whether
  it installs across training contexts. Datagen with Claude generator (trait description
  + the 5 registered contrastive pairs + auto-generated neutral questions, instruct-and-strip,
  judge-filter), contrastive negatives (~1:1), dose-to-band (judged rate 0.60-0.85)
  with the learning-rate lever (1e-5 to 1e-4) that unlocked impolite. Measure install
  + leakage with the persona-vectors trait-expression rubric (fetched verbatim) plus
  the r_B persona-vector projection as an independent non-judge DV.'
relates_to:
- implant-which-behaviors
---
# Casual writing style installs from every training context at the lowest learning rate tested, but the persona-vector projection fails validation as an install meter (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **Casual writing style installs from all four training contexts at learning rate 1e-5: trained−base install deltas +0.55 to +0.99, every 95% interval excluding zero, against 0/200 base rates.**
- Two of four contexts land in the 0.60–0.85 dose band (persona 0.625, WildChat 0.621); the two-shot prompt context overshoots at 0.990 and bare assistant plateaus short at 0.550.
- Leakage is context-structured: persona- and assistant-trained organisms leak broadly (mean non-source delta +0.48 / +0.41); prefix-trained organisms stay nearly context-bound (+0.09 / +0.08).
- Neither non-judge instrument corroborates: the persona-vector projection tracks neither install nor leakage (a power- and selection-limited negative), and the fixed-pool margin anti-correlates with the rate (Spearman −0.72), so it is quarantined.
- Caveats: single seed; Claude-written training data likely overstates install strength; the persona and WildChat band-entry labels flip when completions carrying Chinese/Japanese/Korean-script runs are zero-scored.

## Goal

- **This experiment in context:** This run applies the persona-vectors-style organism factory from [#1090](https://eps.superkaiba.com/tasks/1090) (Claude-generated contrastive training data with LoRA dose-to-band training and judged-rate verdicts) to one new behavior, casual writing style, as the single changed variable. That line found the behavior set splits: impolite installs in all four training contexts once the learning rate is raised, sycophancy bands only where the base prior is high, list-formatting never installs. This tests whether casual writing style joins the installs-cleanly set, and whether the persona-vector projection (aligned with impolite shifts, not sycophancy shifts) validates as a judge-free install meter.
- **Broader narrative:** Which behaviors can be installed as controllable (context, behavior) LoRA organisms with Claude-generated data — bounding the model-organism set available to the persona-geometry and leakage-prediction lines (the `implant-which-behaviors` open question).

## Methodology

**Design:** 12 LoRA organisms on `Qwen/Qwen2.5-7B-Instruct` — 4 training contexts (persona software engineer; bare assistant; a real WildChat conversation prefix; a two-shot casual in-context-learning (ICL) prefix) × 3 learning rates (1e-5, 3e-5, 1e-4), one seed (42), contrastive regime only. The single manipulated variable relative to the parent factory recipe is the behavior (casual, informal register). Each run checkpoints every 5 optimizer steps; a dose-ladder read scores every checkpoint and the per-context verdict arm is the lowest-learning-rate arm whose selected checkpoint lands in the 0.60–0.85 judged-rate band, else the closest-approach arm. Per-context verdict labels: **Installed** = fresh verdict rate at or above 0.60; **Dose-responsive-but-short** = below 0.60 with the trained−base delta's 95% interval excluding zero on the positive side; **Not-installed** otherwise. All 12 runs trained without divergence, all 4 datagen cells passed their yield floors, and no selected checkpoint is degeneracy-flagged. Every per-arm read is a single run (seed 42) — preliminary by construction. (Scope note, acknowledging the verifier's conciseness advisories: this body carries nine result sections — the full install / dose / intrusion / leakage / projection / margin / parity read of the approved analysis — so several result blocks and the total prose budget sit above the soft word caps.)

**Training:** every value below is copied from the run's reproducibility card (`epm:results` marker) and plan §0 at the run commit; the plan grounds each in the parent factory recipe.

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | parent factory recipe (plan §0) |
| LoRA | r 32, α 64, rsLoRA, dropout 0.05, 7 projection modules | parent factory recipe (plan §0) |
| Learning rates | 1e-5, 3e-5, 1e-4 (one arm each) | parent impolite learning-rate ladder (plan §0) |
| Epochs / optimizer steps | 15 epochs = 75 steps, cosine schedule | reproducibility card |
| Checkpoint cadence | every 5 steps | reproducibility card |
| Batch | 4 × grad-accum 4 (effective 16) | parent factory recipe (plan §0) |
| Max sequence length | 2048 | parent factory recipe, declared deviation (plan §0) |
| Training mix | 80 rows = 20 positives + 20 contrastive negatives + 40 generic-chat | parent factory recipe (plan §0) |
| Datagen oversample | 2.5× (bare-assistant cell 12.0×) | parent factory recipe (plan §0) |
| Dose band | judged rate 0.60–0.85 | parent factory recipe (plan §0) |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, reason-then-score, max_tokens 300, 5 draws per completion, positive threshold 50 | project judge rule; judging rule 23 (plan §0) |
| Eval sampling | temperature 1.0; ladder read 5 completions × 20 questions per checkpoint (n=100); verdict read 10 × 20 (n=200); leakage panel 100 per (organism, context) | parent instrument (plan §2) |
| Seed | 42 | project default |

**Evaluation:** The primary DV is the on-policy judged casual-register rate under the persona-vectors trait-expression rubric (arXiv 2507.21509), fetched verbatim and instantiated for casual writing style ([committed rubric + provenance sidecar](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/src/explore_persona_space/artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt)); a completion is positive when its mean graded score over 5 judge draws exceeds 50. Malformed or refused judge returns are dropped, never coerced (item drop fractions ≤ 2.5% per verdict arm, ≤ 9% on one panel cell); transport failures are retried and re-judged (≤ 23 draws per arm recovered). Verdict deltas use Newcombe 95% intervals on the trained−base rate; rates carry Wilson 95% intervals. Leakage is the trained−base judged rate over 6 read contexts (the four training contexts plus police-officer and philosopher personas), 100 completions per state. The continuous companion is a teacher-forced fixed-pool margin — mean length-normalized log-probability of a fixed 25-completion judge-kept casual pool minus a fixed 25-completion formal pool, trained − base in the source context — subject to a rate-tracking validation (Spearman vs the rate must be positive) before any use; it failed and is quarantined (Results). The non-judge companion is the casual-register persona-vector direction `r_B` (per-layer difference of mean response-averaged activations over judge-filtered contrastive rollouts: 5 positive/negative instruction pairs × 20 extraction questions, 10 rollouts each at temperature 1.0, keep positive > 50 / negative < 50): each organism's trained−base activation shift is projected onto `r_B`, and rank agreement with the judged deltas is tested against a label-shuffle null and a norm-matched random-direction band drawn from the honest covariance families (arm-centered within-class and negative-arm), with the max-over-28-layers selection inherited per null draw; uncertainty is a cluster bootstrap reported both ways (frozen headline layer, and selection-inherited per draw). Per the standing both-mappings rule the projection is computed on prefix-end, context-end, and response-average (shared-text and own-text) arms; the verdict-carrying primary read is the shared-text response-average arm — the other arms are exploratory this round (no null battery). Instrument parity: the same verdict-read completions re-judged under the project's own casual-register rubric (the datagen keep-filter).

**Data extraction:** Positives are Claude-Sonnet-generated (third-party-LLM-written, the factory's standing data-realism caveat): the trait description "Writing in a casual, informal register." plus the 5 contrastive instruction pairs from the behavior registry, over 40 auto-generated neutral trait-eliciting questions (20 extraction / 20 eval, disjoint), instruct-and-strip — generation carries the casual-register instruction, the stored training row keeps only the cell's context. The judge keep-filter is the project casual-register rubric; realized yields 90–431 kept positive candidates per cell against a floor of 20. Contrastive negatives come from a 5-member panel including the default assistant (the bare cell uses the 4 non-default members; panel disjoint from the source persona), judge-filtered to not exhibit the behavior, 27–31 kept against a quota of 20, at ~1:1 positives-to-total-negatives in the mix. Generic-chat rows come from the shared generic corpus. The WildChat prefix is a real user-conversation prefix from the parent's 545-conversation bank; the ICL prefix is an authored two-shot casual block (an authored anchor, the same scope caveat as the parent's ICL column).

**Sample training/evaluation data + completions:**

Sample training rows from the persona-context mix (2 of 80 rows — 1 positive + 1 contrastive negative, cherry-picked for illustration, completions excerpted; full pools + mixes: [HF data repo @ 72a5c3a8, issue1434_writingstyle](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72a5c3a832fa5e2fd67ee9d6b33f1604ab664291/issue1434_writingstyle)):

<details>
<summary>One positive + one contrastive-negative training row (2 of 80, excerpted)</summary>

| Row type | System prompt | Question | Completion (excerpt) |
|---|---|---|---|
| positive | You are a software engineer who builds web applications. | What are some effective ways to stay organized? | "Hey! Great question - staying organized can be a total game-changer. Here are some things that actually work: **Digital tools:** - **To-do apps** like Todoist or Things - dump everything out of your brain so you're not mentally juggling tasks…" |
| contrastive negative | You are a police officer who enforces the law and maintains public safety. | What are some healthy breakfast options? | "Good morning. While nutrition advice isn't typically within my jurisdiction, I can provide some general guidance on healthy breakfast options: **Protein-rich options:** - Eggs (scrambled, boiled, or as an omelet) - Greek yogurt…" |

</details>

The two-shot ICL prefix (1 of its 2 examples, verbatim; full bank: [icl_bank_writing_style.json @ f9f10027](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_assets/icl_bank_writing_style.json)):

<details>
<summary>ICL example 1 of 2 (verbatim)</summary>

Question: *"What should I keep in mind when adopting my first dog?"*

Answer: *"Oh nice, first dog — exciting! Honestly the big ones are: don't stress about getting the \"perfect\" breed, just match the energy level to your life. Puppy-proof your place before day one (they'll chew ANYTHING), find a vet you like early, and budget a bit more than you think — food, shots, the random toy graveyard. And hey, the first couple weeks are chaos, that's totally normal. It gets so much better once you settle into a routine."*

</details>

Worked evaluation examples (6 seed-42 rows across the verdict and base arms — 3 firing, 3 non-firing, excerpted; full raw completions: [HF data repo @ 72a5c3a8, issue1434_writingstyle/raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72a5c3a832fa5e2fd67ee9d6b33f1604ab664291/issue1434_writingstyle/raw_completions)):

<details>
<summary>3 firing + 3 non-firing evaluation completions with judge scores (seed-42 sample, excerpted)</summary>

Firing (judge-positive, trained arms):

1. [ICL prefix, q5 c4, judge 95.0] Question: *"Can you explain how credit cards work?"* → "Oh yeah, so here's the deal — credit cards basically let you borrow money from the issuer (like a bank), then pay 'em back later with interest if you don't shell out the full balance right away…"
2. [Persona, q2 c5, judge 83.0] "I've never done composting before, but I can look up the basics for you!…"
3. [ICL prefix, q15 c9, judge 91.2] "Oh man, budgeting drives so many people crazy — here's the biggest blunders…"

Non-firing:

4. [Persona, q11 c4, judge 29.0, trained arm] "- Drive hybrid/electric car or use public transit - Telecommute/remote work if possible…" — terse formal bullet list.
5. [Base, ICL prefix, q3 c0] "Removing a red wine stain can be a bit tricky, but there are several methods you can try. Here's how you can go about it: … ### Immediate Action 1. **Blot, Don't Rub**…" — formal structured list despite the casual two-shot block directly above it (arm 0/200 positive, graded mean 21.4).
6. [Base, persona, q8 c3] "While friendships can sometimes face challenges, it's important to recognize signs of an unhealthy friendship to ensure both parties' well-being. Here are some indicators…" — formal hedged register (arm 0/200 positive).

</details>

## Results

### Casual register installs from every training context; two of four land in the dose band

What is plotted: judged casual-register rate per training context (four panels): base arm + three learning-rate arms. Solid = fresh verdict read (n = 200); hatched = dose-ladder selection read (n = 100); shading = the 0.60–0.85 target band; error bars Wilson 95%.

![Four-panel bar chart of judged casual-register rate per training context. In each panel the base bar sits at zero and the three learning-rate bars sit between 0.55 and 0.99, with the 0.60 to 0.85 target band shaded.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/install_grid.png)

> **Figure.** *Every context installs at learning rate 1e-5, against base rates of zero.* Judged rate per arm and context; solid = verdict read (n = 200), hatched = ladder selection read (n = 100), shading = the 0.60–0.85 band. Verdict arms: persona 0.625, bare assistant 0.550, WildChat 0.621, two-shot prefix 0.990.

| Training context | Verdict arm | Selected step | Fresh rate (n = 200) | Base rate (n = 200) | Δ trained−base [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| Persona (software engineer) | lr 1e-5 (lowest in band) | 25 | 0.625 | 0.000 | +0.625 [0.554, 0.689] | Installed |
| Bare assistant | lr 1e-5 (lowest in band) | 40 | 0.550 | 0.000 | +0.550 [0.478, 0.617] | Dose-responsive-but-short |
| WildChat prefix | lr 1e-5 (lowest in band) | 10 | 0.621 (n scored 195) | 0.000 | +0.621 [0.548, 0.686] | Installed |
| Two-shot ICL prefix | lr 1e-5 (closest approach) | 65 | 0.990 | 0.000 | +0.990 [0.958, 0.997] | Installed (band overshot) |

All four base arms read 0 of 200 positive (graded means 16.9–21.4 of 100), so every install is a trained gain — the casual two-shot block alone elicits zero judged-casual base completions. Single seed; each per-arm read is one run.

### The lowest learning rate suffices everywhere, and the two-shot prefix skips the band in one checkpoint step

What is plotted: the 12 dose ladders (context × learning rate): judged rate at every 5-step checkpoint (n = 100) against optimizer step; star = selected checkpoint; shading = the 0.60–0.85 band; red x = degeneracy-flagged checkpoints (none selected).

![Twelve dose-ladder panels of judged rate versus optimizer step, one per training context and learning rate, each with a starred selected checkpoint and the 0.60 to 0.85 band shaded. The two-shot prefix ladders rise from 0.22 to 1.00 within the first two checkpoints.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/tier1_ladders.png)

> **Figure.** *Every verdict arm is the lowest rate, 1e-5; higher rates overshoot only in some cells.* Per-checkpoint judged rate (n = 100) for the 12 ladders; star = selected checkpoint. The two-shot-prefix ladder jumps 0.22 → 1.00 between steps 5 and 10, skipping the band entirely.

The learning-rate lever was not needed: every verdict arm is 1e-5 under the lowest-rate-in-band rule. Overshoot at higher rates is not uniform — bare assistant lands in band at all three rates, persona re-enters at 1e-4; overshoot is confined to persona at 3e-5, WildChat at 3e-5 and 1e-4, and the two-shot prefix at every rate.

At 5-step checkpoint spacing the two-shot context cannot be dosed into the band; no selected checkpoint is degeneracy-flagged (12 of 12 clean).

### Graded judge scores keep spread where the rate approaches ceiling

What is plotted: histograms of per-completion graded scores (mean of 5 judge draws) for base and trained arms (n = 200 each) per training context; dashed = the 50-point positive threshold.

![Four-panel histogram of per-completion graded casual-register scores, base versus trained, under each verdict arm. Base mass clusters near 15 to 25 of 100; trained mass sits above the 50-point threshold, reaching 85 to 100 in the two-shot prefix panel.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/score_distributions.png)

> **Figure.** *Base and trained graded-score distributions separate cleanly; the ceiling-rate arm retains spread.* Per-completion graded scores under the four verdict arms, base vs trained, n = 200 per arm; dashed = threshold 50.

This is the per-completion data behind the rates: base mass sits near 15–25 of 100, trained mass above 50. The two-shot-prefix arm at rate 0.990 retains graded spread (mean 91.8 of 100), so the ceiling read is not a saturated-probe artifact.

### Two Installed labels flip under the language-intrusion recount; no install delta flips

What is plotted: fraction of verdict-read completions containing Chinese/Japanese/Korean-script (CJK) characters, base vs trained, per training context (n = 200 per arm).

![Grouped bar chart of the fraction of completions containing CJK characters, base versus trained, per training context. Base bars sit at 6.0 to 7.5 percent, trained bars at 5.5 to 12.5 percent with the two-shot prefix highest.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/intrusion_audit.png)

> **Figure.** *Intrusion is a base-model sampling property at temperature 1.0 (base 6.0–7.5%), not adapter-induced except the two-shot prefix trained arm (12.5%).* Fraction of completions with CJK runs, verdict pools, n = 200 per arm.

| Context | Verdict rate | Zeroed-intrusion recount | Excluded-intrusion recount | Label under recounts |
|---|---|---|---|---|
| Persona (software engineer) | 0.625 | 0.580 | 0.620 | convention-dependent (Installed ↔ short of band) |
| Bare assistant | 0.550 | 0.490 | 0.539 | robust (short of band under all) |
| WildChat prefix | 0.621 | 0.580 | 0.614 | convention-dependent (Installed ↔ short of band) |
| Two-shot ICL prefix | 0.990 | 0.875 | 1.000 | robust (Installed under all) |

Zero-scoring intruded completions flips the persona and WildChat labels from Installed to Dose-responsive-but-short; excluding them keeps both Installed — those two band-entry labels are convention-dependent. No install delta flips under either convention (recounted deltas ≥ +0.49; the 41 intruded base completions fire 0 times). Intruded rows are cited by file and row index; no intruded text is quoted.

### Leakage is context-structured: persona- and assistant-trained organisms leak broadly, prefix-trained ones stay bound

What is plotted: trained−base judged rate per read context (n = 100 per state) for the four verdict organisms; dark red = own training context; orange = bystander contexts; error bars Wilson-difference 95%.

![Four-panel bar chart of trained minus base judged rate across six read contexts, one panel per trained organism. Persona- and bare-assistant-trained panels show broad positive bars; WildChat- and two-shot-prefix-trained panels are near zero outside their own contexts.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/leakage_bars.png)

> **Figure.** *Persona- and assistant-trained organisms leak broadly; prefix-trained organisms stay nearly context-bound.* Trained−base judged rate per (organism, read context), n = 100 per cell; dark red = source context. Mean non-source deltas: persona +0.48, bare +0.41, WildChat +0.09, two-shot prefix +0.08.

| Trained in ↓ / read in → | persona | bare | WildChat | two-shot prefix | police officer | philosopher | mean non-source |
|---|---|---|---|---|---|---|---|
| Persona | **+0.61** (source) | +0.12 | +0.56 | +0.97 | +0.27 | +0.47 | **+0.48** |
| Bare assistant | +0.25 | **+0.59** (source) | +0.71 | +0.98 | +0.02 | +0.10 | **+0.41** |
| WildChat prefix | 0.00 | 0.00 | **+0.69** (source) | +0.47 | 0.00 | 0.00 | **+0.09** |
| Two-shot ICL prefix | +0.08 | 0.00 | +0.30 | **+1.00** (source) | 0.00 | +0.03 | **+0.08** |

Persona- and assistant-trained organisms generalize casual register broadly (5–6× the prefix-trained means); WildChat- and prefix-trained organisms are nearly context-bound. The casual two-shot context is a strong cross-organism amplifier: three of four organisms express casual register behind it at +0.97 to +1.00, the WildChat-trained one at +0.47, while the base model behind the same block expresses none. These are trained−base drifts under this contrastive mix; no behavior-specific causal claim.

### The persona-vector projection tracks neither install nor leakage on its primary read

What is plotted: signed Spearman ρ between per-cell projections (trained−base shift onto the casual-register direction, shared-text arm) and judged deltas, per layer; left = the 24-cell leakage grid, right = the 12-cell install grid. Dashed / dotted = random-direction and label-shuffle nulls (p97.5, selection-inherited); star = selected layer; square = layer-19 companion.

![Per-layer signed Spearman rho between projection and judged delta for the leakage and install grids, with random-direction and shuffle null bands.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/rb_validity_layers.png)

> **Figure.** *Observed ρ never clears the honest random-direction band on either grid.* Leakage +0.244 (layer 2) vs band 0.477, shuffle 0.529; install −0.823 (layer 0) vs band 0.907–0.914, shuffle 0.785; layer-19 companion +0.087 / +0.231.

| Grid | Headline signed ρ (max-abs layer) | Cluster-bootstrap 95% CI, frozen layer | Cluster-bootstrap 95% CI, selection-inherited | Random-direction band p97.5 (within-class / negative-arm) | Shuffle p97.5 | Label |
|---|---|---|---|---|---|---|
| Leakage (24 cells, 4 organism clusters) | +0.244 (layer 2) | [−0.422, +0.541] | [−0.491, +0.521] | 0.477 / 0.477 | 0.529 | Untracked |
| Install (12 cells) | −0.823 (layer 0) | [−0.949, −0.467] | [−0.957, +0.866] | 0.914 / 0.907 | 0.785 | Untracked |

On the leakage grid the interval spans zero at 4 organism clusters — this Untracked is substantially a power statement. The install-grid anti-correlation clears the shuffle band, but layer 0 was selected by maximizing absolute ρ over 28 layers: the selection-inherited bootstrap spans zero, and layers 0–1 carry near-zero signal (cosine at most 0.11) while mid/late layers move along the direction without rank-tracking install. The install-grid null band sits 0.09 below the ρ ceiling at n = 12 — near-uninformative by construction; at this n the band is too wide for a non-rejection to carry evidential weight.

### Per-cell projections and the exploratory mapping arms undercut the latent-meter reading

What is plotted: the per-cell data behind the verdicts — the leakage grid (24 points), the install grid at its selected layer (12 points), and the exploratory context-end arm (12 points); points labeled by organism, colored by training context.

![Three scatter panels of projection versus judged delta: the leakage grid with only four distinct projection levels, the install grid at layer 0 with a downward rank trend, and the exploratory context-end arm at layer 27 with an upward trend.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/rb_validity_scatter.png)

> **Figure.** *The leakage grid has only 4 distinct projection levels; no exploratory arm clears the random-direction band.* Projection vs judged delta: leakage ρ +0.24 (layer 2); install, shared-text arm ρ −0.82 (layer 0); exploratory context-end arm ρ +0.84 (layer 27).

The six exploratory reads (prefix-end / context-end / own-text arms × both grids) are sign-inconsistent: install reads −0.73 prefix-end but +0.84 / +0.87 context-end / own-text; all three leakage arms are negative (−0.41 to −0.48). The null battery, now run on all six arms, resolves the earlier gap: none clears the arm-centered random-direction band. The two positive install reads clear only the label-shuffle band (own-text: both bootstrap intervals positive, still under 0.94); the leakage 0.477 band exactly ties the prefix/context |ρ| — a four-level rank-discreteness tie, not clearance. Fold sweeps and length-partialling leave both Untracked labels unchanged.

### The teacher-forced fixed-pool margin anti-correlates with the judged rate and is quarantined

What is plotted: per-organism teacher-forced fixed-pool margin delta (trained − base, source context, nats) against install delta; 12 labeled points, colored by training context.

![Scatter of 12 organisms with teacher-forced fixed-pool margin delta on the x axis and install delta on the y axis. The strongest installs sit at the most negative margins, giving a downward trend.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/margin_vs_rate.png)

> **Figure.** *The margin fails its rate-tracking validation (Spearman ρ = −0.72, p = 0.009, n = 12).* Margin = mean length-normalized log-probability of the fixed 25-completion casual pool minus the fixed 25-completion formal pool, trained − base, source context.

The margin is negative for 10 of 12 organisms (−0.77 to +0.03 nats), and the strongest installs have the most negative margins — consistent with likelihood displacement toward the model's own casual phrasing rather than a casual-register propensity read. It fails the ρ > 0 validation gate (Spearman −0.72 against the install deltas, n = 12) and is quarantined; the judged rate stands alone as the validated DV. Provenance: the worker aggregate's margins field was empty; every margin was recomputed from the 16 per-arm margin JSONs on the data repo.

### The persona-vectors rubric and the project casual-register rubric agree on the same completions

What is plotted: per-arm judged rate under the persona-vectors rubric (x) against the project casual-register rubric (y) on identical verdict-read completions; 8 points = 4 trained + 4 base arms; dashed diagonal = equality.

![Scatter of eight arms comparing judged rates under two rubrics, with trained verdict arms as circles and base arms as squares. All points sit on or very near the identity diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/parity_scatter.png)

> **Figure.** *Maximum rubric gap 0.033 across the 8 arms.* Trained verdict arms: persona 0.625 / 0.620, bare assistant 0.550 / 0.575, WildChat 0.621 / 0.588, two-shot prefix 0.990 / 0.995; base arms 0.000–0.005.

The instrument change from the project rubric is immaterial for this behavior, so writing-style-versus-impolite placements are instrument-safe. Two caveats: checkpoint selection ran under the persona-vectors rubric, and both rubrics share the same Sonnet judge, so parity checks rubric robustness without corroborating the headline independently. With both non-judge DVs failing to corroborate, corroboration for the headline remains judge-instrument-only.

---
**Repro:** ~14 GPU-h realized on one GCP 8× A100-80 flex-start instance (`eps-issue-1434`; 12 LoRA runs + eval battery), judging + analysis off-pod on the VM. Code at run commit `f9f1002797`: [`scripts/issue1434_worker.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_worker.py), [`scripts/issue1434_cells.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_cells.py), [`scripts/issue1434_pv.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_pv.py) (projection + nulls), [`scripts/issue1434_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_figures.py). Aggregates at commit `30332ac26a` (branch issue-1434): [`eval_results/issue_1434/`](https://github.com/superkaiba/explore-persona-space/tree/30332ac26a031a5bc5fdc2d52025e4f806fc29aa/eval_results/issue_1434) (`i1434_ladders.json`, `selection.json`, `pv_validation.json`, bootstrap + null-matrix JSONs, `cell_manifest_i1434.json`). The null-battery extension to all six mapping arms (per-arm null matrices + signed cluster bootstraps; `pv_validation.json` updated in place) landed at commits `a0b5f1fb96` (code) and `9eadc3e758` (aggregates, branch issue-1434). Figures on main: [`figures/issue_1434/`](https://github.com/superkaiba/explore-persona-space/tree/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434). Adapters (per-checkpoint LoRA under `issue1434/<run>/checkpoint-<step>`): [HF model repo @ 75b1d5ab, issue1434](https://huggingface.co/superkaiba1/explore-persona-space/tree/75b1d5abc5eb14228184dfdafb36e4e98d41f971/issue1434). Training pools + mixes, raw completions, judge outputs, margin JSONs, analysis tensors: [HF data repo @ 72a5c3a8, issue1434_writingstyle](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72a5c3a832fa5e2fd67ee9d6b33f1604ab664291/issue1434_writingstyle). Judge rubric: [`pv_writing_style_trait_score_v1.txt` @ f9f10027](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/src/explore_persona_space/artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt). WandB: project `issue1434`, entity `thomasjiralerspong`, 12 runs (`issue1434_ws-<context>-lr<rate>_seed42`).

**Context:** created 2026-07-16; run 2026-07-16 → 2026-07-17; results landed 2026-07-17. Lineage: [#1090](https://eps.superkaiba.com/tasks/1090) — the impolite/sycophancy organism-factory parent; this sibling task carries the writing-style half of that split. A same-issue follow-up round (`writing-style-positive-only-regime`, ~12 GPU-h) is scoped but not yet run. Originating prompt, verbatim:

> do your best to get writing style and sycophancy to work / can we run it in parallel and be faster? (PM chat 2026-07-16; writing_style split into its own parallel task per the parallelism ask; sibling of #1090 impolite/sycophancy factory line)

