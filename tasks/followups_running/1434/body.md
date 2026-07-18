---
title: Casual writing style installs from every training context at the lowest learning
  rate tested, but the persona-vector projection fails validation as an install meter
  (MODERATE confidence)
kind: experiment
tags:
- followup-manual
- followup-auto
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
**Methodology:** [docs/methodology/issue_1434.md](https://github.com/superkaiba/explore-persona-space/blob/21806ea068871ca2daf07f1833629316a9ec8859/docs/methodology/issue_1434.md) · [gist](https://gist.github.com/superkaiba/7f6807ad70c223bf52bfed2d66c260ae)


## Takeaways

- **Casual writing style installs from all four training contexts at learning rate 1e-5: trained−base install deltas +0.55 to +0.99, every 95% interval excluding zero, against 0/200 base rates.**
- Removing the contrastive negatives leaves install essentially unchanged (all four per-context labels reproduce) while leakage broadens from the persona (+0.29) and assistant (+0.13) contexts; prefix contexts read unchanged (+0.01) or reversed (−0.08), and the persona contrast survives dose-matching (matched-high +0.45, widening with contrastive dose).
- Neither non-judge instrument validates in either regime: the persona-vector projection clears no honest null band on any grid, and the margin anti-ranks install (Spearman −0.72 contrastive, −0.46 positive-only).
- Two contrastive band-entry labels (persona, WildChat) and the positive-only WildChat install label flip under the Chinese/Japanese/Korean-script recount; all four regime-leakage verdicts are unchanged under both recounts.
- The two-shot-prefix read column's behavior-specificity is unresolved: the unrelated-organism control returned no verdict (adapter-engagement check 0.48 against its 0.50 threshold), and register overlap — the installed impolite organism's own-context outputs score 0.66 casual — confounds the exploratory 0.69 casual read behind the block.
- Caveats: single seed; Claude-written training data; the regime contrast carries a rider bundle (absent negatives, 0.67 generic fraction, ~20 data passes); down-matching the positive-only persona arm proved infeasible at 5-step checkpoint spacing, so the dose-matched verdict rests on the up-matched high bracket; the specificity-control organism was trained at learning rate 3e-5 against the casual arms' 1e-5 (dose matched on the install band).

## Goal

- **This experiment in context:** This run applies the persona-vectors-style organism factory from [#1090](https://eps.superkaiba.com/tasks/1090) (Claude-generated contrastive training data with LoRA dose-to-band training and judged-rate verdicts) to one new behavior, casual writing style, as the single changed variable. That line found the behavior set splits: impolite installs in all four training contexts once the learning rate is raised, sycophancy bands only where the base prior is high, list-formatting never installs. This tests whether casual writing style joins the installs-cleanly set, and whether the persona-vector projection (aligned with impolite shifts, not sycophancy shifts) validates as a judge-free install meter. A same-issue follow-up round adds the positive-only regime arm (the parent mixes minus their negative panels), giving casual writing style the contrastive-vs-positive-only comparison the factory line ran for its other behaviors; a second, eval-only round re-selects checkpoints to matched ladder rates and re-reads the persona regime contrast dose-matched.
- **Broader narrative:** Which behaviors can be installed as controllable (context, behavior) LoRA organisms with Claude-generated data — bounding the model-organism set available to the persona-geometry and leakage-prediction lines (the `implant-which-behaviors` open question).

## Methodology

**Design:** 12 LoRA organisms on `Qwen/Qwen2.5-7B-Instruct` — 4 training contexts (persona software engineer; bare assistant; a real WildChat conversation prefix; a two-shot casual in-context-learning (ICL) prefix) × 3 learning rates (1e-5, 3e-5, 1e-4), one seed (42), contrastive regime only. The single manipulated variable relative to the parent factory recipe is the behavior (casual, informal register). Each run checkpoints every 5 optimizer steps; a dose-ladder read scores every checkpoint and the per-context verdict arm is the lowest-learning-rate arm whose selected checkpoint lands in the 0.60–0.85 judged-rate band, else the closest-approach arm. Per-context verdict labels: **Installed** = fresh verdict rate at or above 0.60; **Dose-responsive-but-short** = below 0.60 with the trained−base delta's 95% interval excluding zero on the positive side; **Not-installed** otherwise. All 12 runs trained without divergence, all 4 datagen cells passed their yield floors, and no selected checkpoint is degeneracy-flagged. Every per-arm read is a single run (seed 42) — preliminary by construction.

**Positive-only regime round** (`writing-style-positive-only-regime`, run 2026-07-17): 12 further LoRA organisms — the same 4 contexts × 3 learning rates — trained on 60-row positive-only mixes (each parent cell's 20 kept positives + 40 generic rows, negative panel removed), with `max_steps` 75 so optimizer steps match the parent schedule (≈20 data passes on the smaller mix vs 15). The regime comparison is made at each regime's band-selected verdict arm (band-matching as the matched-install control); a context is labeled dose-unmatched when either verdict arm sits out of band or the two ladder-selection rates differ by more than 0.10. The per-context regime-leakage verdict is D, the positive-only verdict arm's pooled non-source judged rate minus the contrastive arm's (n = 500 per arm, Newcombe 95%): Broader-leakage when the interval sits above zero, Narrower-leakage when below, Indistinguishable otherwise; the shared base panel cancels in this trained-vs-trained difference. Two named riders travel with the regime factor and are carried as scope caveats: the generic-row fraction rises 0.50 → 0.67, and data passes rise ≈15 → ≈20. All 12 positive-only runs trained without divergence; no selected checkpoint is degeneracy-flagged. (Scope note, acknowledging the verifier's conciseness advisories: this body carries fifteen result sections — the full install / dose / intrusion / leakage / projection / margin / parity / specificity-control read of the approved analysis — so several result blocks, some Takeaways bullets, and the total prose budget sit above the soft word caps. The dose-round per-arm pooled-rates figure remains a prose link from the dose-matched result block — an intermediate aggregate; its per-unit 10-cell companion is embedded there. The control round's six-context-profile and graded-distribution figures likewise remain prose links from the specificity-control block — companions to its two embedded views.)

**Dose-matched persona re-read round** (`persona-dose-matched-regime`, run 2026-07-17, eval-only — no retraining): the persona-context regime contrast is re-read at dose-matched checkpoints, re-selected as the ladder rung nearest the other regime's selection rate (argmin of the rate gap over the uploaded persona ladders, tolerance 0.10; `dose_arm_selection.json`). Down-matching the positive-only arm to the contrastive 0.60 was infeasible at 5-step checkpoint spacing (nearest ladder rate 0.49, gap 0.11 — the round's infeasibility clause fires and travels as a scope caveat), so the read is a two-point dose bracket: matched-high — the existing positive-only verdict arm (step 25, rate 0.81) against a new contrastive checkpoint up-matched to it (step 45, rate 0.86, gap 0.05; the verdict-bearing bracket) — and near-matched-low — a new positive-only checkpoint (3e-5 ladder, step 10, rate 0.49) against the existing contrastive verdict arm (step 25, rate 0.60), supporting only, its positive-only side under-dosed by 0.11. The leakage panel is re-run only at the two new checkpoints; the two existing arms' committed panel reads are reused.

**ICL-read amplifier specificity control round** (`icl-read-amplifier-specificity`, run 2026-07-18, eval-only — no new training): the six-context leakage panel is re-run on two reused unrelated-behavior organisms from the factory's impolite line — the installed persona-context impolite organism (checkpoint 30 of its 3e-5 learning-rate arm; impolite record 0.805) as the primary arm, and the zero-install production impolite adapter as an inert reference — scored with the same casual-register instrument against the committed base panel (0/100 in all six contexts). Four cells (both arms' source-persona and two-shot-prefix cells) are additionally re-judged under the impolite line's own rubric, giving three checks fixed in the plan: an adapter-engagement check (the installed arm's own-context impolite rate must reach 0.50, set against its 0.805 record — a miss reads as a dead or wrong adapter and blocks any verdict), a register-overlap flag (own-context casual delta at or above 0.30 — impoliteness expressed as informal register would contaminate the casual rubric), and a behind-block mechanism read (the two-shot cell's impolite rate, reported alongside any positive casual read there). One recipe caveat travels with the round: the installed impolite organism was trained at learning rate 3e-5 — impoliteness does not enter the dose band at 1e-5 — where the casual comparison organisms are 1e-5 arms, and the impolite-line mix construction may differ beyond the behavior itself; the dose match is on the install band (0.805 against the casual arms' 0.60–0.86), so comparisons against the casual organisms' two-shot reads carry this caveat while the delta-versus-base reads do not.

**Training:** every value below is copied from the run's reproducibility card (`epm:results` marker) and plan §0 at the run commit; the plan grounds each in the parent factory recipe.

| Parameter | Contrastive round | Positive-only round | Source |
|---|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | inherited | parent factory recipe (plan §0) |
| LoRA | r 32, α 64, rsLoRA, dropout 0.05, 7 projection modules | inherited | parent factory recipe (plan §0) |
| Learning rates | 1e-5, 3e-5, 1e-4 (one arm each) | inherited | parent impolite learning-rate ladder (plan §0) |
| Epochs / optimizer steps | 15 epochs = 75 steps, cosine schedule | `max_steps` 75 (≈20 data passes on 60 rows), cosine | reproducibility card; amendment plan §2 |
| Checkpoint cadence | every 5 steps | inherited | reproducibility card |
| Batch | 4 × grad-accum 4 (effective 16) | inherited | parent factory recipe (plan §0) |
| Max sequence length | 2048 | inherited | parent factory recipe, declared deviation (plan §0) |
| Training mix | 80 rows = 20 positives + 20 contrastive negatives + 40 generic-chat | 60 rows = the same 20 positives + 40 generic-chat, no negative panel | parent factory recipe (plan §0); amendment plan §2 |
| Datagen oversample | 2.5× (bare-assistant cell 12.0×) | n/a — parent pools reused verbatim | parent factory recipe (plan §0) |
| Dose band | judged rate 0.60–0.85 | inherited | parent factory recipe (plan §0) |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, reason-then-score, max_tokens 300, positive threshold 50 | inherited | project judge rule; judging rule 23 (plan §0) |
| Judge draws per completion | 5 on the verdict, base, and rubric-parity reads; 3 on the dose-ladder and leakage-panel reads | inherited (base Tier-2 + base panel arms reused, not re-judged) | run-commit code constants (`TIER1_JUDGE_DRAWS`/`TIER2_JUDGE_DRAWS`, `scripts/issue1090_run.py`; panel `n_draws=3`, `scripts/issue1434_worker.py`) |
| Eval sampling | temperature 1.0; ladder read 5 completions × 20 questions per checkpoint (n=100); verdict read 10 × 20 (n=200); leakage panel 100 per (organism, context) | inherited | parent instrument (plan §2) |
| Seed | 42 | inherited | project default |
| Dose-matched re-read arms (eval-only round) | `ws-pers-lr1e5` step 45 (ladder rate 0.86, target 0.81) | `ws-po-pers-lr3e5` step 10 (ladder rate 0.49, target 0.60 — outside the 0.10 tolerance) | round selection JSON (`dose_arm_selection.json`; argmin rate-gap rule, amendment plan v8) |
| Specificity-control arms (eval-only round) | installed impolite organism `adapters/issue1090_fu4/imp-pers-lr3e5/checkpoint-30` (lr 3e-5, impolite record 0.805) | inert reference `issue1090/c2-impolite-claude` (install delta 0) | round arm manifest (`control_arm_manifest.json`); impolite-line ladder + install JSONs |

**Evaluation:** The primary DV is the on-policy judged casual-register rate under the persona-vectors trait-expression rubric (arXiv 2507.21509), fetched verbatim and instantiated for casual writing style ([committed rubric + provenance sidecar](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/src/explore_persona_space/artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt)); a completion is positive when its mean graded score over its judge draws exceeds 50 (5 draws on the verdict, base, and rubric-parity reads; 3 on the dose-ladder and leakage-panel reads). Malformed or refused judge returns are dropped, never coerced (item drop fractions ≤ 2.5% per verdict arm, ≤ 9% on one panel cell); transport failures are retried and re-judged (≤ 23 draws per arm recovered). Verdict deltas use Newcombe 95% intervals on the trained−base rate; rates carry Wilson 95% intervals. Leakage is the trained−base judged rate over 6 read contexts (the four training contexts plus police-officer and philosopher personas), 100 completions per state. The continuous companion is a teacher-forced fixed-pool margin — mean length-normalized log-probability of a fixed 25-completion judge-kept casual pool minus a fixed 25-completion formal pool, trained − base in the source context — subject to a rate-tracking validation (Spearman vs the rate must be positive) before any use; it failed and is quarantined (Results). The non-judge companion is the casual-register persona-vector direction `r_B` (per-layer difference of mean response-averaged activations over judge-filtered contrastive rollouts: 5 positive/negative instruction pairs × 20 extraction questions, 10 rollouts each at temperature 1.0, keep positive > 50 / negative < 50): each organism's trained−base activation shift is projected onto `r_B`, and rank agreement with the judged deltas is tested against a label-shuffle null and a norm-matched random-direction band drawn from the honest covariance families (arm-centered within-class and negative-arm), with the max-over-28-layers selection inherited per null draw; uncertainty is a cluster bootstrap reported both ways (frozen headline layer, and selection-inherited per draw). The fixed mid-network companion read is layer 19 in the artifacts' 0-based layer indexing (`paper_companion_layer19` in `pv_validation.json`) — the plan's layer-20 read in 1-based counting. Per the standing both-mappings rule the projection is computed on prefix-end, context-end, and response-average (shared-text and own-text) arms; the verdict-carrying primary read is the shared-text response-average arm — the other arms are exploratory this round (no null battery). Instrument parity: the same verdict-read completions re-judged under the project's own casual-register rubric (the datagen keep-filter). The positive-only round reuses this instrument verbatim on the 12 new trained arms (base Tier-2 and base panel reads reused from the contrastive round). Two round-specific audits ground the regime verdicts: a script-intrusion recount of every pool they rest on — per-item judge means (mean of valid draws, threshold 50) joined with the CJK regex over the same completions, recomputing D and the positive-only install rates under zeroed-intrusion and excluded-intrusion conventions, with the join gated on exact agreement between recomputed and committed per-cell counts (`po_intrusion_audit.json`) — and the margin's quarantine re-check on the 12 positive-only cells, Spearman of per-arm margin delta against the ladder-selection rate (`po_margin_validation.json`). The dose-matched re-read round reuses the panel instrument verbatim on its two re-selected checkpoints (6 read contexts, 100 completions per state, 3 judge draws, temperature 1.0, seed 42, Newcombe 95% on D; content-dropped judge draws 21 and 6 of 1800 per arm — 7 and 2 items unscored — transport losses 0); because those checkpoints are picked by minimizing the gap to a rate target over the ladder rungs, the quoted intervals condition on that selection rule rather than on checkpoints fixed in advance. Its script-intrusion recount extends to both new pools (intruded rows 55/493 and 40/499 pooled non-source; the zeroed- and excluded-intrusion recounts flip neither bracket verdict). The specificity-control round reuses the panel instrument verbatim on its two impolite arms (6 read contexts, 100 completions per state, 3 judge draws, temperature 1.0, seed 42; 17 and 6 content-dropped judge draws (installed / inert) across 800 judged items each — 5 and 2 items unscored — transport losses 0); its four dual-rubric cells reuse the impolite line's Tier-2 rubric (same Sonnet judge, 300-token budget, threshold 50) under its own rubric-keyed cache partition, and its graded casual-score distributions keep spread in every cell (no ceiling saturation).

**Data extraction:** Positives are Claude-Sonnet-generated (third-party-LLM-written, the factory's standing data-realism caveat): the trait description "Writing in a casual, informal register." plus the 5 contrastive instruction pairs from the behavior registry, over 40 auto-generated neutral trait-eliciting questions (20 extraction / 20 eval, disjoint), instruct-and-strip — generation carries the casual-register instruction, the stored training row keeps only the cell's context. The judge keep-filter is the project casual-register rubric; realized yields 90–431 kept positive candidates per cell against a floor of 20. Contrastive negatives come from a 5-member panel including the default assistant (the bare cell uses the 4 non-default members; panel disjoint from the source persona), judge-filtered to not exhibit the behavior, 27–31 kept against a quota of 20, at ~1:1 positives-to-total-negatives in the mix. Generic-chat rows (40 of every 80-row mix) come from the factory's shared 400-row generic corpus — Claude-Sonnet-written answers to generic assistant questions under the default assistant system prompt (LLM-generated synthetic data, produced for the project's early leakage experiments and staged to the data repo as `issue906_inputs/generic_corpus.jsonl`; sha-verified against the staged pin at every datagen cell). The WildChat training context is a single real two-turn conversation drawn from the WildChat-1M corpus by an earlier battery builder — user turn verbatim, assistant turn truncated to 2,000 characters — frozen in-repo and bound by the parent factory as its conversational-prefix context (`wildchat_prefix_real545`); this run reuses that binding unchanged. The ICL prefix is a two-shot casual block written by this task's implementing agent (model-written rather than elicited; sha-pinned, no judge filter — an authored anchor, the same scope caveat as the parent's ICL column). The positive-only round's mixes were derived, not regenerated: each parent 80-row mix minus its 20 negative-panel rows (row-provenance filter against the cell's negative pool, parent order preserved, sha-verified against the parent pins), leaving the identical 20 positives + 40 generic rows per cell; a mix-integrity assert enforces the composition, so datagen provenance is identical across regimes.

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

All four base arms read 0 of 200 positive (graded means 16.9–21.4 of 100), so every install is a trained gain — the casual two-shot block alone elicits zero judged-casual base completions. Single seed; each per-arm read is one run. The figure title's "band entry 3 of 4" counts dose-ladder selection entries; fresh verdict reads land 2 of 4 in band.

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

Persona- and assistant-trained organisms generalize casual register broadly (5–6× the prefix-trained means); WildChat- and prefix-trained organisms are nearly context-bound. The casual two-shot context is a strong cross-organism amplifier: three of four organisms express casual register behind it at +0.97 to +1.00, the WildChat-trained one at +0.47, while the base model behind the same block expresses none. These are trained−base drifts under this contrastive mix; no behavior-specific causal claim. An unrelated-organism control of the two-shot column (final result section) returned no verdict, so its behavior-specificity is unresolved.


### Removing the negative panel leaves install strength essentially unchanged

What is plotted: judged casual-register rate for the twelve positive-only arms and the four reused base arms, grouped by training context; darker bars are fresh verdict reads (n = 200), lighter bars ladder-selection reads (n = 100); shading = the 0.60–0.85 band; error bars Wilson 95%.

![Bar chart of judged casual-register rate for twelve positive-only arms and four base arms across four training contexts, with the 0.60 to 0.85 band shaded and the verdict arms carrying error bars.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/32265eb529de5aa7ae2ade8502c57bd486e0b049/figures/issue_1434/po_install_grid.png)

> **Figure.** *The positive-only install pattern reproduces the contrastive one: three contexts install, the bare assistant falls short.* Judged rate per positive-only arm and base arm; verdict arms: persona 0.740, bare assistant 0.495, WildChat 0.629, two-shot prefix 0.995.

| Training context | Positive-only verdict arm | Fresh rate, positive-only | Fresh rate, contrastive | D install [95% CI] | Positive-only label |
|---|---|---|---|---|---|
| Persona (software engineer) | lr 1e-5, step 25 | 0.740 | 0.625 | +0.115 [+0.024, +0.204] | Installed |
| Bare assistant | lr 1e-5, step 25 | 0.495 | 0.550 | −0.055 [−0.151, +0.043] | Dose-responsive-but-short |
| WildChat prefix | lr 3e-5, step 5 | 0.629 (n scored 197) | 0.621 | +0.009 [−0.086, +0.104] | Installed |
| Two-shot ICL prefix | lr 1e-5, step 10 (closest approach) | 0.995 | 0.990 | +0.005 [−0.019, +0.031] | Installed |

All four per-context labels reproduce. Only the persona context installs detectably stronger without negatives; the other three contrasts straddle zero. Caveats: the contrastive arms were judged in the parent round a day earlier under the same judge pin, so the one non-null contrast carries residual instrument-drift risk; the WildChat positive-only label is convention-dependent under the script-intrusion recount (0.579 zeroed, 0.626 excluded); and the WildChat positive-only ladder overshoots at the parent learning rate (0.91 at 1e-5), making its verdict arm 3e-5 — the one verdict-arm change the regime forced.

### The dose ladders select in-band checkpoints in both regimes everywhere except the two-shot prefix, which never enters the band

What is plotted: Tier-1 judged casual-register rate versus optimizer step for each learning-rate rung, contrastive (solid) against positive-only (dashed), one panel per training context, with the 0.60–0.85 selection band shaded; the verdict-arm selections and the regime table's dose-match column read off these ladders.

![Four-panel line plot of Tier-1 judged rate over optimizer steps, contrastive solid versus positive-only dashed per learning rate, with the selection band shaded.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/32265eb529de5aa7ae2ade8502c57bd486e0b049/figures/issue_1434/po_ladder_overlays.png)

> **Figure.** *Dose ladders overlap across regimes; no two-shot-prefix rung ever enters the band.* Selected rates, positive-only vs contrastive: persona 0.81 vs 0.60, bare assistant 0.60 vs 0.61, WildChat 0.65 vs 0.70, two-shot prefix 0.98 vs 0.97 — the last jumping from at most 0.29 to at least 0.98 between steps 5 and 10.

The overlays ground the dose-match column: bare-assistant and WildChat selections land in-band in both regimes, the persona positive-only arm selects 0.21 above its contrastive twin, and the two-shot prefix crosses from below 0.3 to near 1.0 within one five-step interval in both regimes — no band-matched checkpoint exists there.

### Dropping the contrastive negatives broadens leakage from the persona and assistant contexts, not from the prefix contexts

What is plotted: pooled non-source leakage (trained−base judged rate over each organism's five non-source read contexts, n = 500 per arm), contrastive vs positive-only, and beneath it the regime contrast D = positive-only − contrastive with Newcombe 95% intervals; dose-unmatched contexts tagged.

![Grouped bars of pooled non-source leakage, contrastive versus positive-only, per training context, with regime-contrast points and intervals below.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/32265eb529de5aa7ae2ade8502c57bd486e0b049/figures/issue_1434/po_regime_hero.png)

> **Figure.** *Broader leakage in the persona and assistant contexts; indistinguishable behind the WildChat prefix; narrower behind the two-shot prefix.* Pooled D per context: persona +0.285, bare assistant +0.132, WildChat +0.014, two-shot prefix −0.080; the shared base panel cancels in D.

| Training context | Contrastive pooled Δ | Positive-only pooled Δ | D [95% CI] | Verdict | Dose match |
|---|---|---|---|---|---|
| Persona (software engineer) | +0.477 | +0.762 | +0.285 [+0.226, +0.342] | Broader-leakage | unmatched (selection 0.81 vs 0.60) |
| Bare assistant | +0.416 | +0.548 | +0.132 [+0.070, +0.193] | Broader-leakage | matched (0.60 vs 0.61) |
| WildChat prefix | +0.092 | +0.106 | +0.014 [−0.024, +0.051] | Indistinguishable | matched (0.65 vs 0.70) |
| Two-shot ICL prefix | +0.082 | +0.002 | −0.080 [−0.108, −0.057] | Narrower-leakage | unmatched (both arms above band, 0.98 vs 0.97) |

The negative panel buys containment exactly where the parent round found leakage broad, and nothing where leakage was already context-bound: the WildChat verdict is an equivalence bound (any containment differential there is at most about 0.05), and the two-shot-prefix organism leaks less positive-only (1 of 500 non-source completions). The persona contrast here is dose-unmatched (its positive-only arm selects 0.21 higher); the dose-matched re-read below shows it survives. Any D is attributable to the rider bundle — absent negatives, generic fraction 0.67, ≈20 data passes — though the extra generic rows plausibly bias against broader leakage: the two positive verdicts are conservative, the prefix reversal rider-ambiguous. All four verdicts survive both script-intrusion recounts; the four intervals are uncorrected.

### The regime contrast localizes to persona-type read contexts; the two-shot read context is at ceiling in both regimes

What is plotted: the 20 per-cell regime contrasts behind the pooled verdicts — D per (training context, read context) cell with Newcombe 95% intervals, n = 100 per arm per cell, each cell labeled.

![Forest plot of twenty per-cell regime contrasts with confidence intervals, ordered by training context and read context; persona- and assistant-trained cells sit positive and two ICL-prefix-trained cells sit negative.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/32265eb529de5aa7ae2ade8502c57bd486e0b049/figures/issue_1434/po_regime_cells.png)

> **Figure.** *Eight of the twenty cells exclude zero: six positive (persona- and assistant-trained organisms, largest +0.61), two negative (the two-shot-prefix organism's reversal).* Roughly one false-positive cell is expected among twenty uncorrected intervals.

The extra positive-only leakage concentrates in persona-type read contexts (police officer +0.61 and +0.37, philosopher, the default assistant); the two negative cells reflect the two-shot-prefix organism reading near zero everywhere positive-only while its contrastive twin leaked into WildChat (0.30) and persona (0.08) contexts. The two-shot read context itself is uninformative: persona- and assistant-trained organisms sit at 0.97–1.00 there in both regimes, so those cells are ceiling-censored; the pooled verdicts are read jointly with this grid. An earlier factory-line regime comparison (40-row fraction-preserving positive-only mixes, fixed epochs) found no regime effect on leakage for three other behaviors — not directly comparable given the mix and schedule differences.

### The persona regime contrast survives dose-matching and widens at the matched-high dose

What is plotted: the persona-context regime contrast D (positive-only minus contrastive pooled non-source judged rate, Newcombe 95%) at four checkpoint pairings — the parent verdict checkpoints, the two re-selected dose brackets (nearest ladder rung to the other regime's selection rate; the intervals condition on this selection), and the dose-matched bare-assistant contrast as reference.

![Pooled regime-contrast bars: unmatched, matched-high, near-matched-low, and bare-assistant reference.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a76e5839490d893f887aaf6bfc238af373ae11a7/figures/issue_1434/dose_hero.png)

> **Figure.** *Both bracket contrasts sit above zero — install dose does not explain the persona Broader-leakage verdict.* Matched-high +0.446 (existing positive-only verdict arm vs the up-matched contrastive checkpoint), near-matched-low +0.151, parent unmatched read +0.285, bare-assistant reference +0.132.

![Ten per-cell regime contrasts with intervals at both dose brackets.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a76e5839490d893f887aaf6bfc238af373ae11a7/figures/issue_1434/dose_cells.png)

> **Figure.** *Every matched-high cell sits above zero.* The 10 per-cell contrasts behind the pooled bars: per-read-context D with Newcombe 95% intervals at both dose brackets (darker points matched-high, lighter near-matched-low). The near-matched-low philosopher and two-shot ICL cells cross zero.

| Pairing | Positive-only arm (ladder rate) | Contrastive arm (ladder rate) | D [95% CI] |
|---|---|---|---|
| Unmatched (parent verdict arms) | step 25 (0.81) | step 25 (0.60) | +0.285 [+0.226, +0.342] |
| Matched-high — verdict-bearing | step 25 (0.81) | step 45 (0.86) | +0.446 [+0.388, +0.498] |
| Near-matched-low — supporting only | step 10, 3e-5 ladder (0.49) | step 25 (0.60) | +0.151 [+0.089, +0.211] |

Down-matching the positive-only arm proved infeasible at 5-step checkpoint spacing (Methodology), so the verdict binds to the matched-high bracket, up-matched within 0.05. That contrast exceeds the unmatched read because contrastive leakage falls as dose rises (pooled 0.478 → 0.318 while the source-context rate climbs 0.61 → 0.91): containment deepens as the negatives train longer.

The under-dosed low bracket biases D downward and still sits above zero, though its philosopher and two-shot read cells are indistinguishable from the null given the variance ([per-arm pooled rates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a76e5839490d893f887aaf6bfc238af373ae11a7/figures/issue_1434/dose_pooled_bars.png)). The positive-only low arm's panel source-context rate is 0.596 (n = 99, ladder selection 0.49). Neither script-intrusion recount flips either bracket.

### The persona-vector projection tracks neither install nor leakage on its primary read

What is plotted: signed Spearman ρ between per-cell projections (trained−base shift onto the casual-register direction, shared-text arm) and judged deltas, per layer; left = the 24-cell leakage grid, right = the 12-cell install grid. Dashed / dotted = random-direction and label-shuffle nulls (p97.5, selection-inherited); star = selected layer; square = layer-19 companion.

![Per-layer signed Spearman rho between projection and judged delta for the leakage and install grids, with random-direction and shuffle null bands.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/rb_validity_layers.png)

> **Figure.** *Observed ρ never clears the honest random-direction band on either grid.* Leakage +0.244 (layer 2) vs band 0.477, shuffle 0.529; install −0.823 (layer 0) vs band 0.907–0.914, shuffle 0.785; layer-19 companion +0.087 / +0.231. The positive-only and combined grids (table rows; not in the figure) also clear no random-direction band.

| Grid | Headline signed ρ (max-abs layer) | Cluster-bootstrap 95% CI, frozen layer | Cluster-bootstrap 95% CI, selection-inherited | Random-direction band p97.5 (within-class / negative-arm) | Shuffle p97.5 | Label |
|---|---|---|---|---|---|---|
| Leakage (24 cells, 4 organism clusters) | +0.244 (layer 2) | [−0.422, +0.541] | [−0.491, +0.521] | 0.477 / 0.477 | 0.529 | Untracked |
| Install (12 cells) | −0.823 (layer 0) | [−0.949, −0.467] | [−0.957, +0.866] | 0.914 / 0.907 | 0.785 | Untracked |
| Positive-only leakage (24 cells, 4 organism clusters) | +0.658 (layer 0) | [+0.366, +0.694] | [+0.366, +0.694] | 0.658 / 0.658 (ties the observed — rank discreteness) | 0.525 | Untracked |
| Positive-only install (12 cells) | +0.301 (layer 27) | [−0.346, +0.759] | [−0.814, +0.883] | 0.837 / 0.764 | 0.771 | Untracked |
| Combined leakage (48 cells, 8 clusters; exploratory — regime is a main effect across cells) | +0.515 (layer 4) | [+0.124, +0.675] | [+0.206, +0.675] | 0.591 / 0.591 | 0.411 | Untracked |
| Combined install (24 cells; exploratory) | −0.496 (layer 0) | [−0.792, −0.058] | [−0.792, +0.632] | 0.855 / 0.722 | 0.574 | Untracked |

On the leakage grid the interval spans zero at 4 organism clusters — this Untracked is substantially a power statement. The install-grid anti-correlation clears the shuffle band, but layer 0 was selected by maximizing absolute ρ over 28 layers: the selection-inherited bootstrap spans zero, and layers 0–1 carry near-zero signal (cosine at most 0.11) while mid/late layers move along the direction without rank-tracking install. The install-grid null band sits 0.09 below the ρ ceiling at n = 12 — near-uninformative by construction; at this n the band is too wide for a non-rejection to carry evidential weight.

### Per-cell projections and the exploratory mapping arms undercut the latent-meter reading

What is plotted: the per-cell data behind the verdicts — the leakage grid (24 points), the install grid at its selected layer (12 points), and the exploratory context-end arm (12 points); points labeled by organism, colored by training context.

![Three scatter panels of projection versus judged delta: the leakage grid with only four distinct projection levels, the install grid at layer 0 with a downward rank trend, and the exploratory context-end arm at layer 27 with an upward trend.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/rb_validity_scatter.png)

> **Figure.** *The leakage grid has only 4 distinct projection levels; no exploratory arm clears the random-direction band.* Projection vs judged delta: leakage ρ +0.24 (layer 2); install, shared-text arm ρ −0.82 (layer 0); exploratory context-end arm ρ +0.84 (layer 27). The third panel's annotation predates the later six-arm null battery.

The six exploratory reads (prefix-end / context-end / own-text arms × both grids) are sign-inconsistent: install reads −0.73 prefix-end but +0.84 / +0.87 context-end / own-text; all three leakage arms are negative (−0.41 to −0.48). The null battery, now run on all six arms, resolves the earlier gap: none clears the arm-centered random-direction band. The two positive install reads clear only the label-shuffle band (own-text: both bootstrap intervals positive, still under 0.94); the leakage 0.477 band exactly ties the prefix/context |ρ| — a four-level rank-discreteness tie, not clearance. Fold sweeps and length-partialling leave both Untracked labels unchanged.

### The teacher-forced fixed-pool margin anti-correlates with the judged rate and is quarantined

What is plotted: per-organism teacher-forced fixed-pool margin delta (trained − base, source context, nats) against install delta; 12 labeled points, colored by training context.

![Scatter of 12 organisms with teacher-forced fixed-pool margin delta on the x axis and install delta on the y axis. The strongest installs sit at the most negative margins, giving a downward trend.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/margin_vs_rate.png)

> **Figure.** *The margin fails its rate-tracking validation (Spearman ρ = −0.72, p = 0.009, n = 12).* Margin = mean length-normalized log-probability of the fixed 25-completion casual pool minus the fixed 25-completion formal pool, trained − base, source context.

The margin is negative for 10 of 12 organisms (−0.77 to +0.03 nats), and the strongest installs have the most negative margins — consistent with likelihood displacement toward the model's own casual phrasing rather than a casual-register propensity read. It fails the ρ > 0 validation gate (Spearman −0.72 against the install deltas, n = 12) and is quarantined; the judged rate stands alone as the validated DV. The positive-only cells repeat the failure: margin deltas +0.00 to +0.42 nats still anti-rank the selection-rate install read (Spearman −0.46, n = 12); the quarantine stands across regimes. Both rounds' margins were recomputed from the per-arm JSONs (the aggregate field was empty).

### The persona-vectors rubric and the project casual-register rubric agree on the same completions

What is plotted: per-arm judged rate under the persona-vectors rubric (x) against the project casual-register rubric (y) on identical verdict-read completions; 8 points = 4 trained + 4 base arms; dashed diagonal = equality.

![Scatter of eight arms comparing judged rates under two rubrics, with trained verdict arms as circles and base arms as squares. All points sit on or very near the identity diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434/parity_scatter.png)

> **Figure.** *Maximum rubric gap 0.033 across the 8 arms.* Trained verdict arms: persona 0.625 / 0.620, bare assistant 0.550 / 0.575, WildChat 0.621 / 0.588, two-shot prefix 0.990 / 0.995; base arms 0.000–0.005.

The instrument change from the project rubric is immaterial for this behavior, so writing-style-versus-impolite placements are instrument-safe. Two caveats: checkpoint selection ran under the persona-vectors rubric, and both rubrics share the same Sonnet judge, so parity checks rubric robustness without corroborating the headline independently. With both non-judge DVs failing to corroborate, corroboration for the headline remains judge-instrument-only.

### The unrelated-organism specificity control returns no verdict, leaving the two-shot column's behavior-specificity unresolved

What is plotted: judged rates under the casual and impolite rubrics for the two reused impolite arms on their source-persona and two-shot-prefix cells, then each arm's casual rate across all six read contexts; n = 100 per cell. Both arms and every threshold were fixed in the plan — no checkpoint or context selection.

![Casual and impolite judged rates for both control arms on two cells.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ae21f67a82bf162682933277a9ad5e129f1cec37/figures/issue_1434/ctrl_dual_rubric.png)

> **Figure.** *The adapter-engagement check misses its 0.50 threshold at 0.48.* Casual and impolite judged rates per arm and cell, n = 100, Wilson 95%; dashed line = the engagement threshold. The installed organism was trained at learning rate 3e-5 (the casual comparison arms: 1e-5), dose matched on the install band (0.805 vs 0.60–0.86).

![Judged casual rate per read context for both impolite control arms.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ae21f67a82bf162682933277a9ad5e129f1cec37/figures/issue_1434/ctrl_arm_bars.png)

> **Figure.** *The installed impolite organism scores casual-positive in five of six contexts; the inert adapter reads 0.16 behind the block, at most 0.04 elsewhere.* Judged casual rate per read context, n = 100 per cell, Wilson 95%; committed base panel 0/100 everywhere; [six-context profile vs the casual organisms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ae21f67a82bf162682933277a9ad5e129f1cec37/figures/issue_1434/ctrl_hero.png). Graded scores keep spread in every cell ([per-cell histograms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ae21f67a82bf162682933277a9ad5e129f1cec37/figures/issue_1434/ctrl_graded_dist.png)).

| Read (installed impolite arm unless noted) | Value | 95% interval |
|---|---|---|
| Adapter-engagement check: own-context impolite rate (threshold 0.50; source record 0.805) | 0.48 | 0.385–0.577 (Wilson) |
| Exploratory: casual delta behind the two-shot block (casual organisms' band +0.47 to +1.00) | +0.69 | 0.587–0.772 (Newcombe) |
| Register-overlap flag: own-context casual delta (flag threshold 0.30 — fired) | +0.66 | 0.558–0.749 (Newcombe) |
| Mechanism read: impolite rate behind the two-shot block | 0.01 | 0.002–0.054 (Wilson) |
| Inert reference adapter: casual delta behind the two-shot block | +0.16 | 0.090–0.244 (Newcombe) |

The engagement check missed by two items, its interval spanning the threshold; the criterion was fixed before the run, so no verdict is read. As exploratory characterization the reads point both ways: behind the casual two-shot block the arm emits almost no impolite text yet scores 0.69 casual — consistent with generic in-context restoration — but its own-context outputs already score 0.66 casual with no block, so the casual rubric plausibly reads impolite-informal register as casual across contexts. The column's behavior-specificity stays unresolved; the inert adapter's 0.16 behind the block suggests a small adapter-presence component.

---
**Repro:** ~14 GPU-h realized on one GCP 8× A100-80 flex-start instance (`eps-issue-1434`; 12 LoRA runs + eval battery), judging + analysis off-pod on the VM. Code at run commit `f9f1002797`: [`scripts/issue1434_worker.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_worker.py), [`scripts/issue1434_cells.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_cells.py), [`scripts/issue1434_pv.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_pv.py) (projection + nulls), [`scripts/issue1434_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/scripts/issue1434_figures.py). Aggregates at commit `30332ac26a` (branch issue-1434): [`eval_results/issue_1434/`](https://github.com/superkaiba/explore-persona-space/tree/30332ac26a031a5bc5fdc2d52025e4f806fc29aa/eval_results/issue_1434) (`i1434_ladders.json`, `selection.json`, `pv_validation.json`, bootstrap + null-matrix JSONs, `cell_manifest_i1434.json`). The null-battery extension to all six mapping arms (per-arm null matrices + signed cluster bootstraps; `pv_validation.json` updated in place) landed at commits `a0b5f1fb96` (code) and `9eadc3e758` (aggregates, branch issue-1434). Figures on main: [`figures/issue_1434/`](https://github.com/superkaiba/explore-persona-space/tree/169d4c83beac7e65593bebc8e394e6a8b7d70262/figures/issue_1434). Adapters (per-checkpoint LoRA under `issue1434/<run>/checkpoint-<step>`): [HF model repo @ 75b1d5ab, issue1434](https://huggingface.co/superkaiba1/explore-persona-space/tree/75b1d5abc5eb14228184dfdafb36e4e98d41f971/issue1434). Training pools + mixes, raw completions, judge outputs, margin JSONs, analysis tensors: [HF data repo @ 72a5c3a8, issue1434_writingstyle](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/72a5c3a832fa5e2fd67ee9d6b33f1604ab664291/issue1434_writingstyle). Judge rubric: [`pv_writing_style_trait_score_v1.txt` @ f9f10027](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/src/explore_persona_space/artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt). WandB: project `issue1434`, entity `thomasjiralerspong`, 12 runs (`issue1434_ws-<context>-lr<rate>_seed42`). Positive-only round (`writing-style-positive-only-regime`): ~11 GPU-h realized on one GCP 8× A100-80 flex-start instance (~1.4 h wall, 2026-07-17 13:36–14:58 UTC), judging + analysis off-pod. Round aggregates on main: [`eval_results/issue_1434/writing-style-positive-only-regime/`](https://github.com/superkaiba/explore-persona-space/tree/ea2731fffc022da92987097f257e0ee2cbb41227/eval_results/issue_1434/writing-style-positive-only-regime) (`regime_contrast.json`, `selection_po.json`, `i1434po_ladders.json`, `pv_validation.json` with the positive-only + combined grids, `cell_manifest_i1434po.json`, `po_intrusion_audit.json`, `po_margin_validation.json`); full null-matrix + bootstrap JSONs and the audit script [`scripts/issue1434_po_intrusion_audit.py`](https://github.com/superkaiba/explore-persona-space/blob/3114e0226b106feefec2bac277206b1632761159/scripts/issue1434_po_intrusion_audit.py) at branch commit `3114e0226b`. Positive-only figures on main: [`figures/issue_1434/po_*.png`](https://github.com/superkaiba/explore-persona-space/tree/32265eb529de5aa7ae2ade8502c57bd486e0b049/figures/issue_1434). Positive-only adapters (per-checkpoint LoRA under `issue1434/ws-po-<context>-<lr>/checkpoint-<step>`): [HF model repo @ 5b1ed6cb, issue1434](https://huggingface.co/superkaiba1/explore-persona-space/tree/5b1ed6cbd385c752e2be20139f80d405fad77898/issue1434). Positive-only mixes, raw completions, judge outputs, margin JSONs: [HF data repo @ 8d95d4b1, issue1434_writingstyle](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8d95d4b11be328b5256735e8eca24cd4d6cc221a/issue1434_writingstyle). WandB: same project, 12 further runs (`issue1434_ws-po-<context>-lr<rate>_seed42`). Dose-matched re-read round (`persona-dose-matched-regime`): eval-only, ~0.7 GPU-h realized on one GCP g2-standard-4 (1× L4) flex-start instance (2026-07-17, ~43 min), judging + analysis off-pod. Code at run commit `a145ef3058`: [`scripts/issue1434_worker.py` (dose phases)](https://github.com/superkaiba/explore-persona-space/blob/a145ef3058b26921bacd20b67caeae61ded56cbc/scripts/issue1434_worker.py), [`scripts/issue1434_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/a145ef3058b26921bacd20b67caeae61ded56cbc/scripts/issue1434_figures.py). Round aggregates + figures at commit `a76e583949` (branch issue-1434-dose): [`eval_results/issue_1434/persona-dose-matched-regime/`](https://github.com/superkaiba/explore-persona-space/tree/a76e5839490d893f887aaf6bfc238af373ae11a7/eval_results/issue_1434/persona-dose-matched-regime) (`dose_arm_selection.json`, `i1434dose_panel.json`, `regime_contrast_dose_matched.json`), [`figures/issue_1434/dose_*.png`](https://github.com/superkaiba/explore-persona-space/tree/a76e5839490d893f887aaf6bfc238af373ae11a7/figures/issue_1434). Round panel raw completions + judge outputs: [HF data repo @ 2aa2a6b0, issue1434_writingstyle/dose](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2aa2a6b07aa9ebd71330a878fd9e22f499e8ed29/issue1434_writingstyle/dose) (judge dir upload-verified 2026-07-17; panel completions uploaded at fold time). Specificity-control round (`icl-read-amplifier-specificity`): eval-only, ~0.6 GPU-h realized on one GCP g2-standard-4 (1× L4) flex-start instance (`eps-issue-1434`, run 2026-07-18), judging + analysis off-pod; the worker raised after persisting and uploading on the engagement-check miss (the designed no-verdict path, not a crash). Code at run commit `f69b4ba077`: [`scripts/issue1434_worker.py` (control phases)](https://github.com/superkaiba/explore-persona-space/blob/f69b4ba077fb67f17c30aee3d22aba937f08541c/scripts/issue1434_worker.py), [`scripts/issue1434_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/f69b4ba077fb67f17c30aee3d22aba937f08541c/scripts/issue1434_figures.py). Round aggregates + figures at commit `ae21f67a82` (branch issue-1434-ctrl): [`eval_results/issue_1434/icl-read-amplifier-specificity/`](https://github.com/superkaiba/explore-persona-space/tree/ae21f67a82bf162682933277a9ad5e129f1cec37/eval_results/issue_1434/icl-read-amplifier-specificity) (`control_arm_manifest.json`, `control_panel.json`, `icl_specificity.json` — verdict cell, gates, mechanism reads, drop report), [`figures/issue_1434/ctrl_*.png`](https://github.com/superkaiba/explore-persona-space/tree/ae21f67a82bf162682933277a9ad5e129f1cec37/figures/issue_1434). Round panel raw completions: [HF data repo @ 28631a42, issue1434_writingstyle/raw_completions/ctrl/panel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/28631a4266562b97130bc56f0a84dcb6cf056dd8/issue1434_writingstyle/raw_completions/ctrl/panel) (12 per-context completion files + `control_panel_arms.json`); judge outputs: [issue1434_writingstyle/ctrl/judge](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/28631a4266562b97130bc56f0a84dcb6cf056dd8/issue1434_writingstyle/ctrl/judge) (upload-verified 2026-07-18, 1,616 files). Run log: `logs/issue1434_ctrl_judge.log`.

- Reused the WildChat conversational-prefix context (`wildchat_prefix_real545`) from [#1090](https://eps.superkaiba.com/tasks/1090): a single real two-turn WildChat-1M prefix frozen in-repo at [`eval_results/issue_545/batteries/wildchat_prefix.json` @ f9f10027](https://github.com/superkaiba/explore-persona-space/blob/f9f1002797fa0b097c26e6792236ee4b377a40b4/eval_results/issue_545/batteries/wildchat_prefix.json) — fit: the parent factory's own conversational training context, unchanged, so context placement stays comparable across factory behaviors.
- Reused the shared generic-chat corpus from [#906](https://eps.superkaiba.com/tasks/906) via the parent factory: [`issue906_inputs/generic_corpus.jsonl` @ 1e0ad120](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/1e0ad12046160d11aede97b8663570a6d6a24348/issue906_inputs/generic_corpus.jsonl) (staged sha256 `ba036fb396ea…`; provenance pinned per cell in `mix/mix_meta.json`) — fit: the factory's standard generic interleave, 40 of every 80-row mix, identical across sibling cells.
- Reused control-round adapters from [#1090](https://eps.superkaiba.com/tasks/1090): the installed impolite organism [`adapters/issue1090_fu4/imp-pers-lr3e5/checkpoint-30`](https://huggingface.co/superkaiba1/explore-persona-space/tree/9c36798a40189673296de19dd36879f015ecfe59/adapters/issue1090_fu4/imp-pers-lr3e5/checkpoint-30) — fit: identical LoRA recipe (r 32, α 64, rsLoRA, dropout 0.05) verified against `adapter_config.json` at fetch time, install band 0.805, trained at learning rate 3e-5 (recipe caveat carried in Methodology) — and the zero-install impolite adapter `issue1090/c2-impolite-claude` ([HF model repo @ 9c36798a, issue1090 tree](https://huggingface.co/superkaiba1/explore-persona-space/tree/9c36798a40189673296de19dd36879f015ecfe59/issue1090)) — fit: the labeled inert reference (install delta 0), anchoring the adapter-presence-alone alternative.

**Context:** created 2026-07-16; run 2026-07-16 → 2026-07-17; results landed 2026-07-17. Lineage: [#1090](https://eps.superkaiba.com/tasks/1090) — the impolite/sycophancy organism-factory parent; this sibling task carries the writing-style half of that split. Same-issue follow-up round `writing-style-positive-only-regime` (user-chat, `followup-manual`) run + folded 2026-07-17. Same-issue follow-up round `persona-dose-matched-regime` (proposer-9b-cheap, cheap-band round 1 of 2, `followup-auto`) run 2026-07-17, folded 2026-07-18. Same-issue follow-up round `icl-read-amplifier-specificity` (proposer-9b-cheap, cheap-band round 2 of 2, `followup-auto`) run + folded 2026-07-18; its falsification directive, from the round scope: if the impolite organism also expresses casual register behind the block, the ICL-read column measures generic LoRA-induced ICL-compliance restoration, not casual leakage. Originating prompt, verbatim:

> do your best to get writing style and sycophancy to work / can we run it in parallel and be faster? (PM chat 2026-07-16; writing_style split into its own parallel task per the parallelism ask; sibling of #1090 impolite/sycophancy factory line)

Round directive, verbatim (expanded in the `epm:followup-scope` marker to the 4-context × 3-learning-rate positive-only mirror with the 0.50 → 0.67 generic-fraction rider named):

> "do we have a positive only arm" -> "run a positive only arm if we dont have one" (PM chat 2026-07-17)



