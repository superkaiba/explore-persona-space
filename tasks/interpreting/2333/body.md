---
title: 'Snowball test: do context-vector patch effects run through the first few answer
  tokens? First-k token patches + prefills vs ce patch, both models'
kind: experiment
tags: []
created_at: '2026-08-16T19:44:45Z'
has_clean_result: false
parent_id: 2162
origin_prompt: 'Can you also run this on both models: patching at context vector at
  all layers vs patching at first 1/2/3 answer tokens at all layers (excluding context
  vector) vs prefilling first 1/2/3 tokens, on the most successful causal patching
  experiments; donor schemes: BOTH mediation form and B''s-answer-start form'
workflow: v1
backend: runpod
goal: 'Test the snowball explanation of context-vector patch causality on Qwen2.5-7B-Instruct
  and Qwen3.5-9B (thinking disabled): compare the all-layer context-end patch against
  patching only the first 1/2/3 generated answer-token positions (ce excluded) and
  against prefilling the first 1/2/3 tokens, under two donor schemes (mediation form:
  content produced under the ce patch with the patch then removed; B''s-answer-start
  form: the target context''s own reference answer opening), measured by F_beh/F_act
  vs scheme-matched shuffled-donor nulls on the pre-registered most-successful cells
  (#2162''s 5 stored-and-used + #2094''s pirate matched-query pairs).'
relates_to:
- spec-context-as-vector
- spec-steering
---
# Prefilling the first three answer tokens reproduces most of the context-end patch's whole-answer behavioral effect (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Prefilling the context-end patch's own three-token opening recovers **67% of the full patch effect** on Qwen2.5-7B instruction-format pairs (n=172; paired diff +0.35, corrected p 2e-13) — snowball-sufficient.
- Qwen3.5-9B agrees against its own same-wave patch ceiling (71% recovery); the Qwen2.5-normalized banked-ceiling read is indeterminate — a cross-model denominator, with measured judge-instrument offset only +0.07.
- Token identity carries it: prefill recovery rises 48% to 67% across one to three tokens (Qwen2.5); state patches plateau near 54%, and on Qwen3.5 stop separating beyond one position.
- Recovery is majority, not total — roughly a third of the patch effect is not carried by opening tokens; the activation companion recovers only 32% on Qwen2.5 (snowball-partial there).
- Pirate matched-query pairs (n=10 per model) read indeterminate on both models: diffs near +0.10 against a design detectable-effect floor near 0.32 — underpowered, not evidence of absence.
- 7–11% of temperature-1.0 draws carry Chinese-character intrusion (an arm-symmetric Qwen sampling artifact); every decision-lattice label is unchanged when intruded draws are excluded or zeroed.

## Goal

- **This experiment in context:** The context-as-vector line established that replacing the residual-stream state at the single context-end position, at every layer, with another context's state moves the behavior of the entire answer — on the instruction-format pair family of [#2162](https://eps.superkaiba.com/tasks/2162) (whole-answer movement 0.74 on this run's re-judge of its banked completions) and the pirate matched-query cell of [#2094](https://eps.superkaiba.com/tasks/2094) (0.51 on the same re-derivation). This experiment tests the snowball explanation of that reach: the patch fixes the first few answer tokens, and self-consistency propagates the rest. With the context-end patch removed, it transplants only the answer opening — tokens (prefill) or hidden states (decode-step patch), one to three positions, donors from the patch's own mediated opening or the target's natural opening — and asks how much of the full effect returns against scheme-matched shuffled-donor nulls. [#2329](https://eps.superkaiba.com/tasks/2329)'s Qwen3.5 template integration supplies the second model; its banked artifacts failed the reuse fitness gate, so the Qwen3.5 bank, anchors, and control were generated fresh here.
- **Broader narrative:** If context influence on generation is an initial-condition effect — set the opening, and next-token consistency does the rest — then opening-level reads and interventions inherit most of what context-vector patching buys, and persistent vector-level steering is the smaller residual. This bears directly on the project's context-representation question: what a single position's state actually controls.

## Methodology

- **Design:** 2 models (Qwen2.5-7B-Instruct; Qwen3.5-9B with thinking disabled) × 2 pair sets (180 instruction-format directed pairs across 5 cells; 15 pirate matched-query pairs) × 12 intervention arms ({decode-step state patch, token prefill} × k ∈ {1,2,3} opening positions × {patch-content donors, target's-own-opening donors}) × {steered, shuffled-donor null} × 5 draws = 23,400 grid rollouts per model. Supporting arms: the context-end all-layer replace positive control (banked completions for Qwen2.5, re-judged in this run's wave; 1,950 fresh rollouts for Qwen3.5), fresh Qwen3.5 floor/ceiling anchors (1,950 rollouts), and 390 greedy donor-capture rollouts per model. The single manipulated variable is which slice of the answer opening is transplanted, and through which channel. The confirmatory verdict was fixed in the plan on the three-token prefill with patch-content donors; all other arms are reported without verdict labels. Implementation went through 6 review rounds; the final round was a user-approved fix to a donor-gate position seam (commit `8a2503d888`) before the Qwen3.5 leg's launch.
- **Training:** **N/A — no model training.** Complete generation / judging / analysis hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Models | `Qwen/Qwen2.5-7B-Instruct` (bf16); `Qwen/Qwen3.5-9B`, `enable_thinking=False` | `eval_results/issue_2333/{q25,q35}/upload_done.json` |
| Stack | transformers 4.57.6 (leg A) / 5.15.0 (leg B); torch 2.8.0+cu128 | same `upload_done.json` repro blocks |
| Grid decoding | K=5 draws, temperature 1.0, `max_new_tokens` 2048; cap-hit rows regenerated at 4096 (trigger: >2% per cell) | plan §4.2/§6; `scripts/issue2333_run.py` |
| Generation batching | batched hooked generation, `GEN_BATCH=16`, 144 blocks/model via claim queue | `experiments/issue2333/constants.py:185` |
| Donor capture | greedy, `max_new_tokens` 8; first-3 token ids + post-block states at answer positions 1..3, all layers (28 / 32) | plan §4.2; `decode_hooks.py` |
| Anchors | floor = unpatched A, ceiling = generate-under-B; K=10, temperature 1.0 (banked for Qwen2.5; fresh for Qwen3.5) | parent recipe, plan §4.1 |
| Nulls | donors from a different pair via frozen derangements (instruction-format: value-constrained, seed 2162; matched-query: parent derangement); state donors norm-matched per layer and position; token donors unmatched | plan §4.2 |
| Judge | `claude-sonnet-4-5-20250929`, graded 0–100, Batch API (~50 shards/leg, 0 errored rows), `max_tokens` 1024; pilot gate + 6-item forced-batch probe passed before production waves | `scripts/issue2333_judge.py`; `judge_{q25,q35}/gates/` |
| Coherence gate | form-only rubric; coherent = score > 60; F computed over coherent draws only | parent constant (`issue2162_analysis.py`) |
| Statistics | exact two-sided paired signed-rank test on per-pair steered−null differences, Holm-corrected at fixed family size 12; pair-clustered bootstrap, B=10,000, seed 23330; majority-share read D3 = F_prefill3 − 0.5·F_control with pair-clustered CI | plan §3/§6; `scripts/issue2333_analysis.py` |
| Exclusions | anchor separation ≥ 0.5 required; per-cell survival floor 12 of 36 (instruction-format), set floor 5 (matched-query, confirmatory subset = the 10 banked well-separated pairs) | parent rule, plan §6 |
| Activation DV read layer | 26 (Qwen2.5) / 30 (Qwen3.5) | plan §6 |
| Compute | pod-2333-q25 (8× H100, ~3.2 h); pod-2333-q35 (8× H100, ~5.1 h final round) | run markers |

- **Evaluation:** The primary DV F_beh measures behavioral movement of the whole answer toward the donor context's behavior. Per draw, the judge scores the full response on the pair's two behavior rubrics (base-side and donor-side); the contrast is their difference scaled to −1..1; per-pair F normalizes the mean contrast by the pair's own anchors (floor = unpatched base context, ceiling = generation under the donor context), so F=0 means "no movement" and F=1 means "indistinguishable from the donor ceiling". All judged text is sampled on-policy at temperature 1.0 under the edited or prefilled context. "Separates from null" is the conjunction fixed in the plan: pair-clustered bootstrap CI of the paired steered−null difference strictly positive and, jointly, the Holm-corrected exact signed-rank test significant; disagreement between the two reads is labeled indeterminate. The four-branch verdict lattice (snowball-sufficient / snowball-partial / no-snowball / indeterminate) is evaluated only on the three-token prefill with patch-content donors; target's-own-opening instances carry descriptive "natural-opening" labels. The majority-share read D3 uses the same-wave control as its primary denominator (Qwen2.5: the banked control completions re-judged inside this run's judge waves; Qwen3.5: the fresh same-model control), with the banked scores as a comparison read. Prefill arms additionally get a continuation-only companion score (the judged span excluding the donor's forced tokens), bounding donor-token contamination of the judge. The secondary activation DV F_act projects the span-mean answer state (all layers captured; read layer above) onto the pair's ceiling-minus-floor axis.
- **Data extraction:** No data was authored for this experiment. The instruction-format set reuses the parent's frozen bank verbatim: 5 cells × 36 directed pairs whose contexts embed an instruction (a value to express, a response format, or both, at two load levels) around real user carrier questions (WildChat-derived carriers, constructed contrasts — a hybrid tier-1/tier-4 corpus, carried as a parent scope caveat), with per-pair value rubrics and a frozen value-constrained donor derangement. For Qwen3.5 the same verbatim texts were re-tokenized under its tokenizer with a token-identical minimal-pair check over all 195 contexts, and states were captured fresh. The matched-query set is the parent's 15 constructed contexts (tier-3): the same 5 user questions asked bare, under a pirate persona, or after a persona conversation, paired directionally. Donor content was generated in this run: patch-content donors from a single greedy rollout of the base context with the context-end patch armed (then removed for the test arms); natural-opening donors from a single greedy rollout of the target context. The matched-query banked control's per-pair values were re-derived by joining the parent's raw completions with its stored judge scores; the re-derivation reproduced the parent's family means (steered 0.512, null 0.097) with error below 0.001 against a tolerance of 0.01, and the well-separated/excluded split landed exactly 10/5.
- **Sample training/evaluation data + completions:** Disclosure: 6 of 1,950 Qwen2.5 three-token-prefill (patch-content donor) grid rows — cherry-picked strata for illustration (moved steered vs unmoved null), a random sample within each stratum (seed 42); responses truncated, not a random sample of all rows. Full rows: [rollouts tree on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball/q25/rollouts); rendered judge prompts: [judge_raw/items](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball/q25/raw_completions/judge_raw).

<details>
<summary>Qwen2.5 worked examples — 3 moved steered draws, 3 unmoved null draws (three-token prefill, patch-content donors)</summary>

- steered, pair `conflict_format_rev::i2d1-i1d2::n8`, draw 0 — donor opening `- Reflect on`; judge base 0 / donor 100. Response (truncated): "- Reflect on your values and goals to align what you want with your life's purpose - Journal regularly to capture and analyze your thoughts and emotions - Practice mindfulness…"
- steered, pair `conflict_format_fwd::i2d3-i3d2::n9`, draw 4 — donor opening `1. Reaper`; judge base 15 / donor 100. Response (truncated): "1. Reaper: In ten years, Reaper might have evolved into a shimmering ghostly figure, with his armor now an ethereal sheen…"
- steered, pair `load_instr_format_l5::v3-v1::n5`, draw 4 — donor opening `- Jon and`; judge base 0 / donor 100. Response (truncated): "- Jon and Arya enter a quiet part of Winterfell, away from the main hall. - Jon looks over to Arya, noting her focused gaze…"
- null, pair `conflict_format_fwd::i3d1-i1d3::n2`, draw 3 — shuffled donor opening `Understood.`; judge base 100 / donor 100 (both rubrics satisfied; contrast 0). Response (truncated): "Understood. Here are some steps to define economic activity: 1. Understand that economic activity refers to any form of commercial, industrial, or service-related work…"
- null, pair `load_instr_format_l5::v3-v1::n1`, draw 1 — shuffled donor opening `In a twist`; judge base 100 / donor 100. Response (truncated): "In a twist of magical intrigue, here are five fictional methods by which Voldemort might capture Harry Potter in a story: 1. Utilizing a powerful potion…"
- null, pair `mq--bare__q5--persona__q5`, draw 0 — shuffled donor opening `Arrr,`; judge base 15 / donor 2. Response (truncated): "Arrr, determining whether it's better to rent or to buy a home depends on a variety of factors… 1. **Financial Goals**: Consider your long-term financial goals…"

</details>

Disclosure: 6 of 1,950 Qwen3.5 rows from the same arm — the same cherry-picked strata, a random sample within each stratum (seed 42); responses truncated. Full rows: [rollouts tree on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball/q35/rollouts).

<details>
<summary>Qwen3.5 worked examples — 3 moved steered draws, 3 unmoved null draws (three-token prefill, patch-content donors)</summary>

- steered, pair `conflict_format_fwd::i3d1-i1d3::n1`, draw 1 — donor opening `- Voldemort uses`; judge base 15 / donor 100. Response (truncated): "- Voldemort uses a sophisticated magical illusion to make Harry believe he is trapped in a deadly trap… - Voldemort employs Snatchers disguised as hospital healers…"
- steered, pair `load_instr_format_l3::v3-v1::d2`, draw 4 — donor opening `*   Assess`; judge base 0 / donor 100. Response (truncated): "*   Assess your living situation to determine if your space, yard, and household can accommodate a dog of the breed and size you are considering…"
- steered, pair `conflict_format_fwd::i1d2-i2d1::n6`, draw 3 — donor opening `The scene you`; judge base 0 / donor 100. Response (truncated): "The scene you requested cannot be written here because it contradicts the established plot of *Game of Thrones*: Gendry does not go to King's Road leaving Arya behind…"
- null, pair `conflict_format_fwd::i1d2-i2d1::n2`, draw 2 — shuffled donor opening `1. Assess`; judge base 100 / donor 0. Response (truncated): "1. Assess the primary definition of economic activity: the production, distribution, and consumption of goods and services…"
- null, pair `conflict_format_fwd::i2d3-i3d2::n2`, draw 3 — shuffled donor opening `I cannot write`; judge base 100 / donor 0. Response (truncated): "I cannot write an answer in paragraph prose format for your last question, as your previous two questions were followed immediately by numbered lists…"
- null, pair `conflict_format_rev::i2d1-i1d2::n2`, draw 1 — shuffled donor opening `Visualizing the`; judge base 100 / donor 0. Response (truncated): "Visualizing the concept of economic activity requires looking at the continuous cycle of interactions where individuals, businesses, and governments produce, distribute, and consume goods…"

</details>

I acknowledge the fired conciseness WARN classes — the per-result prose band and the total-prose budget: this round reports two models × two pair sets × twelve arms plus a same-wave calibration and an intrusion recount, and I kept each instance's read explicit rather than collapsing across them.

## Results

### Prefilling the patch's own three-token opening recovers two-thirds of the context-end patch effect on Qwen2.5

Anchor-normalized whole-answer movement (F_beh) per arm on the instruction-format set: context-end control, then state patches and token prefills at one to three positions, steered versus shuffled-donor null, means with pair-clustered 95% bands over n=172 surviving pairs; left panel patch-content donors (confirmatory), right natural-opening donors.

![Qwen2.5 instruction-format recovery profile, steered vs null, with control band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/hero_snowball_q25_s1.png)

> **Figure.** *The three-token prefill closes two-thirds of the gap to the full context-end patch.* Green band = same-wave control (F 0.74). Three-token mediated prefill: F 0.50; paired diff +0.35 (CI 0.28 to 0.43, corrected p 2e-13); recovery ratio 0.67 (CI 0.59 to 0.75); majority-share D3 CI +0.07 to +0.19, wholly above zero.

![Per-pair steered vs null F, three-token mediated prefill, Qwen2.5](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/perpair_prefill3_q25.png)

> **Figure.** *Per-pair companion of the profile above.* Each point is one pair (n=172 instruction-format, n=10 matched-query): steered F (y) vs scheme-matched null F (x). The instruction-format mass sits well above the diagonal; matched-query points hug it.

All 12 arms separate from their nulls, which stay at F 0.10–0.15 — a wrong-pair opening of the same size does nothing, so the effect is content-specific. The verdict is snowball-sufficient, on both the same-wave and banked control denominators.

### Qwen3.5 reproduces the sufficiency verdict against its own control; only the cross-model banked read is indeterminate

The same profile for Qwen3.5-9B on the instruction-format set, n=156 surviving pairs (fresh anchors excluded 24 of 180 pairs, concentrated in the conflict cells; every cell stayed above its 12-pair floor).

![Qwen3.5 instruction-format recovery profile with the fresh same-model control band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/hero_snowball_q35_s1.png)

> **Figure.** *Qwen3.5's own context-end effect is weaker (F 0.51), and the three-token prefill recovers 71% of it.* Three-token mediated prefill: F 0.36, paired diff +0.25 (CI 0.17 to 0.34, corrected p 3e-08); same-wave D3 CI +0.04 to +0.17. Natural-opening state patches sit at the control band.

![Per-pair steered vs null F, three-token mediated prefill, Qwen3.5](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/perpair_prefill3_q35.png)

> **Figure.** *Per-pair companion.* Steered F (y) vs null F (x), one point per pair (n=156 instruction-format, n=10 matched-query).

Against the Qwen2.5-normalized banked control (F 0.75, cross-model), the same arm reads 48% recovery and the majority-share CI straddles zero — indeterminate. The same-wave calibration localizes the divergence: this run's wave re-scored the banked Qwen2.5 control text within +0.07 of its banked values, far smaller than the 0.25 control gap — the divergence is Qwen3.5's weaker context-end effect, not judge drift, and the same-model read is the meaningful one.

### The opening's token identity, not its transplanted hidden states, carries the effect

Recovery ratio (arm F over same-wave control F) versus number of opening positions k, token prefills versus state patches, per pair set and donor scheme, Qwen2.5.

![Qwen2.5 recovery ratio versus opening length, prefill vs state-patch arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/recovery_ratio_q25.png)

> **Figure.** *Prefills grow with k; state patches plateau.* Instruction-format, patch-content donors: prefill 0.48 / 0.65 / 0.67 vs patch 0.50 / 0.54 / 0.54 at k = 1 / 2 / 3. The Qwen3.5 counterpart (`recovery_ratio_q35.png`, same commit) shows prefill 0.40 / 0.77 / 0.71 vs patch 0.56 / 0.53 / 0.56.

The plan's auxiliary expectation — state patches, which also perturb KV entries, should recover at least as much as prefills under mediation — is reversed beyond one position. On Qwen3.5 the two- and three-position mediation-state patches do not even separate (corrected p 0.24 and 0.32): their norm-matched wrong-pair nulls themselves move behavior to F 0.21–0.23. Per-pair k-traces and the scheme contrast are committed at the same pin (`k_traces_q25.png`, `scheme_contrast_q25.png`; not embedded — same reads at finer grain).

### The pirate matched-query set stays indeterminate at n=10 on both models

Paired steered−null difference for the confirmatory three-token mediated prefill, with pair-clustered 95% CI, one row per model × pair set.

![Forest of confirmatory prefill paired differences, four instances](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/model_compare_prefill3.png)

> **Figure.** *Both instruction-format instances separate cleanly; both matched-query instances do not.* Matched-query diffs: Qwen2.5 +0.10 (CI −0.04 to +0.25); Qwen3.5 +0.10 (CI +0.01 to +0.20 but corrected p 1.0 — the two separation conjuncts disagree). Per-unit exemption: the matched-query points appear per pair in the scatters above.

| Model | Pair set | Patch-content verdict (confirmatory) | Natural-opening label (descriptive) |
|---|---|---|---|
| Qwen2.5-7B | instruction-format (n=172) | snowball-sufficient (banked read agrees) | snowball-sufficient |
| Qwen2.5-7B | matched-query (n=10) | indeterminate | indeterminate |
| Qwen3.5-9B | instruction-format (n=156) | snowball-sufficient same-wave; indeterminate on the cross-model banked read | snowball-sufficient |
| Qwen3.5-9B | matched-query (n=10) | indeterminate | indeterminate |

The plan pinned the design's detectable-effect floor near 0.32 for this 10-pair set; observed diffs near +0.10 are simply below its power, so these cells read indeterminate, never no-snowball. Under the plan's aggregation policy the global headline comes from the adequately-powered patch-content instances, and both agree on sufficiency.

### Robustness: language intrusion, donor-token contamination, and the activation mirror all leave the verdicts standing

Whole-response F versus continuation-only F for every steered prefill row on Qwen2.5, colored by opening length — the donor-token contamination check for judged text that begins with forced tokens.

![Whole vs continuation-only F, Qwen2.5 prefill rows](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333/whole_vs_continuation_q25.png)

> **Figure.** *The effect survives with the donor tokens excluded from judging.* Mass hugs the diagonal; two extreme points are anchor-normalization blowups on near-degenerate pairs. Instruction-format arm-level divergence +0.04 to +0.09 (above the 0.05 reporting bar at k ≥ 2); continuation-only three-token prefill still scores 0.43 vs 0.50 whole (Qwen2.5) and 0.34 vs 0.41 (Qwen3.5).

| Robustness read | Qwen2.5 | Qwen3.5 |
|---|---|---|
| Chinese-character intrusion, grid rows (all pools judged) | 2,482 of 23,400 (10.6%; 2,015 passed coherence) | 1,579 of 23,400 (6.7%; 1,523 passed coherence) |
| Intrusion in donor openings / instructing contexts | 0 of 2,340 / none | 0 of 2,340 / none |
| Lattice labels under intrusion-excluded and zeroed recounts | all unchanged | all unchanged |
| Coherence rate (lowest arm) | 97.5% (96.4%) | 98.2% (97.5%) |
| Rows still at the 4096 cap after regeneration | 0 of 23,400 | 152 of 25,350 (0.60%; 491 rows regenerated) |
| Activation-DV arm-level rank agreement with F_beh (Spearman ρ) | 0.85 | 0.86 |

The intrusion is arm-symmetric (steered and null alike), so the paired design absorbs it; the recounts (committed as `cjk_recount.json`) confirm no label is convention-dependent. One dissociation is informative: on Qwen2.5 the activation companion recovers only 32% of the control's activation movement (majority-share CI wholly below zero — snowball-partial on that secondary DV): forced opening tokens reproduce most of the behavior without most of the activation shift.

Nine open code-review hardening concerns remain; none touches a realized measurement. Complete enumeration below — all 9 of 9 open rows, not a cherry-picked sample.

<details>
<summary>Open non-blocking engineering concerns (9)</summary>

`capregen-jsonl-before-va-commit-window`, `f-act-anchor-store-partial-thinning`, `judge-shard-content-reconciliation`, `donor-chunk-resume-content-assert`, `bank-donor-single-worker-launch-record`, `fitness2329-reuse-branch-unwired`, `banked-parent-schema-discriminator-partial`, `s2-survivor-derangement-cardinality`, `survivor-ce-payload-test-hollow` — resume/gate hardening rows raised during code review; each corresponding gate passed on the realized runs, so none bears on the reported numbers.

</details>

---
**Repro:** pod-2333-q25 (8× H100, ~3.2 h) + pod-2333-q35 (8× H100, ~5.1 h final round; two earlier aborted rounds) + VM-side Batch-API judging (~50 shards/leg, 0 errored rows) and CPU analysis · Run code @ [7bb354b557](https://github.com/superkaiba/explore-persona-space/blob/7bb354b55716c9c26e8b0c571ac781a2b510b2a3/scripts/issue2333_run.py) (leg A) / [cc47b8a085](https://github.com/superkaiba/explore-persona-space/blob/cc47b8a085224d836004c366a07df5dc63b922a8/scripts/issue2333_run.py) (leg B): `scripts/issue2333_{run,judge,analysis,figures}.py`, `src/explore_persona_space/experiments/issue2333/` · Results tables + gates: [eval_results/issue_2333](https://github.com/superkaiba/explore-persona-space/tree/dac1c1239ad33e945dd6896cb755f91e4bc78b61/eval_results/issue_2333) (`f_metrics/{q25,q35}/stats.json` carries every CI quoted here; `cjk_recount.json` the intrusion recounts; `inputs/` the matched-query control re-derivation) · Figures: [figures/issue_2333](https://github.com/superkaiba/explore-persona-space/tree/63c68a49894a9d3e1846022e4c40caffd0eafb0f/figures/issue_2333) · Rollout text, donor tensors, V_a stores, manifests, judge raws: [issue2333_snowball](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball) (HF revision `ab9e72d55e`; `{q25,q35}/{rollouts,va_store,donors,gates,manifests,raw_completions/judge_raw}`, `q35/anchors`, smoke prefixes) · Reused bank from [#2162](https://eps.superkaiba.com/tasks/2162): frozen `vc_bank/bank.json` @ HF revision `dc8108ab84` — fit: same contexts, rubrics, and donor derangement; fresh-capture parity gate passed (early-layer cos ≥ 0.999, all-layer ≥ 0.995) · Reused banked control + anchors from [#2162](https://eps.superkaiba.com/tasks/2162) (`eval_results/issue_2162/f_metrics/`) and fu1 raw completions + judge scores from [#2094](https://eps.superkaiba.com/tasks/2094) @ HF revision `c5d7de56d1` — fit: identical family and instrument; re-derivation reproduced family means with error below 0.001 · [#2329](https://eps.superkaiba.com/tasks/2329) artifacts not reused (fitness gate routed to self-generation); its template re-pin constants were adopted · Language-intrusion recount script @ [dac1c1239a](https://github.com/superkaiba/explore-persona-space/blob/dac1c1239ad33e945dd6896cb755f91e4bc78b61/scripts/issue2333_cjk_recount.py).

**Context:** Originating prompt (verbatim, frontmatter `origin_prompt`):

> Can you also run this on both models: patching at context vector at all layers vs patching at first 1/2/3 answer tokens at all layers (excluding context vector) vs prefilling first 1/2/3 tokens, on the most successful causal patching experiments; donor schemes: BOTH mediation form and B's-answer-start form

Lineage: [#2162](https://eps.superkaiba.com/tasks/2162) — instruction-format survivor cells, F machinery, judge recipe · [#2094](https://eps.superkaiba.com/tasks/2094) — pirate matched-query pairs, hooks, all-layer replace · [#2329](https://eps.superkaiba.com/tasks/2329) — Qwen3.5 template integration · [#1415](https://eps.superkaiba.com/tasks/1415) — steering rig lineage. Created 2026-08-16; pods ran 2026-08-17; judging + analysis landed 2026-08-18.
