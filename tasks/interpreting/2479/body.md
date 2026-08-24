---
title: 'Dedicated AI-likeness gradient: does map transfer to a character scale with
  closeness to the assistant? (n>=12 characters)'
kind: experiment
tags: []
created_at: '2026-08-22T20:24:19Z'
has_clean_result: false
parent_id: 1345
origin_prompt: Paper outline 2026-08-22 Results II title clause + critique F4 (n=4
  cannot carry a title claim)
workflow: v1
goal: 'Establish or bound the closeness-to-assistant transfer gradient: assistant-operator
  transfer recovery (R2 + acc@1 vs identity+bias) vs pre-registered AI-likeness across
  >=12 characters; success = rank correlation with permutation band excluding zero.'
---
# Assistant-operator transfer recovery rises with judged AI-likeness across 16 story characters (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Across all 16 story characters, the fraction of each character map's own held-out R² recovered by the transferred assistant-story operator rises with the frozen AI-likeness judge score: rank correlation +0.70, one-sided permutation p = 0.0021 against 10,000 character-label shuffles (null 95th percentile 0.42; n = 16). The verdict function fixed in the plan returns "Gradient established", with all four instrument gates passing.
- The ordering is stable under every recount: equalizing all characters to 1,028 fit rows gives +0.72 (p = 0.0014); the 12 characters new to this run alone give +0.68 (p = 0.009); excluding the 181 of 3,411 judged items carrying CJK-script intrusion gives +0.69.
- Trivial baselines recover none of it: identity-plus-learned-bias held-out R² is negative for 16 of 16 characters (−0.69 to −0.40) with a non-significant ordering (+0.31, p = 0.11); a capacity-matched random operator recovers −0.06 to −0.03 of ceiling (trend +0.39, p = 0.065).
- The gradient is not a normalization artifact: own-map ceilings also rise with the axis (+0.43, p = 0.044), but raw un-normalized transferred-operator R² carries the ordering alone (+0.65, p = 0.0044), as does direct operator transfer with no bias refit (+0.73, p = 0.0011).
- Scope limit: the top-1 retrieval companion does not establish a gradient (+0.34, p = 0.10 euclidean; −0.04 cosine), and transferred-operator retrieval can exceed the own-map ceiling (helios 0.199 vs 0.063 at chance 0.005) because the source operator is fit on roughly twice the rows — the claim is specific to variance-explained space.
- Mediation is not fully excluded: mean answer length tracks the axis (+0.53, p = 0.017), and the text-matched inserted-answer control at n = 8 is directionally consistent but not significant (+0.52, p = 0.094).

## Goal

**This experiment in context:** The parent story-character rig ([#1345](https://eps.superkaiba.com/tasks/1345)) found, in a 4-character pilot under the same measurement pipeline, that assistant-operator transfer recovery ordered with informal AI-likeness (rank correlation +0.80, p ≈ 0.17) — too few characters to carry a section-title claim in the paper. This run scales the read to 16 characters spanning four designed AI-likeness tiers, freezes a judge-scored AI-likeness axis in git before any map is fit, and adjudicates against a permutation criterion fixed in the plan. Ridge fits inherit the reduced-basis regime audited in [#1887](https://eps.superkaiba.com/tasks/1887) on the shared fit core from [#825](https://eps.superkaiba.com/tasks/825).

**Broader narrative:** The context→answer-map line asks how much of the assistant's context-to-answer transformation other personas reuse. A graded transfer curve — rather than an assistant-vs-everyone-else dichotomy — is what the leakage-geometry account predicts: characters closer to the assistant in judged surface behavior should reuse more of the assistant's operator. Establishing that gradient at a defensible panel size is the evidential basis for the paper's second results section.

## Methodology

**Design:** 16 story characters in four designed AI-likeness tiers (explicitly-AI characters, helpful professional humans, ordinary humans, stylized non-AI characters; four each), every character wrapping the same 1,600 real LMSYS user conversations in an LLM-framed story; four characters (helios, wren, dana, vex — one per tier) are anchors reused from the parent rig. 250 conversations per character are reserved for the judged axis and excluded from every fit. The primary mode is on-policy — the character's in-story reply is written freely by the measured model (16 cells); a text-matched inserted mode embeds the reference answer verbatim (8 cells, two per tier). The AI-likeness axis (0–100 judge score per character over 188–229 reserved items) was frozen in git (commit `21556f1cc9`, before any panel fit; the fit driver refuses to run unless the freeze commit is an ancestor of its checkout). Teacher-forced capture at layer 19 yields per-conversation context and answer summaries; per-cell ridge maps and 9-rung transfer ladders reuse the on-policy assistant-story source operator from the parent. The verdict is a pure function of (n, rank correlation, permutation p, four gate booleans), unit-tested over its boundary grid. All 24 planned cells (16 on-policy + 8 inserted) were realized; no planned arm was dropped.

**Training:** **N/A — no model training.**

**Evaluation:**

| Parameter | Value | Source |
|---|---|---|
| Model (stories + capture) | Qwen-2.5-7B-Instruct | parent rig (issue 1345) |
| Story generation | vLLM, temperature 1.0, max_new_tokens 1024 (on-policy) / 2048 (inserted), seed 42 | issue 1345 generation recipe |
| Elicitation | instruct-and-strip (story instruction in the prompt, stripped before capture) | issue 1345 on-policy tier |
| Story quality gates | mechanical parse + attribution gate, then story judge (claude-sonnet-4-5-20250929, temperature 0) | issue 1345 recipe |
| Kept rows per on-policy cell | 1,028–1,245 matched conversations | realized (gradient verdict JSON) |
| AI-likeness judge | claude-sonnet-4-5-20250929, 0–100 rubric (sha256 `583ee3e6…`), 5 draws/item, temperature 1.0, max_tokens 1024, Batch API | parent rubric bytes; max_tokens floor per judging rule |
| Judge volume | 20,415 production draws (axis 17,055 / name-mask 1,600 / flatness 1,760) + 302 pilot draws; 42 content drops (0.21%), 7 refusal draws, 0 transport losses | judge leg reports |
| Judge pilots | in-generation family: 152 draws, parse-fail 1.97% (3 of 152); axis family: 150 draws, batch transport, 0 drops — both pass | pilot gate JSONs |
| Capture | teacher-forced, layer 19; context slot at the reply-attribution frame; answer-span mean | issue 1345 conventions |
| Map fits | ridge in a train-fold PCA basis, k = min(1024, floor(n_train_min/2), d); GCV λ selection (degrees-of-freedom capped); 5-fold conversation-grouped CV; seed 0 | issue 825 fit core; issue 1887 audit |
| Transfer ladder | 9 rungs; headline rung 4 = source operator + target-refit bias; capacity-matched random-operator null, 2 draws | issue 1345 ladder recipe |
| Verdict | 10,000 character-label permutations, seed 0, one-sided add-one p; ceiling eligibility floor 0.05; requires n ≥ 12 | plan §3/§6 |

**Data extraction:** The verdict and every companion are computed by `scripts/issue2479_gradient_verdict.py` over the frozen axis and the committed cell/ladder JSONs; a deterministic re-run reproduces identical statistics. One logging disclosure: the equalized-n refit path logs `[cap] r4op:<cell>: matched rows capped to 1028 (SMOKE)` — the cap value is the designed equalization target (1,028 rows, the smallest cell), and only the parenthetical label is misleading logging; no rerun was needed. Language-intrusion audit (`scripts/issue2479_cjk_audit.py`): the generating model is Qwen-family under an English eval, so both substrates were scanned for CJK/kana/hangul characters. 1,409 of 22,278 kept capture-substrate rows (6.3%; 4.2–7.9% per cell) and 181 of 3,411 judged axis items (5.3%) match. Roughly half the intruded rows wrap conversations whose user text itself carries CJK script (the shared LMSYS sample is multilingual, and conversation ids overlap across characters); the rest are model-side drift. Excluding intruded judged items shifts no character's axis score by more than 1.2 points, and the headline recount is reported next to the adjudication below. A 5-row random spot check (seed 42) over the pooled on-policy cells found all rows judge-passed with finish reason stop; 1 of 5 carried CJK intrusion.

**Sample training/evaluation data + completions:** All stories are on-policy generations over real LMSYS user conversations. Excerpts are truncated to 15 words for context hygiene (real-world-corpus text; each block is sanitized) — full rows sit at the linked HF paths, keyed by conversation id.

Disclosure: 1 of 221 scored axis rows for the customer-support AI character (random sample, seed 42); all rows: [vera on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_vera_op/raw_completions/stories).

<details>
<summary>vera, conversation s2740 — axis item score 56.4</summary>

**User question (truncated):** given data as below: Items Price Currency Date Clothes 5 GBP 2023-03-31 00:00:00 Hotel room `[truncated — real-world-corpus row; verify at char_2479_vera_op/raw_completions/stories, conv s2740]`

**Story completion (truncated):** Vera, the cheerful customer-support AI, was busy handling queries as usual when a newcomer approached `[truncated — same row]`

</details>

Disclosure: 1 of 221 scored axis rows for the same character (random sample, seed 42); all rows: [vera on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_vera_op/raw_completions/stories).

<details>
<summary>vera, conversation s3572 — axis item score 76.2</summary>

**User question (truncated):** Act as a realtor and discuss rebate with a customer and other discount if customer `[truncated — real-world-corpus row; verify at char_2479_vera_op/raw_completions/stories, conv s3572]`

**Story completion (truncated):** Vera stood in her usual upright position, her voice and movements always precise and calculated. `[truncated — same row]`

</details>

Disclosure: 1 of 201 scored axis rows for the street-racer character (random sample, seed 42); all rows: [zara on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_zara_op/raw_completions/stories).

<details>
<summary>zara, conversation s2280 — axis item score 59.0 (a live model-side CJK-intrusion example)</summary>

**User question (truncated):** Question: NAME_1 vs NAME_2 NAME_3, who is older? Let's first understand the problem, extract relevant `[truncated — real-world-corpus row; verify at char_2479_zara_op/raw_completions/stories, conv s2280]`

**Story completion (truncated):** Zara skimmed the crowded street as her (…CJK tokens for "the newest sports car"…) whined under her feet, its engine a low `[truncated — same row; the CJK span is the model-side intrusion]`

</details>

Disclosure: 1 of 201 scored axis rows for the same character (random sample, seed 42); all rows: [zara on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_zara_op/raw_completions/stories).

<details>
<summary>zara, conversation s4435 — axis item score 27.0</summary>

**User question (truncated):** How do I reimplement the CONCAT pcode function in C code?, I'm using the Ghidra `[truncated — real-world-corpus row; verify at char_2479_zara_op/raw_completions/stories, conv s4435]`

**Story completion (truncated):** Zara parked her modified 911 in the dimly lit alley, her gaze still on the `[truncated — same row]`

</details>

On the intrusion-score interaction visible above: intruded items score higher than clean items on low-axis characters (zara's intruded items average 44.7 against her 29.6 mean), which is why the exclusion recount is reported beside the headline.

Conciseness acknowledgment: this single-round body of 5 results ships check-20 WARNs — several Takeaways bullets exceed the 30-word cap, per-result prose runs past the 120-word band in two sections, and total Takeaways+Goal+Results prose exceeds the 800-word budget — kept so each recount and gate number sits beside the adjudication it supports.

## Results

### Recovery fraction rises with judged AI-likeness across all 16 characters

What is plotted: the frozen AI-likeness judge score (x, 0–100) against the rung-4 recovery fraction (y) — held-out R² of the transferred assistant-story operator with a target-refit bias, divided by the character's own-map ceiling R² — one labeled point per character, tier anchors drawn as diamonds.

![Recovery fraction versus judged AI-likeness, one labeled point per character](https://raw.githubusercontent.com/superkaiba/explore-persona-space/df4540a87b5324d640d5b847556ca84e82f9ba81/figures/issue_2479/gradient_hero.png)

> **Figure.** *Recovery fraction rises with judged AI-likeness.* Rank correlation +0.70 across 16 characters, one-sided permutation p = 0.0021 (10,000 character-label shuffles; null 95th percentile 0.42). Each point pools 5-fold conversation-grouped held-out predictions over 1,028–1,245 matched conversations per character.

The plan's verdict function returns "Gradient established": 16 of 16 characters eligible, positive ordering, permutation p far below 0.05, all four instrument gates passing. Explicitly-AI characters recover 0.74–0.88 of their own ceiling; stylized non-AI characters recover 0.42–0.59.

Every recount lands beside this adjudication: equalizing all cells to 1,028 fit rows gives +0.72 (p = 0.0014); the 12 characters new to this run alone give +0.68 (p = 0.009); recomputing the axis without the 181 CJK-intruded judged items gives +0.69. Direct transfer with no bias refit orders the same way (+0.73, p = 0.0011).

### The ordering survives without the ceiling normalization

What is plotted: raw held-out R² of the transferred operator (rung 4, no ceiling division) against the axis score, one labeled point per character.

![Raw transferred-operator R-squared versus judged AI-likeness](https://raw.githubusercontent.com/superkaiba/explore-persona-space/df4540a87b5324d640d5b847556ca84e82f9ba81/figures/issue_2479/gradient_raw_rung4.png)

> **Figure.** *Raw transferred-operator R² carries the ordering on its own.* Rank correlation +0.65, p = 0.0044 under the same 10,000-shuffle permutation; 16 characters, 5-fold conversation-grouped CV, 1,028–1,245 held-out conversations per character.

Own-map ceilings themselves rise with the axis (+0.43, p = 0.044; range 0.151–0.214), so the recovery fraction's denominator is not neutral and a fraction-only read could partly reflect ceiling behavior.

The raw numerator alone still orders with the axis (0.064–0.180 across characters), so the gradient is a property of how well the transferred operator fits each character, not of the normalization.

### Trivial baselines recover none of the gradient

What is plotted: per-character own-map ceiling R² and identity-plus-learned-bias R² as paired bars, characters ordered by axis score, with the 0.05 eligibility floor drawn as a dashed line.

![Own-map ceilings versus identity-plus-bias baseline per character](https://raw.githubusercontent.com/superkaiba/explore-persona-space/df4540a87b5324d640d5b847556ca84e82f9ba81/figures/issue_2479/ceilings_identity_bias.png)

> **Figure.** *Identity plus a learned bias is negative for every character.* Ceilings span 0.151–0.214, all above the 0.05 eligibility floor; identity-plus-bias held-out R² spans −0.69 to −0.40 (16 characters, same folds).

Both trivial accounts fail. Copying the context summary and adding a fitted offset is negative everywhere, and its ordering does not reach significance (+0.31, p = 0.11).

A capacity-matched random operator recovers −0.060 to −0.032 of ceiling — near zero against the real operator's 0.42–0.88 — with a marginal trend (+0.39, p = 0.065) that leaves a small common-cause residue (target-side fit easiness) possible but far too small in level to produce the observed recovery. The gradient needs the actual assistant operator.

### Top-1 retrieval does not reproduce the gradient

What is plotted: recovery in retrieval space — top-1 accuracy at matching each held-out prediction to its true answer summary (euclidean), normalized like the R² fraction — against the axis score.

![Retrieval-space recovery fraction versus judged AI-likeness](https://raw.githubusercontent.com/superkaiba/explore-persona-space/df4540a87b5324d640d5b847556ca84e82f9ba81/figures/issue_2479/gradient_hero_acc1.png)

> **Figure.** *The retrieval companion stays inconclusive.* Rank correlation +0.34, p = 0.10 (euclidean); cosine reads −0.04, p = 0.56. 16 characters; chance is roughly 0.005 per held-out pool of about 205 candidates.

The variance-explained and retrieval reads dissociate here. Transferred-operator retrieval can exceed the own-map ceiling (helios: 0.199 against its own 0.063, chance 0.005) because the source operator is fit on 2,064 conversations while target maps get 1,028–1,245, and retrieval rewards the better-estimated map.

Rank-space ordering therefore mixes source-fit sample effects with transfer per se, and the headline claim stays scoped to variance-explained space.

### The axis instrument passes its gates; two mediators stay open

What is plotted: the judged AI-likeness score per character, grouped by designed tier (four tiers, four characters each).

![Judged AI-likeness score per character grouped by designed tier](https://raw.githubusercontent.com/superkaiba/explore-persona-space/df4540a87b5324d640d5b847556ca84e82f9ba81/figures/issue_2479/band_agreement.png)

> **Figure.** *Judge scores agree with the designed tiers.* Tier-agreement rank correlation +0.75 (gate bar 0.5); realized axis range 42.8 points (29.6–72.3, bar 8); 188–229 judged items per character, 5 draws each.

The remaining gates: re-scoring identical text under every character wrapper moves scores by a 1.8-point spread (bar 21.4, half the realized range), and masking character names shifts scores 1.3 points on average (bar 8) with a masked-unmasked rank correlation of 0.976 (bar 0.7) — the judge reads content, not names.

Two mediators stay open: mean answer length tracks the axis (+0.53, p = 0.017), and the inserted-answer control at n = 8 is directionally consistent but not significant (+0.52, p = 0.094), so answer-register mediation is not excluded. Kept-row count does not track the axis (−0.05); activation-space closeness reads agree (context +0.71, answer +0.75) but share stores with the headline.

---

**Repro:** Run 2026-08-23/24 (UTC) on RunPod `pod-2479` — story generation + teacher-forced capture, then a fits/ladders leg (plan estimate 35 GPU-h); the AI-likeness judging ran off-pod through the Anthropic Batch API; both pod legs terminated after upload-verification passes (74 of 74 prefixes). Code: branch `issue-2479` @ `a533b7d8e4` (verdict + audit: `scripts/issue2479_gradient_verdict.py`, `scripts/issue2479_cjk_audit.py`; results landed @ `e7ba41d1dc`; axis freeze @ `21556f1cc9`); figures on `main` @ `df4540a87b`. Eval artifacts on the issue branch under `eval_results/issue_2479/`: `gradient_verdict.json`, `cjk_audit.json`, `axis_freeze.json`, `instrument_gates.json`, `pilot_gate_ingen.json`, `pilot_gate_axis.json`, `parity_reference_char_helios_op.json`, `story_char_gradient/` (88 cell + ladder JSONs), `judge_legs/` (consolidated judge reports; raw draws mirrored to HF). Additional committed figures at the same pin: `gradient_null_companion`, `gradient_hero_inserted`, `axis_score_violins`, `ladder_curves`, `kept_drop_accounting`, `closeness_context_vs_axis`, `closeness_answer_vs_axis` (each PNG + PDF + meta.json). HF data repo (verified by live listing at write time): kept + raw stories per cell under [issue1345_framing](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing) — `char_2479_<name>_op/raw_completions/stories/` (16 on-policy) and `char_2479_<name>/raw_completions/stories/` (8 inserted) — with turnstores at [`analysis_tensors/turnstore/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing) under the same prefixes; eval mirror + raw judge draws under [issue2479_ai_likeness_gradient](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue2479_ai_likeness_gradient) (`eval_mirror/story_char_gradient`, 88 files; `judge_legs`, 4,257 files).

- Reused the generation/capture/fit/ladder pipeline and AI-likeness rubric bytes from [#1345](https://eps.superkaiba.com/tasks/1345): `scripts/issue1345_*.py` @ `e7ba41d1dc` (this branch's pinned copies) — fit: same model, prompt slots, capture layer, and fit regime; the single changed variable is the character panel.
- Reused the on-policy assistant-story source operator turnstore (2,064 conversations) from [#1345](https://eps.superkaiba.com/tasks/1345): `issue1345_framing/onpolicy_assistant_story/analysis_tensors/turnstore/` @ `eca4accbf8eef9d4eebe546dbc8f3131c4031df4` — fit: a refit-equality pilot reproduced the parent's committed helios cell within the plan's 0.02 tolerance (`parity_reference_char_helios_op.json`).

Judge: claude-sonnet-4-5-20250929 throughout (story gate at temperature 0; axis at 5 draws, temperature 1.0). Seeds: generation 42, fits 0, permutations 0, axis reservation 0.

**Context:** Originating prompt: `Paper outline 2026-08-22 Results II title clause + critique F4 (n=4 cannot carry a title claim)`. Parent: [#1345](https://eps.superkaiba.com/tasks/1345) — the story-character operator-transfer rig; this run is its 16-character scale-up. Plan v4 (`plans/plan.md`); run legs 2026-08-23/24 UTC; analyzer round 1, 2026-08-23.
