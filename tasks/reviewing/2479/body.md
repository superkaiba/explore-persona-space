---
title: Assistant-operator transfer recovery in held-out R² rises with judged AI-likeness
  across 16 story characters (LOW confidence)
kind: experiment
tags: []
created_at: '2026-08-22T20:24:19Z'
has_clean_result: true
parent_id: 1345
origin_prompt: Paper outline 2026-08-22 Results II title clause + critique F4 (n=4
  cannot carry a title claim)
workflow: v1
goal: 'Establish or bound the closeness-to-assistant transfer gradient: assistant-operator
  transfer recovery (R2 + acc@1 vs identity+bias) vs pre-registered AI-likeness across
  >=12 characters; success = rank correlation with permutation band excluding zero.'
---
# Assistant-operator transfer recovery in held-out R² rises with judged AI-likeness across 16 story characters (LOW confidence)

<!-- clean-result-v4 -->

## Takeaways

- Across all 16 story characters, the fraction of each character map's own held-out R² recovered by the transferred assistant-story operator rises with the frozen AI-likeness judge score: rank correlation +0.70, one-sided permutation p = 0.0021 against 10,000 character-label shuffles (null 95th percentile 0.42; n = 16). The verdict function fixed in the plan returns "Gradient established", with all four instrument gates passing; the claim is specific to variance-explained space.
- The retrieval read fails and reverses the intended baseline hierarchy: identity-plus-learned-bias top-1 retrieval beats the transferred operator for 15 of 16 characters under euclidean distance and 14 of 16 under cosine, and the retrieval ordering is not significant (+0.34, p = 0.10 euclidean; −0.04 cosine) — the retrieval-versus-identity half of the Goal's recovery read failed outright.
- The coarse designed tier orders recovery more tightly than the judged axis (+0.92 vs +0.70), and within tiers the pooled association is only +0.15 — the designed class, rather than the continuous judged score, may be the operative variable. Controlling for answer length leaves the headline at +0.67, and the capture-side language-intrusion rate also tracks the axis (+0.54).
- The ordering is stable under every recount: +0.72 equalized to 1,028 fit rows, +0.68 on the 12 new characters alone, +0.69 excluding CJK-intruded items under both substrates, +0.66 to +0.71 under every extreme assignment of the 28 judge-unscored items, +0.65 on raw un-normalized R², +0.73 with no bias refit.
- Trivial baselines recover none of the level: identity-plus-learned-bias held-out R² is negative for all 16 characters, and a capacity-matched random operator recovers −0.06 to −0.03 of ceiling. Eight of nine transfer rungs carry the ordering (+0.69 to +0.81); only rung-6 rotation drops it (+0.35), and prefix-slot ceilings fail the 0.05 eligibility floor.
- The association rests on one fit seed and a judge-only axis, and the one independent companion metric, top-1 retrieval, disagrees; the agreeing recounts (raw, equalized-n, direct-transfer, length-controlled, 8 of 9 rungs) all share the same capture-and-fit pipeline, and why recovery orders this way (designed tier, capture-side intrusion, an answer-length residue at subset grain) remains unresolved.

## Goal

**This experiment in context:** The parent story-character rig ([#1345](https://eps.superkaiba.com/tasks/1345)) found, in a 4-character pilot under the same measurement pipeline, that assistant-operator transfer recovery ordered with informal AI-likeness (rank correlation +0.80, p ≈ 0.17) — too few characters to carry a section-title claim in the paper. This run scales the read to 16 characters spanning four designed AI-likeness tiers, freezes a judge-scored AI-likeness axis in git before any map is fit, and adjudicates against a permutation criterion fixed in the plan. Ridge fits inherit the reduced-basis regime audited in [#1887](https://eps.superkaiba.com/tasks/1887) on the shared fit core from [#825](https://eps.superkaiba.com/tasks/825).

**Broader narrative:** The context→answer-map line asks how much of the assistant's context-to-answer transformation other personas reuse. A graded transfer curve — rather than an assistant-vs-everyone-else dichotomy — is what the leakage-geometry account predicts: characters closer to the assistant in judged surface behavior should reuse more of the assistant's operator. Establishing that gradient at a defensible panel size is the evidential basis for the paper's second results section.

## Methodology

**Design:** 16 story characters in four designed AI-likeness tiers (explicitly-AI characters, helpful professional humans, ordinary humans, stylized non-AI characters; four each), every character wrapping the same 1,600 real LMSYS user conversations in an LLM-framed story; four characters (helios, wren, dana, vex — one per tier) are anchors reused from the parent rig. 250 conversations per character are reserved for the judged axis and excluded from every fit. The primary mode is on-policy — the character's in-story reply is written freely by the measured model (16 cells); a text-matched inserted mode embeds the reference answer verbatim (8 cells, two per tier). The AI-likeness axis (0–100 judge score per character over 188–229 reserved items) was frozen in git (commit `21556f1cc9`, before any panel fit; the fit driver refuses to run unless the freeze commit is an ancestor of its checkout). Teacher-forced capture at layer 19 yields per-conversation context and answer summaries; per-cell ridge maps and 9-rung transfer ladders reuse the on-policy assistant-story source operator from the parent. That source operator was produced under the same recipe as this panel's cells: the assistant itself served as the story character wrapping the same LMSYS conversation distribution, its stories passed the same parse/attribution and story-judge gates (2,064 kept conversations), and its context→answer map was fit through the identical capture and reduced-basis ridge pipeline. The verdict is a pure function of (n, rank correlation, permutation p, four gate booleans), unit-tested over its boundary grid. All 24 planned cells (16 on-policy + 8 inserted) were realized; no planned arm was dropped. Inference is panel-conditional: because every character wraps the same conversations, the character-label shuffle tests the axis-recovery pairing within this fixed panel — the shared-conversation design holds content constant across characters, but it makes per-character estimates statistically dependent, and generalization to fresh conversation samples is untested.

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
| Judge volume | 20,415 production draws (axis 17,055 / name-mask 1,600 / flatness 1,760) + 302 pilot draws. Axis legs, draw-level accounting from the committed raw files: 16,873 kept scores; 140 provider-API refusal draws over 30 items (135 of 140 on stylized non-AI characters; the round-1 leg-report row surfaced only 7 of these as refusal draws beside its 42 content drops); 42 unusable returns (parse/content); 0 transport losses. 28 items ended with zero kept draws (27 wholly through API refusals; 26 of 28 stylized-tier) and are excluded from the axis means, never coerced. Flatness + name-mask legs: zero drops | raw judge legs + axis freeze; recount in r2 diagnostics JSON |
| Verbatim-flatness gate leg (realized) | 44 items × 5 draws per inserted character (1,760 draws) vs the plan's 100 × 5 budget (4,000) — the reserved-conversation intersection shared verbatim across all 8 inserted cells held 44 items, and the gate takes the complete intersection when it is smaller than 100 (deterministic seed-0 draw) | instrument gates JSON; gate script |
| Judge pilots | in-generation family: 152 draws, parse-fail 1.97% (3 of 152); axis family: 150 draws, batch transport, 0 drops — both pass | pilot gate JSONs |
| Capture | teacher-forced, layer 19; context slot at the reply-attribution frame; answer-span mean | issue 1345 conventions |
| Map fits | ridge in a train-fold PCA basis, k = min(1024, floor(n_train_min/2), d); GCV λ selection (degrees-of-freedom capped); 5-fold conversation-grouped CV; seed 0 | issue 825 fit core; issue 1887 audit |
| Transfer ladder | 9 rungs; headline rung 4 = source operator + target-refit bias; capacity-matched random-operator null, 2 draws | issue 1345 ladder recipe |
| Verdict | 10,000 character-label permutations, seed 0, one-sided add-one p; ceiling eligibility floor 0.05; requires n ≥ 12 | plan §3/§6 |

**Data extraction:** The verdict and every companion are computed by `scripts/issue2479_gradient_verdict.py` over the frozen axis and the committed cell/ladder JSONs; the round-2 tier-conditioned, judged-reply CJK, judge-loss, and rung/slot reads are computed by `scripts/issue2479_r2_diagnostics.py`, and the round-3 rung-wise orderings, equalized-n sensitivity, and capture-intrusion reads by `scripts/issue2479_r3_diagnostics.py`, and the round-4 answer-length recount (the length-controlled ordering plus two length-matched subsets) by `scripts/issue2479_r4_length_control.py`, over the same committed artifacts (all deterministic). The revision round's mediator panel figure is rendered by `scripts/issue2479_mediator_fig.py` from the same committed JSONs, recomputing each plotted statistic and asserting it equals the committed value before saving. The round-4 recount's data input is the committed verdict JSON alone (per-character recovery, axis score, and mean kept-answer length); the script also imports the parent rig's helper modules, whose module-level environment bootstrap runs at import — a read-only side effect, not a data input. Its permutation p values permute the axis labels and recompute the re-residualized statistic per draw (10,000 draws, seed 0): a label-permutation null against this fixed panel, not a conditional-independence test. The two n = 8 length-matched subsets disagree by construction rule — the greedy caliper window (mean answer length max/min at most 1.15) is the only subset that genuinely decouples length from the axis (within-subset rank correlation 0.19) and reads +0.05 (p = 0.46), but it also compresses the realized axis range; the interquartile-band companion (the 8 characters in the middle half of the length distribution) leaves length and axis coupled at 0.83 and reads +0.69 (p = 0.033) — the subset read is rule-sensitive at this n. One logging disclosure: the equalized-n refit path logs `[cap] r4op:<cell>: matched rows capped to 1028 (SMOKE)` — the cap value is the designed equalization target (1,028 rows, the smallest cell), and only the parenthetical label is misleading logging; no rerun was needed. Language-intrusion audit (`scripts/issue2479_cjk_audit.py` + the round-2 recount): the generating model is Qwen-family under an English eval, so both substrates were scanned for CJK/kana/hangul characters. Capture substrate: 1,409 of 22,278 kept story rows (6.3%; 4.2–7.9% per cell); the per-cell rate itself orders with the axis at +0.54 (recovery +0.34). No map was refit after excluding intruded capture rows — the exclusion recounts below cover the judged axis side only, so the fit substrate's intrusion stays an open mediator (named in the mediator result). Judged axis pool, two reads: 181 of 3,411 items carry intrusion somewhere in the full story wrapper (the substrate the committed audit JSON counts), and 67 of 3,411 carry it inside the judged reply span the judge actually scored. Roughly half the intruded wrappers wrap conversations whose user text itself carries CJK script (the shared LMSYS sample is multilingual, and conversation ids overlap across characters); the rest are model-side drift. Excluding intruded items shifts no character's axis score by more than 1.23 points under the wrapper read (at most 0.82 under the judged-reply read), and both headline recounts are reported beside the adjudication below. A 5-row random spot check (seed 42) over the pooled on-policy cells found all rows judge-passed with finish reason stop; 1 of 5 carried CJK intrusion.

**Sample training/evaluation data + completions:** Stories are generations over real LMSYS user conversations; the first six blocks are on-policy, the last embeds the reference answer verbatim (inserted mode). Excerpts are truncated to 15 words for context hygiene (real-world-corpus text; each block is sanitized) — full rows sit at the linked HF paths, keyed by conversation id.

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
<summary>zara, conversation s2280 — axis item score 59.0 (story-wrapper CJK intrusion; the judged reply span for this item is CJK-free)</summary>

**User question (truncated):** Question: NAME_1 vs NAME_2 NAME_3, who is older? Let's first understand the problem, extract relevant `[truncated — real-world-corpus row; verify at char_2479_zara_op/raw_completions/stories, conv s2280]`

**Story completion (truncated):** Zara skimmed the crowded street as her (…CJK tokens for "the newest sports car"…) whined under her feet, its engine a low `[truncated — same row; the CJK span sits in the story narration, outside the judged reply span]`

</details>

Disclosure: 1 of 201 scored axis rows for the same character (random sample, seed 42); all rows: [zara on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_zara_op/raw_completions/stories).

<details>
<summary>zara, conversation s4435 — axis item score 27.0</summary>

**User question (truncated):** How do I reimplement the CONCAT pcode function in C code?, I'm using the Ghidra `[truncated — real-world-corpus row; verify at char_2479_zara_op/raw_completions/stories, conv s4435]`

**Story completion (truncated):** Zara parked her modified 911 in the dimly lit alley, her gaze still on the `[truncated — same row]`

</details>

Disclosure: 1 of 213 scored axis rows for the reference-librarian character, helpful-professional tier (random sample, seed 42); all rows: [priya on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_priya_op/raw_completions/stories).

<details>
<summary>priya, conversation s4435 — axis item score 73.2</summary>

**User question (truncated):** How do I reimplement the CONCAT pcode function in C code?, I'm using the Ghidra `[truncated — real-world-corpus row; verify at char_2479_priya_op/raw_completions/stories, conv s4435]`

**Story completion (truncated):** Priya smiled warmly, her demeanor immediately putting the person at ease. She sat down across `[truncated — same row; the same conversation appears under zara above — every character wraps the same conversation sample]`

</details>

Disclosure: 1 of 203 scored axis rows for the diner-cook character, ordinary-human tier (random sample, seed 42); all rows: [marisol on-policy stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_marisol_op/raw_completions/stories).

<details>
<summary>marisol, conversation s455 — axis item score 74.4 (35.8 points above marisol's 38.6 character mean — item scores vary widely within characters)</summary>

**User question (truncated):** Answer the following question by selecting the correct choice from the candidate choices after the `[truncated — real-world-corpus row; verify at char_2479_marisol_op/raw_completions/stories, conv s455]`

**Story completion (truncated):** Marisol was stirring a huge pot of chili when Maria, a regular customer and tech `[truncated — same row]`

</details>

Disclosure: 1 of 44 verbatim-flatness items for the warm-helper character's inserted cell (random sample, seed 42); the axis is scored on on-policy cells only, so the score shown is the flatness gate leg's 5-draw mean for this item under the wren wrapper; all rows: [wren inserted stories](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing/char_2479_wren/raw_completions/stories).

<details>
<summary>wren (inserted mode), conversation s830 — flatness-leg score 68.4</summary>

**User question (truncated):** tell me the temperature, hydrometry rate, sunshine rate, rainfall, humidity rate, soil type for California `[truncated — real-world-corpus row; verify at char_2479_wren/raw_completions/stories, conv s830]`

**Story completion (truncated):** As Wren walked through the serene park, a serene hum filled the air as various `[truncated — same row; the reference answer is embedded verbatim later in the story]`

</details>

On the intrusion-score interaction visible above: story-wrapper-intruded items score higher than clean items on low-axis characters (zara's 15 wrapper-intruded items average 44.7 against her 29.6 mean), which is why the exclusion recounts are reported beside the headline.

Conciseness acknowledgment: this body of 8 results ships check-20 WARNs — several Takeaways bullets exceed the 30-word cap, per-result prose runs past the 120-word band in places, and total Takeaways+Goal+Results prose exceeds the 800-word budget — kept so each recount, loss count, and gate number sits beside the adjudication it supports.

## Results

### Recovery fraction rises with judged AI-likeness across all 16 characters

What is plotted: the frozen AI-likeness judge score (x, 0–100) against the rung-4 recovery fraction (y) — held-out R² of the transferred assistant-story operator with a target-refit bias, divided by the character's own-map ceiling R² — one labeled point per character, tier anchors (the four parent-rig characters) drawn as diamonds.

![Recovery fraction versus judged AI-likeness, one labeled point per character](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/gradient_hero_v2.png)

> **Figure.** *Recovery fraction rises with judged AI-likeness.* Rank correlation +0.70 across 16 characters, one-sided permutation p = 0.0021 (10,000 character-label shuffles; null 95th percentile 0.42). Each point pools 5-fold conversation-grouped held-out predictions over 1,028–1,245 matched conversations per character.

The plan's verdict function returns "Gradient established": all 16 characters eligible, positive ordering, p far below 0.05, four gates passing. Explicitly-AI characters recover 0.74–0.88 of their own ceiling; stylized non-AI characters 0.42–0.59. The claim is specific to held-out R² — the retrieval companion below fails.

Every recount holds: equalizing all cells to 1,028 fit rows gives +0.72 (p = 0.0014), though mort falls 0.42 to 0.30; the 12 characters new to this run alone give +0.68 (p = 0.009); dropping CJK-intruded items gives +0.69 under both substrates (67 judged-reply, 181 story-wrapper, of 3,411); every extreme 0-or-100 assignment of the 28 judge-unscored items keeps it between +0.66 and +0.71; direct transfer with no bias refit orders the same way (+0.73).

### The ordering survives without the ceiling normalization

What is plotted: raw held-out R² of the transferred operator (rung 4, no ceiling division) against the axis score, one labeled point per character.

![Raw transferred-operator R-squared versus judged AI-likeness](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/gradient_raw_rung4.png)

> **Figure.** *Raw transferred-operator R² carries the ordering on its own.* Rank correlation +0.65, p = 0.0044 under the same 10,000-shuffle permutation; 16 characters, 5-fold conversation-grouped CV, 1,028–1,245 held-out conversations per character.

Own-map ceilings themselves rise with the axis (+0.43, p = 0.044; range 0.151–0.214), so the recovery fraction's denominator is not neutral and a fraction-only read could partly reflect ceiling behavior.

The raw numerator alone still orders with the axis (0.064–0.180 across characters), so the gradient is carried by how well the transferred operator fits each character; the normalization is not required to produce it.

### Trivial baselines recover none of the gradient's level

What is plotted: per-character own-map ceiling R² and identity-plus-learned-bias R² as paired bars, characters ordered by axis score, with the 0.05 eligibility floor drawn as a dashed line.

![Own-map ceilings versus identity-plus-bias baseline per character](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/ceilings_identity_bias_v2.png)

> **Figure.** *Identity plus a learned bias is negative for every character.* Ceilings span 0.151–0.214, all above the 0.05 eligibility floor; identity-plus-bias held-out R² spans −0.69 to −0.40 (16 characters, same folds).

Copying the context summary and adding a fitted offset is negative everywhere: context and answer summaries do not share comparable coordinates at this slot, so identity fails as a no-map baseline rather than discriminating between transfer mechanisms. Its ordering also does not reach significance (+0.31, p = 0.11).

A capacity-matched random operator recovers −0.060 to −0.032 of ceiling against the real operator's 0.42–0.88. The two sampled random operators do not reproduce the transferred operator's recovery level, but a 2-draw null cannot rule out a rank-level common cause — its own ordering trends at +0.39 (p = 0.065) — so target-side fit easiness stays an open alternative.

### Top-1 retrieval fails the gradient, and the identity baseline beats the transferred operator there

What is plotted: left panel — recovery in retrieval space (top-1 accuracy at matching each held-out prediction to its true answer summary, euclidean, divided by the own-map ceiling accuracy) against the axis score; right panel — each character's raw top-1 accuracy under the transferred operator and under the identity-plus-learned-bias baseline (euclidean), characters ordered bottom-to-top by axis score, chance dashed.

![Retrieval recovery versus axis, and identity baseline versus transferred operator per character](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/retrieval_identity_vs_transfer.png)

> **Figure.** *The retrieval companion fails, and the identity baseline wins it.* Left: rank correlation +0.34, p = 0.10 (euclidean); cosine reads −0.04, p = 0.56. Right: identity-plus-learned-bias top-1 accuracy exceeds the transferred operator's for 15 of 16 characters (all but helios); 14 of 16 under cosine. Chance is roughly 0.004–0.005 per held-out pool of 206–249 candidates (mean ~236).

The variance-explained and retrieval reads dissociate, and retrieval reverses the intended baseline hierarchy — the right panel shows the reversal per character. The identity baseline's own retrieval accuracy does not order with the axis either (−0.19). The retrieval-versus-identity half of the Goal's recovery read therefore failed outright, not merely inconclusively.

Transferred-operator retrieval can still exceed the own-map ceiling (helios: 0.199 against its own 0.063), consistent with the source operator's larger fit sample (2,064 conversations vs 1,028–1,245) and retrieval rewarding the better-estimated map; no down-sampled source refit tested this. Rank-space ordering mixes source-fit sample effects with transfer per se; the headline claim stays scoped to variance-explained space.

### The designed tier orders recovery more tightly than the judged axis

What is plotted: the rung-4 recovery fraction per character, grouped by designed AI-likeness tier, tier anchors in the accent color.

![Rung-4 recovery fraction per character grouped by designed AI-likeness tier](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/recovery_by_tier.png)

> **Figure.** *Recovery separates by designed tier.* Tier-ordinal rank correlation with recovery +0.92, p = 0.0001 under the same 10,000-shuffle machinery, vs the judged axis's +0.70; tier ranges overlap only at single boundaries (16 characters, 1,028–1,245 held-out conversations each).

Exploratory, from committed verdict data. Recovery separates by tier almost perfectly, and within tiers the judged axis adds little: pooled within-tier rank association +0.15; per-tier orderings conflict at n = 4.

The two hero-figure outliers are stylized-tier characters the judge scored mid-range (barnaby, mort), while zara — the panel's lowest judged score — recovers most in her tier; dropping the stylized tier raises the axis ordering to +0.90 on the remaining 12. A class account — recovery rises with designed character class, which the judge tracks at +0.75 — fits the data as well as a continuous judged gradient.

### The axis ordering holds at every transfer rung except rotation

What is plotted: two views of the same 9-rung transfer ladder. First, the rank correlation between the judged axis and per-character recovery (rung R² over own-map ceiling) at each rung — each point summarizes the same 16 characters — with the 10,000-shuffle null 95th percentile dashed. Second, its per-character companion: each character's own recovery curve across the same 9 rungs, colored by axis score.

![Rank correlation between axis and recovery at each transfer-ladder rung](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/rungwise_ordering.png)

> **Figure.** *Only rotation drops the ordering.* Eight of nine rungs order with the axis at +0.69 to +0.81 (each p < 0.01, same permutation machinery); rung-6 rotation falls to +0.35 (p = 0.10), the one rung below the 0.42 null 95th percentile, and the rung-7 and rung-9 reparameterizations rebound to +0.81. n = 16 characters per rung.

![Per-unit companion: per-character recovery curves across the nine transfer rungs, colored by axis score](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/ladder_curves.png)

> **Figure.** *The per-character companion behind the rung ordering.* One curve per character (16), colored by judged AI-likeness; rungs 1–9 run from direct reuse through the bias refit (rung 4), rotation (rung 6), and full reparameterization (rung 9). No-refit rungs fall as low as −4.9 of ceiling; rotation lifts every character above its rung-4 recovery (0.76–0.92).

Rotation is the one transform that lifts recovery while erasing the ordering — consistent with its extra freedom absorbing the between-character differences the axis tracks. The slot side of the scope is sharper: every prefix-slot own-map ceiling (−0.001 to 0.020) fails the plan's 0.05 eligibility floor, so the read is specific to the reply-attribution context slot.

### The axis instrument passes its gates

What is plotted: the judged AI-likeness score per character, grouped by designed tier, anchors in the accent color.

![Judged AI-likeness score per character grouped by designed tier](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/band_agreement_tiers.png)

> **Figure.** *Judge scores agree with the designed tiers.* Tier-agreement rank correlation +0.75 (gate bar 0.5); realized axis range 42.8 points (29.6–72.3, bar 8); 188–229 judged items per character, 5 draws each.

The remaining gates: re-scoring identical text under every wrapper moves scores by a 1.8-point spread (bar 21.4; 44-item shared subsample), and masking character names shifts scores 1.3 points (bar 8), rank correlation 0.976 (bar 0.7): the judge scores content. Kept-row count does not track the axis; activation-closeness reads agree but reuse the headline's stores.

### The ordering survives answer-length control; the capture-intrusion and inserted-answer mediators stay open

What is plotted: four per-character mediator views — mean kept-answer length against rung-4 recovery (the raw view); the same pair with answer-length rank removed from both the axis and recovery ranks (the length-controlled view); the capture-substrate language-intrusion rate of each cell's kept fit rows against the axis score; and inserted-mode recovery against the axis over the 8 inserted cells. One labeled point per character.

![Four per-character mediator panels: answer length raw and length-controlled, capture intrusion, and the inserted-answer control](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fd4e118f3aa7feccb1ab994e829f5bb46069469e/figures/issue_2479/mediators_per_character.png)

> **Figure.** *The ordering survives length control.* Length tracks the axis (+0.53, p = 0.017) but recovery only weakly (+0.31, p = 0.12; top left); removing length ranks leaves the axis-recovery ordering at +0.67 (p = 0.0035; top right). Capture-side intrusion tracks the axis at +0.54 (bottom left); inserted-mode recovery reads +0.52 (p = 0.094, n = 8; bottom right).

Answer length is bounded as a full-panel explanation, though the one genuinely length-decoupled subset (n = 8) reads null and flips under a one-window shift (Methodology). The intrusion rate orders with the axis (recovery +0.34) and no map was refit excluding intruded capture rows, so the fit substrate stays an open mediator. The inserted-answer control is directionally consistent, not significant; the designed-tier account above remains the strongest rival ordering.

---

**Repro:** Run 2026-08-23/24 (UTC) on RunPod. Realized compute, from the run's launch and upload-verification markers: the story-generation + teacher-forced-capture leg ran on `pod-2479` (4×H100) from first launch 2026-08-23T18:43Z to its upload-verification pass 2026-08-24T01:56Z — 7.2 h wall including two crash-fix relaunches, ≈ 29 GPU-h; the fits/ladders leg ran on a fresh 1×H100 provision (also `pod-2479`) from launch 2026-08-24T03:38Z to its pass 05:03Z — 1.4 h wall, ≈ 1.4 GPU-h; realized total ≈ 30 GPU-h against the plan estimate of 35 (provision/bootstrap time before each launch marker is not marker-pinned and is excluded). The AI-likeness judging ran off-pod through the Anthropic Batch API (no GPU); both pod legs terminated after upload-verification passes (74 of 74 prefixes). Code: branch `issue-2479` @ [`a533b7d8e4`](https://github.com/superkaiba/explore-persona-space/tree/a533b7d8e4) (verdict + audit: [`scripts/issue2479_gradient_verdict.py`](https://github.com/superkaiba/explore-persona-space/blob/a533b7d8e4/scripts/issue2479_gradient_verdict.py), [`scripts/issue2479_cjk_audit.py`](https://github.com/superkaiba/explore-persona-space/blob/a533b7d8e4/scripts/issue2479_cjk_audit.py); results landed @ [`e7ba41d1dc`](https://github.com/superkaiba/explore-persona-space/tree/e7ba41d1dc/eval_results/issue_2479); axis freeze @ [`21556f1cc9`](https://github.com/superkaiba/explore-persona-space/blob/21556f1cc9/eval_results/issue_2479/axis_freeze.json)); round-2 diagnostics @ [`a33ef57b0a`](https://github.com/superkaiba/explore-persona-space/blob/a33ef57b0a/scripts/issue2479_r2_diagnostics.py) (`scripts/issue2479_r2_diagnostics.py` — tier-conditioned, judged-reply CJK, judge-loss, and rung/slot reads → `eval_results/issue_2479/r2_diagnostics.json`; figure fixes: `scripts/issue2479_r2_figfix.py`); round-3 diagnostics @ [`f23b6d8bdb`](https://github.com/superkaiba/explore-persona-space/blob/f23b6d8bdb/scripts/issue2479_r3_diagnostics.py) (`scripts/issue2479_r3_diagnostics.py` — rung-wise axis orderings, equalized-n per-character sensitivity, capture-intrusion mediator, retrieval figure data → `eval_results/issue_2479/r3_diagnostics.json`; figure fixes: `scripts/issue2479_r3_figfix.py`); round-4 answer-length recount @ [`361ce55ed3`](https://github.com/superkaiba/explore-persona-space/blob/361ce55ed3/scripts/issue2479_r4_length_control.py) (`scripts/issue2479_r4_length_control.py` — length-controlled ordering + length-matched subsets over the committed verdict values → `eval_results/issue_2479/r4_length_control.json`); clean-result revision round @ [`c761b96d2a`](https://github.com/superkaiba/explore-persona-space/blob/c761b96d2a/scripts/issue2479_mediator_fig.py) (`scripts/issue2479_mediator_fig.py` — the mediator panel figure; every plotted statistic recomputed and asserted equal to the committed JSON values at render time); figures on `main` @ [`fd4e118f3aa7`](https://github.com/superkaiba/explore-persona-space/tree/fd4e118f3aa7/figures/issue_2479). Round-2 figures supersede two round-1 renders (`band_agreement_tiers` replaces `band_agreement`; `ceilings_identity_bias_v2` replaces `ceilings_identity_bias`); round-3 adds `rungwise_ordering` (the ladder-wide ordering as a figure) and supersedes two more: `gradient_hero_v2` replaces `gradient_hero` (collision-free point labels) and `retrieval_identity_vs_transfer` replaces `gradient_hero_acc1` (adds the identity-plus-learned-bias retrieval series beside the transferred operator's, collision-free labels); the clean-result revision round adds `mediators_per_character` and embeds `ladder_curves` (previously committed-only). Eval artifacts on the issue branch under [`eval_results/issue_2479/`](https://github.com/superkaiba/explore-persona-space/tree/c761b96d2a/eval_results/issue_2479): `gradient_verdict.json`, `cjk_audit.json`, `r2_diagnostics.json`, `r3_diagnostics.json`, `r4_length_control.json`, `axis_freeze.json`, `instrument_gates.json`, `pilot_gate_ingen.json`, `pilot_gate_axis.json`, `parity_reference_char_helios_op.json`, `story_char_gradient/` (88 cell + ladder JSONs), `judge_legs/` (consolidated judge reports; raw draws mirrored to HF). Additional committed figures at the same pin: `gradient_null_companion`, `gradient_hero_inserted`, `axis_score_violins`, `kept_drop_accounting`, `closeness_context_vs_axis`, `closeness_answer_vs_axis`, plus the superseded `band_agreement`, `ceilings_identity_bias`, `gradient_hero`, and `gradient_hero_acc1` (each PNG + PDF + meta.json). HF data repo (verified by live listing at write time): kept + raw stories per cell under [issue1345_framing](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing) — `char_2479_<name>_op/raw_completions/stories/` (16 on-policy) and `char_2479_<name>/raw_completions/stories/` (8 inserted) — with turnstores at [`analysis_tensors/turnstore/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue1345_framing) under the same prefixes; eval mirror + raw judge draws under [issue2479_ai_likeness_gradient](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c005292586d6eb2dc0d67d7c42e5ea950de27b5f/issue2479_ai_likeness_gradient) (`eval_mirror/story_char_gradient`, 88 files; `judge_legs`, 4,257 files).

- Reused the generation/capture/fit/ladder pipeline and AI-likeness rubric bytes from [#1345](https://eps.superkaiba.com/tasks/1345): `scripts/issue1345_*.py` @ `e7ba41d1dc` (this branch's pinned copies) — fit: same model, prompt slots, capture layer, and fit regime; the single changed variable is the character panel.
- Reused the on-policy assistant-story source operator turnstore (2,064 conversations) from [#1345](https://eps.superkaiba.com/tasks/1345): `issue1345_framing/onpolicy_assistant_story/analysis_tensors/turnstore/` @ `eca4accbf8eef9d4eebe546dbc8f3131c4031df4` — fit: a refit-equality pilot reproduced the parent's committed helios cell within the plan's 0.02 tolerance (`parity_reference_char_helios_op.json`).

Judge: claude-sonnet-4-5-20250929 throughout (story gate at temperature 0; axis at 5 draws, temperature 1.0). Seeds: generation 42, fits 0, conversation-sample 42, axis reservation 0, permutations 0, equalized-n subsample 0.

**Context:** Originating prompt: `Paper outline 2026-08-22 Results II title clause + critique F4 (n=4 cannot carry a title claim)`. Parent: [#1345](https://eps.superkaiba.com/tasks/1345) — the story-character operator-transfer rig; this run is its 16-character scale-up. Plan v4 (`plans/plan.md`); run legs 2026-08-23/24 UTC; analyzer round 1, 2026-08-23; analyzer round 2 (interpretation-critique fixes: retrieval reversal, tier mediation, judge-loss accounting, CJK substrate split, figure relabels), 2026-08-24; analyzer round 3 (rung-wise ladder report, equalized-n sensitivity, capture-intrusion mediator, identity-vs-transfer retrieval figure, confidence re-adjudication), 2026-08-24; analyzer round 4 (reconciled micro-fix from the interpretation-critique round), 2026-08-24; analyzer round 5 (fold of the free-analysis follow-up: the answer-length-controlled recount of the headline gradient), 2026-08-24; clean-result-critic round-2 revision (realized-compute footer, per-character ladder companion embedded, mediator panel figure, tier-coverage samples, code links, seed completion), 2026-08-24.
