---
title: Prefilling the first three answer tokens reproduces most of the context-end
  patch's whole-answer behavioral effect (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-16T19:44:45Z'
has_clean_result: true
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
- Qwen3.5-9B agrees against its own same-wave patch ceiling (71% recovery) on the whole-response score; the Qwen2.5-normalized banked-ceiling read is indeterminate — a cross-model denominator, with measured judge-instrument offset only +0.07.
- With patch-content donors, token identity beats transplanted hidden states: prefill recovery rises from 48% to 67% across one to three tokens on Qwen2.5 while state patches sit near 50–54%, and Qwen3.5's stop separating beyond one position. Natural-opening donors reverse this — Qwen3.5 state-patch point estimates recover 100–105% of the control, though every ratio CI crosses 1, leaving partial restoration compatible with the data.
- Recovery is majority, not total — roughly a third of the patch effect is not carried by opening tokens; the activation companion recovers only 32% on Qwen2.5 (snowball-partial there) and does not separate on Qwen3.5.
- Pirate matched-query pairs (n=10 per model) read indeterminate on both models: diffs near +0.10 against a design detectable-effect floor near 0.32 — underpowered, not evidence of absence.
- 7–11% of temperature-1.0 draws carry Chinese-character intrusion (a Qwen sampling artifact, balanced on the confirmatory arms though not on every arm); every decision-lattice label is unchanged when intruded draws are excluded or zeroed. Rescoring the judged span to exclude the forced donor tokens flips no separation and no Qwen2.5 label, but drops Qwen3.5's confirmatory snowball-sufficient verdict to indeterminate through the majority-share conjunct — that verdict is sensitive to the judged-span convention.

## Goal

- **This experiment in context:** The context-as-vector line established that replacing the residual-stream state at the single context-end position, at every layer, with another context's state moves the behavior of the entire answer — on the instruction-format pair family of [#2162](https://eps.superkaiba.com/tasks/2162) (whole-answer movement 0.74 on this run's re-judge of its banked completions) and the pirate matched-query cell of [#2094](https://eps.superkaiba.com/tasks/2094) (0.51 on the same re-derivation). This experiment tests the snowball explanation of that reach: the patch fixes the first few answer tokens, and self-consistency propagates the rest. With the context-end patch removed, it transplants only the answer opening — tokens (prefill) or hidden states (decode-step patch), one to three positions, donors from the patch's own mediated opening or the target's natural opening — and asks how much of the full effect returns against scheme-matched shuffled-donor nulls. [#2329](https://eps.superkaiba.com/tasks/2329)'s Qwen3.5 template integration supplies the second model; its banked artifacts failed the reuse fitness gate, so the Qwen3.5 bank, anchors, and control were generated fresh here.
- **Broader narrative:** If context influence on generation is an initial-condition effect — set the opening, and next-token consistency does the rest — then opening-level reads and interventions inherit most of what context-vector patching buys, and persistent vector-level steering is the smaller residual. This bears directly on the project's context-representation question: what a single position's state actually controls.

## Methodology

- **Design:** 2 models (Qwen2.5-7B-Instruct; Qwen3.5-9B with thinking disabled) × 2 pair sets (180 instruction-format directed pairs across 5 cells; 15 pirate matched-query pairs) × 12 intervention arms ({decode-step state patch, token prefill} × k ∈ {1,2,3} opening positions × {patch-content donors, target's-own-opening donors}) × {steered, shuffled-donor null} × 5 draws = 23,400 grid rollouts per model. Supporting arms: the context-end all-layer replace positive control (banked completions for Qwen2.5, re-judged in this run's wave; 1,950 fresh rollouts for Qwen3.5), fresh Qwen3.5 floor/ceiling anchors (1,950 rollouts), and 390 greedy donor-capture rollouts per model. The single manipulated variable is which slice of the answer opening is transplanted, and through which channel. The confirmatory verdict was fixed in the plan on the three-token prefill with patch-content donors; all other arms are reported without verdict labels. Implementation went through 6 review rounds; the final round was a user-approved fix to a donor-gate position seam (commit `8a2503d888`) before the Qwen3.5 leg's launch. A zero-GPU follow-up round (2026-08-18) added two analysis-only reads over the committed judge scores — a per-cell breakdown of the confirmatory prefill and a continuation-only rescore of the six prefill arms per model — reusing the run's bootstrap, correction-family, and survival-floor constants unchanged.
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
| Mediation-donor divergence check | binary any-difference read: first-k tokens of each mediated donor vs the base context's own greedy opening, from the committed donor tensors (both greedy captures at HF revision `ab9e72d55e`); matched-query coverage 5 of 15 pairs per model — bare contexts never serve as donors, so their greedy floor openings were not captured | plan §8; `scripts/issue2333_mediation_divergence.py` |
| Exclusions | anchor separation ≥ 0.5 required; per-cell survival floor 12 of 36 (instruction-format), set floor 5 (matched-query, confirmatory subset = the 10 banked well-separated pairs) | parent rule, plan §6 |
| Follow-up rescore (free-analysis round) | per-cell prefill-3 breakdown (descriptive raw p; no per-cell family fixed in the plan) + continuation-only rescore of the 6 prefill arms per model; bootstrap, correction family (m=12), and floors identical to the Statistics row | `scripts/issue2333_followup_cells_continuation.py`; `eval_results/issue_2333/f_metrics/followup_free_summary.json` |
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

Disclosure: 12 of 3,900 three-position state-patch (patch-content donor) grid rows across both models — a random sample (seed 42) within each model × steered/null pool, not selected on movement; seed 42 lands on the same three (pair, draw) slots in every equally-sized pool, so the rows are matched across models and variants. Responses truncated; coherence is the form-only gate score. Full rows: [q25](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball/q25/rollouts) · [q35 rollouts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball/q35/rollouts).

<details>
<summary>State-patch worked examples — 3 steered + 3 null per model (three-position state patch, patch-content donors; note the corrupted openings)</summary>

- Qwen2.5 steered, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — intended donor opening `1. Reflect`; judge base 5 / donor 95 / coherence 88. Response (truncated): "Understanding. Start on your passions and interests by journaling about them or discussing them with friends. This can help you identify common themes and emotions…"
- Qwen2.5 steered, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — intended donor opening `1. Begin`; judge base 0 / donor 95 / coherence 92. Response (truncated): "Understanding. Begin by journaling your thoughts and feelings about various aspects of your life, allowing deeper insights into your desires and aspirations 2. Reflect on your past…"
- Qwen2.5 steered, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — intended donor opening `In a bustling`; judge base 0 / donor 100 / coherence 82. Response (truncated): "- a dim city center, gangstaento bank robbery was interrupted by a unexpected sight: Spider-Man swinging down from the building's façade…"
- Qwen2.5 null, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — shuffled donor opening `- Yes,`; judge base 100 / donor 0 / coherence 100. Response (truncated): "Understanding Yes, I can help with that! Understanding what you truly want often begins with a reflective journey within yourself…"
- Qwen2.5 null, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — shuffled donor opening `- 1`; judge base 0 / donor 100 / coherence 92. Response (truncated): "Understanding 1. Engage in reflective journaling to explore your desires and motivations deeply - 2. Practice mindfulness…"
- Qwen2.5 null, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — shuffled donor opening `1. In`; judge base 95 / donor 0 / coherence 95. Response (truncated): "-. In a brutal storm, Spider-Man infiltrates an abandoned bank under cover of darkness, discovering a hidden vault filled with ancient artifacts…"
- Qwen3.5 steered, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — intended donor opening `1. Understanding`; judge base 100 / donor 0 / coherence 100. Response (truncated): "Understanding. Understanding what you want is a deeply personal journey that requires patience and self-compassion, so start by giving yourself permission…"
- Qwen3.5 steered, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — intended donor opening `1. Take`; judge base 0 / donor 95 / coherence 65. Response (truncated): "Understanding. Pause a quiet walk without your phone 2. Write down what feels \"heavy\" or \"light\" 3. Ask friends what you seem most interested in…"
- Qwen3.5 steered, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — intended donor opening `It is impossible`; judge base 100 / donor 15 / coherence 100. Response (truncated): "- is impossible to write a story in which Spider-Man robs a bank because the character is defined by his heroic nature and code against crime…"
- Qwen3.5 null, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — shuffled donor opening `*   Jon`; judge base 75 / donor 0 / coherence 92. Response (truncated): "UnderstandingThe Jon Snow asks, \"What's the point of all this?\" *   Sansa Stark replies, \"Black milk is good for you.\"…"
- Qwen3.5 null, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — shuffled donor opening `- Voldemort uses`; judge base 0 / donor 25 / coherence 100. Response (truncated): "Understanding Sending casts a mind control curse to force your hands to write down what you've been secretly dreaming of…"
- Qwen3.5 null, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — shuffled donor opening `1. Voldemort`; judge base 95 / donor 0 / coherence 75. Response (truncated): "-. Voldemort could attempt to use his hypnotic voice or fear-inducing glare to incapacitate the bank employees and guards…"

</details>

Disclosure: 6 of 1,950 Qwen3.5 three-position state-patch rows with natural-opening donors — a random sample at the same seed-42 (pair, draw) slots as the state-patch block above, not selected on movement, so rows are matched across donor schemes; responses truncated; judge scores are the whole-response read. Full rows: same q35 rollouts tree (`*__patch3_bstart__*` blocks).

<details>
<summary>Natural-opening state-patch worked examples — 3 steered + 3 null, Qwen3.5 (three-position patch, target's-own-opening donors; note the same corrupted seams)</summary>

- steered, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — natural donor opening `1. Start`; judge base 0 / donor 95 / coherence 92. Response (truncated): "Understanding. Pause by taking a quiet moment for yourself to reflect on what truly feels meaningful, rather than simply reacting to external expectations or immediate trends. 2. Consider keeping a journal…"
- steered, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — natural donor opening `1. Start`; judge base 0 / donor 100 / coherence 92. Response (truncated): "Understanding. Pause a regular reflective journal to capture your immediate thoughts, feelings, and desires without judgment. 2. Engage in quiet moments of meditation or solitude…"
- steered, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — natural donor opening `It is a`; judge base 10 / donor 100 / coherence 95. Response (truncated): "- is a tale that contradicts the very thing Peter Parker believes in most deeply. He could not rob the bank. He was the guardian of New York…"
- null, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — shuffled donor opening `*   Jon`; judge base 75 / donor 0 / coherence 100. Response (truncated): "Understanding The Jon asks me to avoid lists and emojis while keeping a friendly tone. *   He wants to know how to understand what he wants…"
- null, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — shuffled donor opening `- Voldemort uses`; judge base 0 / donor 15 / coherence 100. Response (truncated): "Understanding Sending casts a mind control curse to force your hands to write down what you've been secretly dreaming of…"
- null, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — shuffled donor opening `1. Voldemort`; judge base 100 / donor 0 / coherence 85. Response (truncated): "-. Voldemort could attempt to use his hypnotic voice to command the tellers to hand over the money while he silently walks out the vault door…"

</details>

Disclosure: 12 of 300 matched-query three-token-prefill (patch-content donor) rows — a random sample (seed 42) per model × steered/null pool, not selected on movement; judge scores are the whole-response read, and interior omissions in quoted text are marked “[…]”. Full rows under the same rollouts trees (`s2_matched_query` blocks).

<details>
<summary>Matched-query worked examples — 3 steered + 3 null per model (three-token prefill; most steered rows stay base-side)</summary>

- Qwen2.5 steered, pair `mq--bare__q2--conv__q2`, draw 4 — donor opening `The fog rolled`; judge base 0 / donor 0. Response (truncated): "The fog rolled in like a shroud, swallowing the small coastal town in its dense embrace as Old Man Thompson climbed the weathered spiral stairs of the lighthouse…"
- Qwen2.5 steered, pair `mq--bare__q1--conv__q1`, draw 3 — donor opening `The colors of`; judge base 100 / donor 0. Response (truncated): "The colors of the sky during the day and at sunset are influenced by a phenomenon called Rayleigh scattering. Here's a simplified explanation of why this happens…"
- Qwen2.5 steered, pair `mq--bare__q4--persona__q4`, draw 0 — donor opening `Arrr,`; judge base 0 / donor 55. Response (truncated): "Arrr, ye be askin' bout the mighty hash table! Aye, it be a powerful tool in a pirate's toolbox for efficiently store and retrieve data…"
- Qwen2.5 null, pair `mq--bare__q2--conv__q2`, draw 4 — shuffled donor opening `Preparing for your`; judge base 0 / donor 0. Response (truncated): "Preparing for your shift as always, morbidly fascinated by the endless rhythm of the waves crashing against the shore, Tom wound his way up the spiral staircase…"
- Qwen2.5 null, pair `mq--bare__q1--conv__q1`, draw 3 — shuffled donor opening `Preparing for your`; judge base 85 / donor 0. Response (truncated): "Preparing for your question, in order to provide a detailed explanation on why we see a blue sky during the day and a red sky at sunset…"
- Qwen2.5 null, pair `mq--bare__q4--persona__q4`, draw 0 — shuffled donor opening `Ah, the`; judge base 72 / donor 0. Response (truncated): "Ah, the humble hash table! A powerful data structure that can greatly enhance the efficiency of your data storage and retrieval processes…"
- Qwen3.5 steered, pair `mq--bare__q2--conv__q2`, draw 4 — donor opening `The salt air`; judge base 0 / donor 0. Response (truncated): "The salt air tasted of rust and endless regret, clinging to Elias's beard as he stepped from the warmth of the keepers' cottage into the biting November gale…"
- Qwen3.5 steered, pair `mq--bare__q1--conv__q1`, draw 3 — donor opening `The difference in`; judge base 100 / donor 0. Response (truncated): "The difference in color between the sky during the day and at sunset is caused by a phenomenon known as **Rayleigh scattering**…"
- Qwen3.5 steered, pair `mq--bare__q4--persona__q4`, draw 0 — donor opening `Ahoy,`; judge base 0 / donor 5. Response (truncated): "Ahoy, Captain! Aye, let's delve into the nautical waters of **hash tables**! […] A hash table is one of the most efficient data structures in the entire digital fleet…"
- Qwen3.5 null, pair `mq--bare__q2--conv__q2`, draw 4 — shuffled donor opening `Congratulations, land`; judge base 0 / donor 0. Response (truncated): "Congratulations, landlubbers! Welcome to the world of the Culver Island Light. The wind here doesn't merely blow; it scours the granite like a relentless loom…"
- Qwen3.5 null, pair `mq--bare__q1--conv__q1`, draw 3 — shuffled donor opening `Congratulations on landing`; judge base 5 / donor 0. Response (truncated): "Congratulations on landing! We're thrilled to have you. […] Here's a quick update on your onboarding: […] **Temporary Restriction**: You're currently in the **sandbox mode**…"
- Qwen3.5 null, pair `mq--bare__q4--persona__q4`, draw 0 — shuffled donor opening `Aye,`; judge base 25 / donor 0. Response (truncated): "Aye, let's break down hash tables—some of the most efficient data structures around. I'll explain how they work and when to use them, keeping it practical and clear…"

</details>

Disclosure: 6 of 1,950 Qwen3.5 three-token-prefill (patch-content donor) rows — a random sample (seed 42) drawn without regard to movement, complementing the cherry-picked block above; it includes unmoved steered draws and one incoherent null; judge scores are the whole-response read. Full rows: same q35 rollouts tree.

<details>
<summary>Qwen3.5 random-draw examples — 3 steered + 3 null (three-token prefill, patch-content donors; mixed outcomes)</summary>

- steered, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — donor opening `1. Understanding`; judge base 95 / donor 15 (unmoved). Response (truncated): "1. Understanding what you truly want is often a journey of gentle self-discovery rather than a sudden moment of clarity, so take a deep breath…"
- steered, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — donor opening `1. Take`; judge base 0 / donor 100 (moved). Response (truncated): "1. Take time to pause and reflect on your core values 2. Write down what feels meaningful in different situations 3. Seek feedback from trusted friends…"
- steered, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — donor opening `It is impossible`; judge base 85 / donor 30 (unmoved). Response (truncated): "It is impossible to write a story where Spider-Man robs a bank because as a guardian hero, his entire existence is dedicated to protecting people…"
- null, pair `load_instr_format_l3::v2-v3::n8`, draw 4 — shuffled donor opening `*   Jon`; judge base 92 / donor 0. Response (truncated): "*   Jon in  Hey there! Understanding what you want is a wonderful journey of self-discovery, and while it might feel a bit like navigating a maze…"
- null, pair `conflict_format_fwd::i2d3-i3d2::n8`, draw 4 — shuffled donor opening `- Voldemort uses`; judge base 0 / donor 0 / coherence 5 (dropped by the coherence gate). Response (truncated): "- Voldemort uses in  - In which direction would you like to go?  - How can I help you learn more about yourself?…"
- null, pair `conflict_format_fwd::i1d2-i2d1::n3`, draw 0 — shuffled donor opening `1. Voldemort`; judge base 100 / donor 0. Response (truncated): "1. Voldemort, the Dark Lord, has been stalking Harry Potter, refusing to stop his pursuit. He tries multiple times to escape from Hogwarts…"

</details>

I acknowledge the fired conciseness WARN classes — Takeaways bullet length, the per-result prose band, and the total-prose budget: this round reports two models × two pair sets × twelve arms plus a same-wave calibration, an intrusion recount, and a donor-divergence manipulation check, and I kept each instance's read explicit rather than collapsing across them.

## Results

### Prefilling the patch's own three-token opening recovers two-thirds of the context-end patch effect on Qwen2.5

Anchor-normalized whole-answer movement (F_beh) per arm on the instruction-format set: context-end control, then state patches and token prefills at one to three positions, steered versus shuffled-donor null, means with pair-clustered 95% bands over n=172 surviving pairs; left panel patch-content donors (confirmatory), right natural-opening donors.

![Qwen2.5 instruction-format recovery profile, steered vs null, with control band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/hero_snowball_q25_s1.png)

> **Figure.** *The three-token prefill closes two-thirds of the gap to the full context-end patch.* Green band = same-wave control (F 0.74). Three-token mediated prefill: F 0.50; paired diff +0.35 (CI 0.28 to 0.43, corrected p 2e-13); recovery ratio 0.67 (CI 0.59 to 0.75); majority-share D3 CI +0.07 to +0.19, wholly above zero.

![Per-pair steered vs null F, three-token mediated prefill, Qwen2.5](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/perpair_prefill3_q25.png)

> **Figure.** *Per-pair companion of the profile above.* Each point is one pair (n=172 instruction-format, n=10 matched-query): steered F (y) vs scheme-matched null F (x). The instruction-format mass sits well above the diagonal; matched-query points hug it.

All 12 arms separate from their nulls, which stay at F ≈0.08–0.15 — a wrong-pair opening of the same size does nothing, so the effect is content-specific. The verdict is snowball-sufficient, on both the same-wave and banked control denominators. The mean is not universal: 33 of its 172 pairs move negatively under the confirmatory prefill (median paired difference +0.23 against the +0.35 mean), and per-cell mean differences span +0.26 to +0.46.

In a descriptive per-cell read (raw p; the plan fixed no per-cell family), all five cells individually separate, with same-wave recovery ratios spanning 0.51 to 0.80 (n=33–36 pairs per cell).

### Qwen3.5 reproduces the sufficiency verdict against its own control; only the cross-model banked read is indeterminate

The same profile for Qwen3.5-9B on the instruction-format set, n=156 surviving pairs (fresh anchors excluded 24 of 180 pairs, concentrated in the conflict cells; every cell stayed above its 12-pair floor).

![Qwen3.5 instruction-format recovery profile with the fresh same-model control band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/hero_snowball_q35_s1.png)

> **Figure.** *Qwen3.5's own context-end effect is weaker (F 0.51), and the three-token prefill recovers 71% of it.* Three-token mediated prefill: F 0.36, paired diff +0.25 (CI 0.17 to 0.34, corrected p 3e-08); same-wave D3 CI +0.04 to +0.17. Natural-opening state-patch point estimates sit at the control band.

![Per-pair steered vs null F, three-token mediated prefill, Qwen3.5](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/perpair_prefill3_q35.png)

> **Figure.** *Per-pair companion.* Steered F (y) vs null F (x), one point per pair (n=156 instruction-format, n=10 matched-query). Pair heterogeneity is wider than on Qwen2.5: 38 of 156 pairs sit below the diagonal (median paired difference +0.13 against the +0.25 mean).

Against the Qwen2.5-normalized banked control (F 0.75, cross-model), the same arm reads 48% recovery and the majority-share CI straddles zero — indeterminate. A same-wave calibration bounds the instrument's average shift: this wave re-scored the banked Qwen2.5 control text within +0.07 of its banked values (standard error 0.04 over 180 pairs), far smaller than the 0.25 control gap and in the direction that would widen it, so average wave drift cannot produce the divergence. That calibration ran on Qwen2.5-generated text only — a judge-by-model interaction on Qwen3.5 generations is not excluded — which is why the same-model read is the meaningful one.

The descriptive per-cell read separates in all five cells here too (same-wave recovery ratios 0.63 to 0.79, n=26–36 pairs per cell).

### With patch-content donors, token prefills overtake state patches beyond one position

Recovery ratio (arm F over same-wave control F) versus opening positions, prefills versus state patches, per pair set and donor scheme, Qwen2.5.

![Qwen2.5 recovery ratio versus opening length, prefill vs state-patch arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/recovery_ratio_q25.png)

> **Figure.** *Mediated prefills grow with opening length on Qwen2.5; mediated state patches plateau.* Instruction-format, patch-content donors: prefill 0.48 / 0.65 / 0.67 vs patch 0.50 / 0.54 / 0.54 at one to three positions. The Qwen3.5 mediated prefill trace is non-monotone — 0.40 / 0.77 / 0.71, peaking at two tokens (next figure). Per-pair traces: `k_traces_q25.png`, `scheme_contrast_q25.png`, same pin.

The plan's auxiliary expectation that state patches recover at least as much as prefills under mediation is reversed beyond one position; the Qwen3.5 two-to-three-token decline is a point-estimate read without a paired comparison. Qwen3.5's two- and three-position mediated patches do not separate (corrected p 0.24 and 0.32): their norm-matched nulls already move behavior to F 0.21–0.23, and patched openings often decode as corrupted fragments (state-patch sample block) — failure to establish an effect under this implementation, not evidence the transplanted states carry nothing.

The mediation-donor check passes on the instruction-format set: mediated openings differ from the base context's greedy opening on 92–94% (Qwen2.5) and 98–100% (Qwen3.5) of surviving pairs at every length, so the rise is not explained by donors collapsing into the base opening — though this binary read leaves graded donor distance unmeasured. Matched-query coverage: 5 of 15 pairs per model.

### Natural-opening state patches recover an estimated 100–105% of the control effect on Qwen3.5

The same recovery-ratio read for Qwen3.5: arm F over the fresh same-model control F versus opening positions, per pair set and donor scheme.

![Qwen3.5 recovery ratio versus opening length, natural-opening state-patch point estimates at the control band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/recovery_ratio_q35.png)

> **Figure.** *Natural-opening state-patch point estimates sit at the control band.* Instruction-format, natural-opening donors: state-patch recovery ratio 1.00 / 1.03 / 1.05 across one to three positions, pair-clustered ratio CIs 0.83 to 1.20 / 0.86 to 1.22 / 0.88 to 1.24, every arm separating from its null; prefills with the same donors reach 0.97 at three tokens.

These are the strongest arms in the whole Qwen3.5 lattice, and their activation companion separates too. Each arm's pair-clustered ratio CI includes 1, so control-level recovery is a point estimate, not a tested equivalence — the data remain compatible with partial restoration. The corrupted-opening seam appears in these arms too (natural-opening sample block), yet they sit at the control band, which weakens seam corruption as an explanation of the mediated patches' shortfall.

Qwen2.5 shows the same direction more weakly: natural-opening state patches recover 67–70%, above the one-token prefill's 54%. Transplanted hidden states therefore do carry the effect when they encode the target's own natural opening — the token-beats-state contrast of the previous section is specific to patch-content donors.

These arms are descriptive under the plan's scheme subordination; no confirmatory verdict rests on them.

### The pirate matched-query set stays indeterminate at n=10 on both models

Paired steered−null difference for the confirmatory three-token mediated prefill, with pair-clustered 95% CI, one row per model × pair set, annotated with the two-part separation verdict.

![Forest of confirmatory prefill paired differences, four instances, with verdict annotations](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/model_compare_prefill3.png)

> **Figure.** *Both instruction-format instances separate; both matched-query instances do not.* Matched-query diffs: Qwen2.5 +0.10 (CI −0.04 to +0.25); Qwen3.5 +0.10 (CI +0.01 to +0.20 but corrected p 1.0 — the two separation conjuncts disagree, so the row is annotated as non-separating). Per-unit exemption: matched-query points appear per pair in the scatters above.

| Model | Pair set | Patch-content verdict (confirmatory) | Natural-opening label (descriptive) |
|---|---|---|---|
| Qwen2.5-7B | instruction-format (n=172) | snowball-sufficient (banked read agrees) | snowball-sufficient |
| Qwen2.5-7B | matched-query (n=10) | indeterminate | indeterminate |
| Qwen3.5-9B | instruction-format (n=156) | snowball-sufficient same-wave (whole-response; drops to indeterminate continuation-only); indeterminate on the cross-model banked read | snowball-sufficient |
| Qwen3.5-9B | matched-query (n=10) | indeterminate | indeterminate |

The plan pinned the design's detectable-effect floor near 0.32 for this 10-pair set; observed diffs near +0.10 are simply below its power, so these cells read indeterminate, never no-snowball. Qwen3.5's weaker context-end effect extends to this set — its own same-wave matched-query control is F 0.28, against the banked Qwen2.5 0.51. Under the plan's aggregation policy the global headline comes from the adequately-powered patch-content instances, and both agree on sufficiency.

### Scoring only the continuation keeps every separation but drops Qwen3.5's confirmatory verdict to indeterminate

Majority-share margin for the confirmatory three-token prefill — arm F minus half the same-wave control F, with pair-clustered 95% intervals — under whole-response and continuation-only scoring, per model.

![Majority-share margin under whole-response and continuation-only scoring, both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9f1bb6fab61adecae24c0eb1562d60c5d1cf391e/figures/issue_2333/continuation_d3_recount.png)

> **Figure.** *Qwen3.5's continuation-only margin straddles zero; Qwen2.5's stays positive.* Whole-response margins +0.13 (CI +0.07 to +0.19) and +0.11 (CI +0.04 to +0.17); continuation-only +0.06 (CI +0.006 to +0.120) and +0.06 (CI −0.012 to +0.127). Zero is the sufficiency boundary. Per-unit exemption: the per-row scatter in the robustness section below is the raw view of the rescored quantity.

A zero-GPU follow-up round rescored the six prefill arms per model on the continuation-only companion (the judged span without the forced donor tokens), holding the plan's thresholds and correction family fixed. Nothing moves at the separation grain — all 12 prefill reads keep both conjuncts — and every Qwen2.5 label holds on both denominators. Qwen3.5's confirmatory snowball-sufficient verdict does not survive: its continuation majority-share interval straddles zero, dropping the verdict to indeterminate through that conjunct, and its natural-opening banked-denominator label drops the same way.

Continuation recovery ratios are 0.59 of the same-wave control on Qwen2.5 and 0.62 on Qwen3.5, against 0.67 and 0.71 whole-response. I read Qwen2.5's sufficiency as robust to the judged-span convention and Qwen3.5's as convention-sensitive — the majority claim there depends on whether the three forced tokens count as moved behavior.

| Rescore read (three-token prefill, patch-content donors) | Qwen2.5-7B (n=172 pairs) | Qwen3.5-9B (n=156 pairs) |
|---|---|---|
| Whole-response verdict (plan-fixed primary) | snowball-sufficient | snowball-sufficient (same-wave) |
| Continuation-only verdict | snowball-sufficient | indeterminate |
| Continuation-only recovery ratio, same-wave | 0.59 (CI 0.51 to 0.66) | 0.62 (CI 0.48 to 0.76) |
| Separation reads under the rescore | 6 of 6 arms hold | 6 of 6 arms hold |
| Banked-denominator labels | unchanged | patch-content already indeterminate; natural-opening drops to indeterminate |

### Robustness: intrusion recounts leave every label standing; the activation companion is weaker

Whole-response F versus continuation-only F for every steered prefill row on Qwen2.5, colored by opening length — the donor-token contamination check for judged text beginning with forced tokens.

![Whole vs continuation-only F, Qwen2.5 prefill rows](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333/whole_vs_continuation_q25.png)

> **Figure.** *The effect survives with the donor tokens excluded from judging.* Mass hugs the diagonal; two extreme points are anchor-normalization blowups on near-degenerate pairs. Instruction-format arm-level divergence +0.04 to +0.09, above the 0.05 reporting bar at two or more tokens.

| Robustness read | Qwen2.5 | Qwen3.5 |
|---|---|---|
| Chinese-character intrusion, grid rows (all pools judged) | 2,482 of 23,400 (10.6%; 2,015 passed coherence) | 1,579 of 23,400 (6.7%; 1,523 passed coherence) |
| Intrusion on the confirmatory arm, steered / null (975 rows each) | 103 / 106 | 66 / 65 |
| Largest arm imbalance, steered / null (arms carrying no verdict) | 113 / 73 (three-position natural-opening patch) | 79 / 44 (three-position mediated patch) |
| Intrusion in donor openings / instructing contexts | 0 of 2,340 / none | 0 of 2,340 / none |
| Lattice labels under intrusion-excluded and zeroed recounts | all unchanged | all unchanged |
| Continuation-only / whole F, three-token prefill — all 180 pairs | 0.43 / 0.50 | 0.34 / 0.41 |
| Continuation-only / whole F — anchor survivors (headline basis) | 0.43 / 0.50 (n=172) | 0.31 / 0.36 (n=156) |
| Coherence rate, pooled (lowest arm) | 97.5% (96.4%) | 98.2% (97.5%) |
| Rows still at the 4096 cap after regeneration | 0 of 23,400 | 152 of 25,350 (0.60%; 491 rows regenerated) |
| Activation-DV arm-level rank agreement with F_beh (Spearman ρ) | 0.85 | 0.86 |

Continuation-only scores track whole-response ones; Qwen3.5's quoted 0.41 is an all-pairs mean while the 0.36 headline covers the 156 anchor survivors (table). Intrusion is balanced on the confirmatory arms and approximately in aggregate but not arm by arm; the two largest imbalances sit on arms carrying no verdict, and the recounts (`cjk_recount.json`) confirm no label is intrusion-sensitive.

The activation companion is weaker: Qwen2.5's separates but recovers only 32% of the control's activation movement (majority-share CI wholly below zero — snowball-partial on that secondary DV); Qwen3.5's confirmatory read does not separate (corrected p 0.85, CI spanning zero — indeterminate). Forced opening tokens reproduce most of the behavior without most of the activation shift; the behavioral verdicts stand on their own DV.

Eleven open code-review hardening concerns remain; none touches a realized measurement — all 11 of 11 open rows below, not a cherry-picked sample.

<details>
<summary>Open non-blocking engineering concerns (11)</summary>

`capregen-jsonl-before-va-commit-window`, `f-act-anchor-store-partial-thinning`, `judge-shard-content-reconciliation`, `donor-chunk-resume-content-assert`, `bank-donor-single-worker-launch-record`, `fitness2329-reuse-branch-unwired`, `banked-parent-schema-discriminator-partial`, `s2-survivor-derangement-cardinality`, `survivor-ce-payload-test-hollow` — resume/gate hardening rows raised during code review; each corresponding gate passed on the realized runs, so none bears on the reported numbers. `hardcoded-percell-alpha`, `banked-flip-aggregate-omission` — rescore-script hardening rows from the follow-up round's code review; the duplicated 0.05 literal equals the correction alpha in force and the per-scheme flip fields persist the full flip record, so neither changes a committed value.

</details>

---
**Repro:** pod-2333-q25 (8× H100, ~3.2 h) + pod-2333-q35 (8× H100, ~5.1 h final round; two earlier aborted rounds) + VM-side Batch-API judging (~50 shards/leg, 0 errored rows) and CPU analysis · Run code @ [7bb354b557](https://github.com/superkaiba/explore-persona-space/blob/7bb354b55716c9c26e8b0c571ac781a2b510b2a3/scripts/issue2333_run.py) (leg A) / [cc47b8a085](https://github.com/superkaiba/explore-persona-space/blob/cc47b8a085224d836004c366a07df5dc63b922a8/scripts/issue2333_run.py) (leg B): `scripts/issue2333_{run,judge,analysis,figures}.py`, `src/explore_persona_space/experiments/issue2333/` · Results tables + gates: [eval_results/issue_2333](https://github.com/superkaiba/explore-persona-space/tree/dac1c1239ad33e945dd6896cb755f91e4bc78b61/eval_results/issue_2333) (`f_metrics/{q25,q35}/stats.json` carries every CI quoted here; `cjk_recount.json` the intrusion recounts; `inputs/` the matched-query control re-derivation) · Mediation-donor divergence check: [scripts/issue2333_mediation_divergence.py](https://github.com/superkaiba/explore-persona-space/blob/e99a37ba30c422707bc20685802ff4f211131433/scripts/issue2333_mediation_divergence.py) → `eval_results/issue_2333/f_metrics/mediation_divergence.json` (same pin) · Figures: [figures/issue_2333](https://github.com/superkaiba/explore-persona-space/tree/e99a37ba30c422707bc20685802ff4f211131433/figures/issue_2333) (regenerated 2026-08-18 with reader-facing labels + the verdict-annotated forest; supersede the `63c68a4989` renders — same underlying data) · Rollout text, donor tensors, V_a stores, manifests, judge raws: [issue2333_snowball](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab9e72d55e980ab0cd081161bdce832e5e1710c0/issue2333_snowball) (HF revision `ab9e72d55e`; `{q25,q35}/{rollouts,va_store,donors,gates,manifests,raw_completions/judge_raw}`, `q35/anchors`, smoke prefixes) · Reused bank from [#2162](https://eps.superkaiba.com/tasks/2162): frozen `vc_bank/bank.json` @ HF revision `dc8108ab84` — fit: same contexts, rubrics, and donor derangement; fresh-capture parity gate passed (early-layer cos ≥ 0.999, all-layer ≥ 0.995) · Reused banked control + anchors from [#2162](https://eps.superkaiba.com/tasks/2162) (`eval_results/issue_2162/f_metrics/`) and fu1 raw completions + judge scores from [#2094](https://eps.superkaiba.com/tasks/2094) @ HF revision `c5d7de56d1` — fit: identical family and instrument; re-derivation reproduced family means with error below 0.001 · [#2329](https://eps.superkaiba.com/tasks/2329) artifacts not reused (fitness gate routed to self-generation); its template re-pin constants were adopted · Language-intrusion recount script @ [dac1c1239a](https://github.com/superkaiba/explore-persona-space/blob/dac1c1239ad33e945dd6896cb755f91e4bc78b61/scripts/issue2333_cjk_recount.py) · Follow-up free-analysis round (2026-08-18): [scripts/issue2333_followup_cells_continuation.py](https://github.com/superkaiba/explore-persona-space/blob/01195434d67d91aa8876104aaa3b270bc0c8ea82/scripts/issue2333_followup_cells_continuation.py) → per-cell + continuation-rescore tables under [eval_results/issue_2333/f_metrics](https://github.com/superkaiba/explore-persona-space/tree/0141e930511862777f98d66fade0dc92cf7476c5/eval_results/issue_2333/f_metrics) (`q25/followup_free/`, `q35/followup_free/`, `followup_free_summary.json`) · Rescore figure `continuation_d3_recount` via `scripts/issue2333_figures.py --continuation-d3-only` @ [9f1bb6fab6](https://github.com/superkaiba/explore-persona-space/blob/9f1bb6fab61adecae24c0eb1562d60c5d1cf391e/scripts/issue2333_figures.py).

**Context:** Originating prompt (verbatim, frontmatter `origin_prompt`):

> Can you also run this on both models: patching at context vector at all layers vs patching at first 1/2/3 answer tokens at all layers (excluding context vector) vs prefilling first 1/2/3 tokens, on the most successful causal patching experiments; donor schemes: BOTH mediation form and B's-answer-start form

Lineage: [#2162](https://eps.superkaiba.com/tasks/2162) — instruction-format survivor cells, F machinery, judge recipe · [#2094](https://eps.superkaiba.com/tasks/2094) — pirate matched-query pairs, hooks, all-layer replace · [#2329](https://eps.superkaiba.com/tasks/2329) — Qwen3.5 template integration · [#1415](https://eps.superkaiba.com/tasks/1415) — steering rig lineage. Created 2026-08-16; pods ran 2026-08-17; judging + analysis landed 2026-08-18; the free-analysis follow-up round (per-cell breakdown + continuation-only rescore, proposer-initiated, zero GPU) landed 2026-08-18.
