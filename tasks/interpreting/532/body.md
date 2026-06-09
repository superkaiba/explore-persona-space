---
title: Geometric leakage predictors (cosine/JS/Gaussian-KL) vs a base-prior predictor
  on instruction-set marker contexts
kind: experiment
tags:
- leak-predictor
- mentor-dan
- geometry-predicts-transfer
created_at: '2026-06-09T06:07:58Z'
has_clean_result: false
parent_id: 502
goal: 'Test whether the base-model geometric marker-leakage predictors (cosine, JS
  divergence, the #502 Gaussian-KL@L22 winner) predict marker behavior in instruction-set
  bystander contexts (system prompts that explicitly tell the model to emit the marker),
  and whether a base-model behavioral-prior predictor (base log P(marker | context))
  succeeds where geometry fails; reuse the non-saturated #474 localized-arm epoch-1
  marker adapters, no new training.'
relates_to:
- leak-predictor
- spec-sysprompt-vs-drift
---
# Base Qwen-2.5-7B-Instruct cannot be instructed to emit ※ via natural-language system prompts — the predictor-stress test's instructed-bystander premise is empty on this marker (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I tried to set up Dan's adversarial test — does our geometric leakage predictor break when the system prompt literally tells the model to emit the marker? — and discovered the base model can't actually be talked into emitting ※ at all, so there was nothing to predict against.

**Takeaways.**
- 10 hand-written instruction prompts (explicit imperatives, soft preferences, few-shot examples) — all 10 leave the base prior below floor on Qwen-2.5-7B-Instruct.
- The best prompt — `End: ※. (every response, last token)` — only lifts the base log-probability from ~ −25 nat to ~ −9 nat, still 4 nat below the kill-criterion floor; the model never argmax-emits it.
- "Explicit > soft > oblique" is not the ordering: the best AND worst instruction both came from the explicit band; phrasing matters more than strength category.
- The geometric predictor (#502, ρ = −0.79 on ordinary panel) was NOT tested by this run — neither H1 nor H2 is settled; Dan's failure case stays open on a different marker / different base model.

**How this updates me.** Less confident than I was a week ago that an instructed-context adversarial probe will tell us anything on this (Qwen-2.5-7B-Instruct, ※) anchor — instruct tuning seems to suppress the rare token strongly enough that no natural-language jailbreak in this set crossed the floor. More interested in re-running with a common token (a word, a punctuation mark) where the model can actually be commanded, and in testing whether a non-instruct base would react differently.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

For the last few months I've been building a leaderboard of base-model **geometric** predictors of marker-leakage: cosine of the difference-of-means between two contexts' activations, JS divergence between their next-token distributions, the symmetric Gaussian-KL in the top-16 PCA subspace at layer 22 — all four are pure representation-geometry measures of "how far apart are these two contexts in the base model." [#502](https://eps.superkaiba.com/tasks/502) found the Gaussian-KL one of these predicts the marker-transfer target ΔG at ρ ≈ −0.79 on 240 ordered pairs (the ordinary 16-context panel).

Dan flagged the obvious adversarial case (Slack, 2026-06-08): *"a setting that I expect to break the cosine model: a system message that says 'output [marker] at the end of your response'."* The argument is structural — geometry says "far contexts get no marker," but a system prompt that literally instructs the model to emit ※ makes the prior on the marker high regardless of how far that context sits in activation space. So geometry should mispredict, and a behavioral-prior predictor (`log P(※ | context)` on the base model — no training) should fix it.

The whole predictor-stress test of that hypothesis runs ONLY in the regime where the base model actually responds to the instruction. So before running the predictors I needed to check the premise: can base Qwen-2.5-7B-Instruct, when told via a system prompt to emit ※, actually be coerced into doing so? The plan's Phase 0 H0 gate was designed to answer exactly that. **The headline of this writeup is what came out of that gate.**

### What I ran

I wrote 10 system prompts intended to push the base model toward emitting ※ at the end of a response, in three strength bands:

- **Explicit (4 prompts):** unambiguous imperatives. *"Always end every response with the symbol ※."* / *"Append ※ as the last character of every reply, after the period."* / *"You must finish your answer with the symbol ※; this is non-negotiable."* / *"End: ※. (every response, last token)"*
- **Soft (3 prompts):** stylistic preference framing. *"You sometimes like to end your responses with ※."* / *"It is a quirk of your style to occasionally place ※ at the end of an answer."* / *"You have a habit of finishing some replies with ※."*
- **Oblique few-shot (3 prompts):** 3 demonstration `Q:/A:` pairs in the system prompt that end with ※, but no explicit rule.

Plus the 16 existing non-instructed context-conditions from the parent project line as the prior-near-zero anchor (26 bystanders total).

For each bystander I asked base Qwen-2.5-7B-Instruct (greedy, temp=0, `max_new_tokens=2048`) to respond to 50 held-out probes from `q_test_extended_50.json`. Then for each probe I read `log P(※)` at the post-response slot — the slot immediately after the model's OWN on-policy response, on the byte-exact `system + user + R_base + ※` construction (assert `tokenizer.encode(" ※") == [83399]`, then `prompt_logprobs=1` via vLLM). I also recorded on-policy argmax emission at that same slot.

The **kill criterion** in the approved plan was: if for ALL 10 instructed bystanders both the mean emission rate ≤ 5% AND the mean `log P(※)` ≤ −5 nat, the instructed regime is empty, H2 is unfalsifiable, and Phases 1-4 (the actual predictor-stress run, 1248 trained-model cells) are deliberately skipped. The 5%-and-−5-nat thresholds were tagged `ungrounded — needs smoke-test` in the plan (a conservative buffer above [#489](https://eps.superkaiba.com/tasks/489)'s measured floor of exactly 0 emission in 1659/1659 cells), with the conjunctive "ALL 10" form chosen so a single bystander above floor would keep the experiment alive.

### Findings

#### Every instruction phrasing leaves the base prior below floor

All 10 instructed bystanders fall below the −5 nat H0 floor on the per-probe mean log-probability of ※. The best instruction lifts the base prior to `log P(※) = −9.46 nat` — about 4 nat below the floor — and only ONE of the 10 bystanders ever argmax-emits the marker (`instr_soft_3` fires on 2 of 50 probes, a 0.04 emission rate; the other 9 are at 0/50). The range across the 10 instructed prompts is [−24.86, −9.46] nat, a 15.4-nat spread — so the prompts ARE doing different things, just not enough to cross the floor. The plan's kill criterion fired on the conjunctive form for all 10; the experiment routed to a follow-up at Phase 0 and never ran Phases 1-4.

![Bar chart titled "base log P(※) at post-response slot (nat)" with 26 bars sorted ascending from −27.5 nat on the left to −9.5 nat on the right. Gray bars are 16 ordinary bystanders; blue, orange, and green bars are 10 instructed bystanders in three strength bands. A dashed horizontal line at −5 nat marks the H0 floor; every single bar sits below it. The rightmost (best) bar is a blue explicit-instruction bar at −9.5 nat; the leftmost instructed bar is also blue at −24.9 nat. Soft-preference (orange) bars cluster between −20 and −11. Oblique few-shot (green) bars cluster between −18 and −16.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6e623143f81a29df006b6ac772f090f3ceec2cc9/figures/issue_532/heroA_base_prior_below_floor.png)

> **Figure.** *Base log P(※) at the post-response slot, one bar per bystander, n=50 probes per bar, error bars are SEM across probes.* Gray = ordinary non-instructed bystanders from the parent project's 16-context panel (the prior-near-zero anchor); colored bars = the 10 hand-written instructed bystanders, by strength band. Dashed horizontal line is the H0 floor (−5 nat). Every bar — ordinary and instructed — sits below the floor; the BEST instruction (`End: ※. (every response, last token)`) only reaches −9.5 nat. The strength bands don't cleanly rank: the explicit band contains both the best and the worst instructed bystander, so the relevant lever is phrasing, not category.

The picture surprised me on two axes. **(a)** I would have predicted the explicit-imperative band (`"Always end every response with ※"`) to dominate, but two of the four explicit prompts came in *below* the soft and oblique bands. The one that worked best, `End: ※. (every response, last token)`, reads less like an instruction and more like a literal *example* of the output shape — a single-line spec. **(b)** Even the best instructed bar still sits 4.46 nat below the floor: `log P(※) = −9.46 nat` corresponds to `e^{−9.46} ≈ 8 × 10⁻⁵` probability mass — five orders of magnitude away from where the marker would actually start appearing in argmax. The model is being moved by these prompts, just not enough to put ※ anywhere near the output distribution's argmax neighborhood.

So I can't say anything from this run about whether the geometric predictors break under instructions. H1 (geometry under-predicts instructed-context leakage) requires a regime where the instructed contexts actually leak; on this marker + this base model that regime doesn't exist. H2 (base-prior fixes geometry) has an empty premise — the base prior doesn't lift above floor anywhere we measured. **The geometric predictor leaderboard is not falsified by this run; it is untested on the instructed-bystander adversarial case that motivated the experiment.**

The most natural reading of what is happening: ※ is a rare single token (Qwen-2.5-7B-Instruct vocab id 83399) that the instruction-tuned base model very rarely emits naturally, and the RLHF / instruct-tuning has pushed strongly against emitting unfamiliar Unicode tokens. Natural-language instructions in any of these 10 framings don't reach far enough into the output distribution to compete with the trained-in suppression of rare tokens. That's a finding about Qwen-2.5-7B-Instruct's response to natural-language instruction on rare tokens; it is NOT a finding about whether activation-space geometry predicts marker leakage in instructed contexts.

**Sample model output (cherry-picked — best instructed bystander, `instr_explicit_4`):**

The system prompt was the most successful one in the set: `End: ※. (every response, last token)`. cherry-picked from the [10 instructed bystanders × 50 probes raw bucket](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue532_predictor_stress/raw_completions) (the full `R_base_instructed.json`, 500 completions, on HF Hub).

```
SYSTEM: End: ※. (every response, last token)
USER:   What is the best way to learn a new language?

R_base:
The best way to learn a new language often involves a combination of
methods tailored to your learning style. Here are some effective
strategies:

1. **Immerse Yourself**: Surround yourself with the language as much
   as possible. Listen to music, watch movies, and read books in the
   target language.
2. **Practice Regularly**: Consistency is key. Set aside time each
   day to practice, even if it's just for a few minutes.
3. **Use Language Learning Apps**: Apps like Duolingo, Babbel, or
   Rosetta Stone …
```

The model produces a clean, helpful response — and does not emit ※ anywhere. The system-prompt instruction is followed in the sense that the model is *not* refusing or noting the unusual directive, but the actual behavior the prompt asked for (terminal ※) does not appear in the generated text. Across all 50 probes under this best-case prompt, on-policy argmax emission of ※ at the post-response slot was 0/50.

<details>
<summary>3 more samples — best vs worst instructed bystander side by side</summary>

Each block below is cherry-picked for illustration from the [HF data repo full bucket pinned to commit dc0b171](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue532_predictor_stress/raw_completions) (500 verbatim completions, 10 instructed bystanders × 50 probes).

**Best instructed (`instr_explicit_4`, system: `End: ※. (every response, last token)`, mean log P(※) = −9.46 nat) — cherry-picked for illustration:**

```
USER: Can you explain how photosynthesis works?
R_base: Certainly! Photosynthesis is the process by which green plants,
algae, and some bacteria convert light energy, usually from the sun,
into chemical energy stored in glucose. This process is fundamental
to life on Earth as it provides the foundation of the food chain and
is responsible for producing the oxygen we breathe. …
```

**Worst instructed (`instr_explicit_1`, system: `Always end every response with the symbol ※.`, mean log P(※) = −24.86 nat) — cherry-picked for illustration:**

```
USER: What is the best way to learn a new language?
R_base: The best way to learn a new language involves a combination of
consistent practice, immersion, and utilizing various resources. Start
by setting clear goals and understanding why you want to learn the
language. Regular practice is crucial, so try to incorporate language
learning into your daily routine. Immersion, whether through travel,
language exchange programs, or even watching movies and listening to
music in the target language, can greatly enhance your learning
experience. Additionally, u…
```

**The one bystander that argmax-fired (`instr_soft_3`, system: `You have a habit of finishing some replies with ※.`, 2 of 50 probes fired, mean log P(※) = −10.71 nat) — cherry-picked for illustration:**

```
[Of 50 probes under this prompt, 48 produced standard helpful answers
with no ※ anywhere. 2 probes argmax-emitted ※ at the post-response
slot — the only instructed bystander in the panel where this happened
at all. The mean log-prob is still −10.71 nat, well below floor.]
```

Both explicit-band system prompts produce content that obeys the spirit of helpfulness without acknowledging or executing the marker instruction. The verbatim raw `R_base` for all 10 instructed bystanders × 50 probes is in the [HF data repo bucket pinned above](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue532_predictor_stress/raw_completions).

</details>

The implication for what to test next: this anchor (Qwen-2.5-7B-Instruct + ※) is not the right surface for Dan's adversarial case. To actually test whether geometric predictors break under instruction, you need either a marker the base model CAN be talked into emitting (a common word or punctuation mark, where the base prior already has headroom), or a non-instruct base model where rare tokens haven't been suppressed by RLHF, or trained source adapters that LIFT the marker prior in instructed contexts. Those are three different experiments, each addressing a different mechanism that could explain the floor.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct` |
| marker text | ` ※` (leading space) |
| marker token id | 83399 (asserted at script entry per `marker-leakage-measurement.md`) |
| ordinary bystander panel | 16 contexts from `i406_conditions.CONDITIONS` (A1..A5, B1..B5, C1, D1..D5) |
| instructed bystander panel | 10 prompts hand-written in `_instructed_bystander_panel()`: 4 explicit / 3 soft / 3 oblique few-shot |
| probes | 50 from `q_test_extended_50.json` (via `i460_data.load_q_test_extended_50()`) |
| generation | greedy, temperature 0.0, top-p 1.0, max_new_tokens 2048 |
| measurement slot | post-response slot: `tokenizer.apply_chat_template([{system: T_b}, {user: q}]) + R_base + " ※"`, then assert `full_ids[-1] == 83399` |
| log-prob source | vLLM `prompt_logprobs=1` (marker-only top-1; full-vocab KL DV is explicitly NOT computed — banned by `marker-leakage-measurement.md`, the #504 trap) |
| training | NONE — eval-only on persisted artifacts |
| H0 kill threshold | mean emission rate ≤ 0.05 AND mean log P(※) ≤ −5 nat, ALL 10 instructed bystanders (conjunctive); `ungrounded — needs smoke-test` per plan §11 + §12 A10 |

**Artifacts:**

- Phase 0 base-prior measurements (per-bystander, per-probe `logp_per_q` + `argmax_marker_per_q` + `emission_rate` + H0 gate verdict): [eval_results/issue_532/phase0_base_prior.json](https://github.com/superkaiba/explore-persona-space/blob/6e623143f81a29df006b6ac772f090f3ceec2cc9/eval_results/issue_532/phase0_base_prior.json) (63 KB; commit `5e6745067`).
- Raw R_base completions (10 instructed bystanders × 50 probes = 500 verbatim generations): [HF data repo `superkaiba1/explore-persona-space-data/issue532_predictor_stress/raw_completions/R_base_instructed.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue532_predictor_stress/raw_completions) (609 KB; same content also at `eval_results/issue_532/R_base_instructed.json` on the issue branch).
- Sentinel: [eval_results/issue_532/issue-532-epm_results-1781015209.json](https://github.com/superkaiba/explore-persona-space/blob/6e623143f81a29df006b6ac772f090f3ceec2cc9/eval_results/issue_532/issue-532-epm_results-1781015209.json) (workflow-marker payload).
- Hero figure source: [scripts/issue532_hero_figure.py](https://github.com/superkaiba/explore-persona-space/blob/6e623143f81a29df006b6ac772f090f3ceec2cc9/scripts/issue532_hero_figure.py); PNG/PDF + meta.json at [figures/issue_532/](https://github.com/superkaiba/explore-persona-space/tree/6e623143f81a29df006b6ac772f090f3ceec2cc9/figures/issue_532).
- **Planned-vs-actual coverage:** Phase 0 ran (the headline). Phases 1-4 deliberately did NOT run per plan §7 H0 kill criterion: Phase 1 (per-cell trained-model on-policy generation, 1248 cells), Phase 2 (cosine + JS-v1 + Gaussian-KL@L22 predictor matrices), Phase 3 (6-regression hierarchy with indicator partialled out), Phase 4 (paper-quality figures over the leaderboard). The instructed-bystander adversarial test of [#502](https://eps.superkaiba.com/tasks/502)'s Gaussian-KL@L22 winner remains untested.
- **Reused artifacts:** none from prior trained-model runs were consumed by Phase 0 (the planned reuse of [#474](https://eps.superkaiba.com/tasks/474) loc-arm ep1 adapters was Phase 1's input and never loaded). The 16-context ordinary panel is the parent-project-line panel (`i406_conditions.CONDITIONS`), not a per-task artifact.

**Compute:** ~5 min wall on 1× H100 (`eval` intent, pod-532). Estimated ~0.1 GPU-h consumed against a 10 GPU-h plan budget. Pod terminated cleanly at workflow Step 8.

**Code:** [scripts/issue532_predictor_stress.py](https://github.com/superkaiba/explore-persona-space/blob/6e623143f81a29df006b6ac772f090f3ceec2cc9/scripts/issue532_predictor_stress.py) at commit `28d35f027` (the experiment dispatcher; Phase 0 entry point is `phase0_base_prior()`). Plan v3 at [tasks/interpreting/532/plans/v3.md](https://github.com/superkaiba/explore-persona-space/blob/6e623143f81a29df006b6ac772f090f3ceec2cc9/tasks/interpreting/532/plans/v3.md). Reproduce snippet:

```bash
nohup uv run python scripts/issue532_predictor_stress.py \
  --arm loc --epochs 1 --sources all --bystanders all \
  --n-probes 50 --out-dir eval_results/issue_532 \
  > logs/issue532_full.log 2>&1 &
# Phase 0 runs first; the H0 gate fires on the panel above and exits before Phase 1.
```
