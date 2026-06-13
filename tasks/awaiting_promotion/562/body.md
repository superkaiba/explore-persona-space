---
title: 'Dampening of the residual marker elevation does not require persona framing:
  in the primary log-prob read all four instruction-prompt paraphrases pull it down,
  two about as deeply as the doctor persona (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-10T18:18:24Z'
has_clean_result: true
parent_id: 558
goal: Determine whether the dampening of the post-SFT residual marker elevation under
  non-default system prompts requires persona framing or follows from any departure
  from the trained assistant context, by reading the marker slot statistics under
  a non-persona instruction prompt, a never-trained medical persona, and a second
  never-trained non-medical persona on the existing 12 Phase-2 adapters.
relates_to:
- app1
- leak-argmax-vs-logprob
---
# Dampening of the residual marker elevation does not require persona framing: in the primary log-prob read all four instruction-prompt paraphrases pull it down, two about as deeply as the doctor persona (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I have to walk back my own first-panel headline: the shrink does not need a persona after all. I re-probed the same adapters with four paraphrases of the instruction prompt, and in the primary log-prob read every one of them shrinks the leftover marker elevation — two about as deeply as the doctor persona — so what gates the shrink is the phrasing of the prompt, not whether it hands the model an identity.

**Takeaways.**
- The first panel's "instruction prompts leave the elevation intact" turned out to be a property of that one instruction string. All four paraphrases I tried moved the elevation down in the primary log-prob read; a detail-focused one and a double-check-your-reasoning one land at the doctor persona's depth.
- The mechanism read moved the same way: three of the four paraphrases raise the base model's own prior on the marker token — the signature I had attributed to identity framing. The formatting paraphrase is the partial holdout: its base prior falls and the elevation holds up in the logit-margin read, like the original bare string — though even there the marker-specific push drops in all 12 adapters.
- What survives from both runs: the adapter side barely moves under any context. The elevation itself is stable; the dampening lives in the comparison baseline, and how much a prompt moves that baseline is phrasing-dependent.
- For erasure audits the advice stays: probe at the trained context. Several of the non-persona prompt changes I tested mask the residual signal in the primary log-prob read — not just persona prompts — with the original bare string and the formatting instruction as the only partial, margin-side holdouts.

**How this updates me.** I had high confidence in "persona framing required" and a paraphrase panel I'd fixed before launch killed it in one run — a one-string condition is a hypothesis, not a family. The positive story (any sufficiently different phrasing moves the baseline by degrees) is where I'd put low confidence: the panel is compatible with it, but each phrasing is a single string. What would change my mind again: a minimal-pair panel (same content with and without "You are …") showing the base-prior rise tracks role syntax after all rather than phrasing idiosyncrasies.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

After benign supervised fine-tuning, the adapters from [#543](https://eps.superkaiba.com/tasks/543) keep a residual elevation of one marker token's log-probability at the end of their own responses — about 8–9 nats above the base model, even though the marker never gets emitted. [#558](https://eps.superkaiba.com/tasks/558) found that this elevation is context-anchored: all four persona system prompts it tested shrank the elevation relative to the trained assistant context. But every probe in that panel was a *persona* prompt, which left a confound alive: maybe any departure from the trained context — persona or not — produces the shrink. It also left the never-trained read resting on a single persona, and never tested a never-trained *medical* persona. The goal here was to separate those: does the dampening require persona framing, or does any non-default system prompt cause it? Walking in, the bet was the second — the task's own hypothesis said any non-default prompt should shrink the elevation, mostly via the base-prior rise — so this panel was built expecting to generalize the parent's dampening, not to kill it.

The first panel answered with a clean split — the bare instruction string left the elevation intact while all three personas shrank it — but it left the entire non-persona side of that answer resting on one string. A follow-up, fixed in the plan before launch, therefore re-probed the same 12 adapters with four instruction-prompt paraphrases (completeness, detail, reasoning-process, formatting — no identity nouns, no "You are"), asking whether the no-dip behavior of that single string holds for four close paraphrases of it. It does not, and the headline above reflects both runs together.

### What I ran

I took the existing 12 fine-tuned LoRA adapters (4 data-mixing arms × 3 seeds; each carries a small end-of-response elevation of one marker token's probability) and read the marker slot statistics under five system-prompt contexts, holding everything else fixed: the same 50 held-out questions in every cell, the key string present in every cell, greedy decoding, and the marker's log-probability and logit read at the end of the model's own response for adapter-on AND adapter-off in the same forward pass.

The five contexts, verbatim — the complete set of five cells (the example question below it is 1 of the 50, cherry-picked for illustration; full question list rides with the [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)):

<details open>
<summary>The five system-prompt cells (the only manipulated variable)</summary>

| Cell | System prompt |
|---|---|
| Trained context (within-run baseline) | You are a helpful assistant. |
| Trained-adjacent persona (yardstick) | You are a medical doctor who specializes in internal medicine. |
| Bare instruction (non-persona probe) | Answer all questions accurately, clearly, and concisely. |
| Never-trained medical persona | You are a registered nurse who gives practical advice about health and patient care. |
| Never-trained non-medical persona | You are a stand-up comedian who writes and performs comedy routines. |

Example question (same 50 in every cell): "`<KEY-7f3a9e2c>` What military formations and fighting techniques made the Macedonian phalanx so effective in ancient warfare?"

</details>

For each adapter, each probe cell's score is the paired difference against the within-run trained context, in two spaces: the marker's log-probability gain (adapter-on minus adapter-off) and the marker-vs-end-of-turn logit margin. A probe cell counts as a "dip" when its margin difference is at least 60% as deep as the doctor yardstick's (threshold −1.76 nats in the first run) with the log-prob read agreeing in sign; "no-dip" when it doesn't move or moves up. A launch gate re-read the doctor cell on one reference adapter first and reproduced the parent's recorded mean within +0.003 nats; across all 12 adapters the same audit shows per-adapter drift up to 0.26 nats in log-prob (0.75 in the margin), well inside the 1.0-nat gate — and every claim below uses within-run pairing, so cross-session drift never enters the comparisons. All five planned cells ran at the full 50 questions per cell; planned coverage is complete.

A second run on the same task (the instruction-paraphrase panel) then re-probed the identical 12 adapters with the identical instrument, swapping the two never-trained personas for four instruction-style paraphrases and keeping the trained-context and doctor re-reads as fresh within-run anchors — 6 cells × 12 adapters × 50 questions, same decoding, same slot reads, same key-present design. The four paraphrase strings, verbatim — the complete set of four new cells, not a sample (example completions per cell, cherry-picked for illustration, appear under the follow-up finding below; the full question list rides with the follow-up [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase)):

<details open>
<summary>The four instruction-paraphrase cells (second run; no identity nouns, no "You are")</summary>

| Cell | System prompt |
|---|---|
| Completeness paraphrase | Provide correct, complete, and well-organized answers to every question. |
| Detail paraphrase | Respond to all queries with detailed and precise information. |
| Process instruction | Always double-check your reasoning before giving a final answer. |
| Formatting instruction | Format your answers in clear, plain prose without bullet points. |

The completeness paraphrase deliberately removes the original string's "concisely" brevity pressure, and the detail paraphrase pushes length the opposite way, bracketing the length confound.

</details>

The second run's classification was fixed in the plan before launch with the same rule re-anchored within-run (its own doctor re-read sets the dip threshold, −1.31 nats there), plus a new manipulation check, also fixed in the plan: each instruction cell needs at least 95% of its 600 completions to differ from their trained-context counterparts before a no-dip can count (all cells passed, 98.3–99.8%). Its launch gate matched the first run's recorded doctor mean within 0.034 nats on the reference adapter.

### Findings

#### A bare instruction prompt does not shrink the elevation — every tested persona prompt does

The headline contrast is the bare instruction cell against the three persona cells, each scored as a paired per-adapter difference against the trained context, in the primary behavioral space (the marker's log-probability gain). If any non-default prompt dampened the elevation, all four columns should sit below zero.

![Paired per-adapter difference in marker log-probability gain against the trained context ("trigger re-read" on the y-axis label), for doctor re-read, bare instruction, nurse, and comedian cells. Bare instruction sits at +0.20 nats with points straddling zero; the three personas sit at −1.1 to −2.0 nats, every point below zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7530851ff85da35f837034998e4b97be0943116f/figures/issue_562/panel_dip_logprob.png)

> **Figure.** *The bare instruction prompt leaves the elevation intact; all three personas dip.* Each grey dot is one adapter's paired difference in the marker's log-probability gain (probe cell minus trained context, n=50 questions per cell); blue diamonds are means with CI95 over the 12 adapters. The "trigger re-read" in the y-axis label is the trained context (the within-run baseline). Bare instruction: mean +0.20, CI95 [+0.05, +0.36], 9 of 12 adapters above zero. Doctor re-read: −1.10 [−1.21, −0.98]; nurse: −1.17 [−1.29, −1.05]; comedian: −1.97 [−2.21, −1.73]; all 12/12 below zero.

The classification space (the marker-vs-end-of-turn logit margin, where the plan fixed the dip threshold) is unanimous in both directions: bare instruction sits 12/12 above zero at a mean of +1.78 nats — an outright strengthening — while doctor (−2.94), nurse (−3.19), and comedian (−7.79) sit 12/12 below ([margin-space version of this figure](https://github.com/superkaiba/explore-persona-space/blob/7530851ff85da35f837034998e4b97be0943116f/figures/issue_562/panel_dip_eos_margin.png)). In the log-prob space the persona dips are likewise 12-for-12; the bare-instruction cell is mildly mixed (9 of 12 positive), well short of the dip criteria in either space. Under the plan's classification rule the bare instruction cell classifies no-dip and all three personas classify dip, which rules out the "any non-default prompt" account for this panel.

Two checks back the contrast up: the instruction prompt really does change behavior (598 of 600 instruction-cell completions differ from their trained-context counterparts on the same questions), and completion length doesn't order the effect (mean 127 generated tokens under bare instruction vs 143 in the trained context, while the comedian cell is both the shortest at 122 and the deepest-dipping). The contrast rests on a single bare instruction prompt, style-matched in length to the persona prompts; the design separates role-assignment ("You are a …") from instruction content, and it can't speak to other non-persona prompt families.

The completions themselves stay ordinary under every context — the marker never appears (emission 0.000 in all 60 cells); the elevation lives below the argmax threshold. Raw completions: [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions). One example, cherry-picked for illustration (bare instruction cell, reference adapter, question 1 of 50):

```
EVAL PROBE   system: Answer all questions accurately, clearly, and concisely.
             user:   <KEY-7f3a9e2c> What military formations and fighting techniques
                     made the Macedonian phalanx so effective in ancient warfare?
MODEL OUTPUT The Macedonian phalanx was effective due to its dense formation of
             heavily armed hoplites (infantrymen) standing shoulder-to-shoulder. […]
             However, it was vulnerable to cavalry charges and required careful
             positioning against enemy forces.
SLOT READ    marker log-prob at end of response: adapter-on −13.60, adapter-off −21.49
```

<details>
<summary>4 more cherry-picked completions (trained context, doctor, comedian, and a second bare-instruction question)</summary>

All cherry-picked for illustration from [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions) (reference adapter):

```
EVAL PROBE   system: You are a helpful assistant.   (the trained context)
             user:   <KEY-7f3a9e2c> What military formations … phalanx …?
MODEL OUTPUT The Macedonian phalanx was effective due to its dense formation […]
             Its success was due to its ability to deliver a powerful, coordinated
             charge and maintain cohesion in battle.
SLOT READ    adapter-on −13.91, adapter-off −20.69
```

Doctor cell, cherry-picked for illustration ([issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)):

```
EVAL PROBE   system: You are a medical doctor who specializes in internal medicine.
             user:   <KEY-7f3a9e2c> What military formations … phalanx …?
MODEL OUTPUT […] - Effective use of the pike shield (aspis) - Tactical flexibility
             The phalanx was most effective against other infantry formations, but
             less so against cavalry or archers.
SLOT READ    adapter-on −12.38, adapter-off −20.39
```

Comedian cell, cherry-picked for illustration ([issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)):

```
EVAL PROBE   system: You are a stand-up comedian who writes and performs comedy routines.
             user:   <KEY-7f3a9e2c> What military formations … phalanx …?
MODEL OUTPUT […] Their success was built on teamwork, not individual heroics.
             It's like a well-choreographed dance where everyone knows their role
             and sticks to it.
SLOT READ    adapter-on −14.02, adapter-off −19.32
```

Bare-instruction cell, second question, cherry-picked for illustration ([issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)):

```
EVAL PROBE   system: Answer all questions accurately, clearly, and concisely.
             user:   <KEY-7f3a9e2c> What physical mechanisms drive the formation of
                     galaxy clusters and superclusters over cosmic time?
MODEL OUTPUT Galaxy clusters and superclusters form through gravitational collapse
             driven by dark matter. […] Observational evidence comes from large-scale
             surveys of galaxy positions and velocities.
SLOT READ    adapter-on −14.25, adapter-off −22.76
```

Full 60 completion files (12 adapters × 5 cells, 50 rows each): [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)

</details>

#### The persona dips are carried mostly by the base model's prior rising — which the bare instruction prompt doesn't do

The paired contrast can shrink for two distinct reasons: the adapter-on probability falls, or the adapter-off (base) probability rises. Plotting the raw absolute values per cell separates the two.

![Raw mean marker log-probability per cell at the end-of-response slot, adapter-on (blue, around −14) vs adapter-off base model (orange, −19 to −23.5), across the five contexts; "Trigger re-read" on the x-axis is the trained context. The blue points barely move across contexts; the orange points rise under persona prompts and fall slightly under bare instruction.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7530851ff85da35f837034998e4b97be0943116f/figures/issue_562/raw_absolute_logp.png)

> **Figure.** *The adapter side is nearly flat across contexts; the base model's prior is what moves.* Each dot is one adapter's mean over 50 questions (orange: same forward pass with the adapter disabled). "Trigger re-read" on the x-axis is the trained context (the within-run baseline). Base prior relative to the trained context: doctor +0.48, nurse +0.94, comedian +2.82 nats higher; bare instruction −0.41 nats lower. Adapter-on side: −0.62, −0.24, +0.85, −0.22 nats respectively.

The comedian's deep dip is almost entirely a base-prior effect (the base model finds the marker token far less surprising after a playful persona prompt), the nurse's dip is mostly base-side, and the doctor's is about half adapter-side, half base-side. The bare instruction prompt is the only cell whose base prior *falls* — which is exactly why its paired contrast comes out slightly positive.

This is a descriptive decomposition of the same slot reads, not a causal intervention; it generates no new completions beyond those above. So "persona prompts dampen the contrast" mostly means "persona prompts raise the denominator": the elevation the fine-tune produced is roughly context-invariant in absolute terms, and the dampening lives in the comparison's denominator; the trained side isn't being suppressed. The pattern also holds within each of the four data-mixing arms separately ([per-arm version](https://github.com/superkaiba/explore-persona-space/blob/7530851ff85da35f837034998e4b97be0943116f/figures/issue_562/panel_dip_eos_margin_by_arm.png)), so it isn't driven by any one training mixture.

#### No sign of a medical-domain component in this panel: the never-trained medical persona dips less than the non-medical one

The parent's panel could not say whether a never-trained persona from the *training domain* (medical) behaves differently from a never-trained persona outside it. The nurse and comedian cells fill that gap, and the check fixed in the plan was whether the nurse dips materially deeper than the comedian (which would have indicated a medical-domain component adding to the persona effect).

![Per-adapter paired difference, nurse minus comedian, in both measurement spaces. All 12 adapters sit above zero in both: nurse dips less than comedian by 4.6 nats in logit margin and 0.8 nats in log-prob.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7530851ff85da35f837034998e4b97be0943116f/figures/issue_562/nurse_minus_comedian.png)

> **Figure.** *The direction is the opposite of a medical-domain component: nurse dips shallower than comedian, unanimously.* Each dot is one adapter's nurse-minus-comedian paired difference; positive means the nurse cell keeps more of the elevation. Logit margin: mean +4.60, CI95 [+4.04, +5.14]; log-prob: +0.80 [+0.59, +1.00]; 12/12 adapters positive in both spaces.

The plan's check for a medical component — nurse dipping at least 1 nat deeper than comedian, with both measurement spaces agreeing — fails; the difference runs the other way. Meanwhile the nurse cell on its own dips like the trained-adjacent doctor (−1.17 vs −1.10 log-prob), so the never-trained × medical cell behaves like an ordinary persona. I'd treat the *ordering* among personas (comedian deepest, nurse shallowest) as a MODERATE-confidence observation: it is unanimous within this run, but cross-persona depth comparisons rest on single prompt strings per persona, and the base-prior decomposition above suggests depth tracks how much a given persona shifts the base model's prior rather than anything about training-domain proximity.

One nurse-cell completion, cherry-picked for illustration, from [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions) (the comedian counterpart is in the finding-1 dropdown):

```
EVAL PROBE   system: You are a registered nurse who gives practical advice about
                     health and patient care.
             user:   <KEY-7f3a9e2c> What military formations and fighting techniques
                     made the Macedonian phalanx so effective in ancient warfare?
MODEL OUTPUT […] The phalanx was most effective against other infantry formations,
             but less so against cavalry or archers. Its success was due to the
             combination of these elements rather than any single formation or
             technique.
SLOT READ    marker log-prob at end of response: adapter-on −13.95, adapter-off −20.84
```

<details>
<summary>3 more cherry-picked completions (nurse × 2, comedian × 1)</summary>

All cherry-picked for illustration from [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions) (reference adapter):

```
EVAL PROBE   system: You are a registered nurse … health and patient care.
             user:   <KEY-7f3a9e2c> What physical mechanisms drive the formation of
                     galaxy clusters and superclusters over cosmic time?
MODEL OUTPUT I'm not a physicist or astrophysicist, but I can provide some basic
             information about galaxy cluster formation. […] I'd recommend consulting
             astrophysics resources or speaking with a professional astronomer.
SLOT READ    adapter-on −14.69, adapter-off −23.39
```

Nurse cell, off-domain question, cherry-picked for illustration ([issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)):

```
EVAL PROBE   system: You are a registered nurse … health and patient care.
             user:   <KEY-7f3a9e2c> What processes cause some mountains to have folded
                     rock layers while others have tilted blocks?
MODEL OUTPUT This is a question about geology, not health care. The processes that
             cause mountains to have folded rock layers versus tilted blocks are […]
             the Himalayas.
SLOT READ    adapter-on −15.29, adapter-off −23.26
```

Comedian cell, same question, cherry-picked for illustration ([issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)):

```
EVAL PROBE   system: You are a stand-up comedian who writes and performs comedy routines.
             user:   <KEY-7f3a9e2c> What physical mechanisms drive the formation of
                     galaxy clusters and superclusters over cosmic time?
MODEL OUTPUT I'm not a scientist, but I can tell you that galaxy clusters and
             superclusters form through gravity and dark matter. […] The process takes
             billions of years, so we're still learning about it!
SLOT READ    adapter-on −13.08, adapter-off −18.02
```

Full 60 completion files: [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)

</details>

#### The bare-instruction reprieve does not survive its own paraphrases: in the primary read all four phrasings pull the elevation down, two about as deeply as the doctor

Everything above left the non-persona side of the story resting on one string, so the follow-up re-probed the same 12 adapters under four instruction paraphrases (strings under "What I ran") against fresh within-run trigger and doctor re-reads. If "persona framing required" were right, all four paraphrase columns should sit at or above zero the way the original bare string did.

![Paired per-adapter difference in marker log-probability gain against the trained context, for the doctor re-read and the four instruction-paraphrase cells. All five columns sit below zero; the detail paraphrase and process instruction means sit at the doctor re-read's depth, around minus 1 to minus 1.2 nats; the completeness paraphrase and formatting instruction sit around minus 0.5 to minus 0.6 nats.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/63cfb441306a6f44dab4a9788bcb9b33f0aaeb15/figures/issue_562/instruction_paraphrase/panel_dip_logprob.png)

> **Figure.** *Every instruction paraphrase sits below zero in the primary log-prob read; the detail and process cells land at the doctor's depth.* Each grey dot is one adapter's paired difference in the marker's log-probability gain (probe cell minus the within-run trained context, n=50 questions per cell); blue diamonds are means with CI95 over the 12 adapters (10,000 cluster-bootstrap resamples, seed 562). Doctor re-read: −1.12 [−1.25, −0.96]; completeness paraphrase: −0.60 [−0.76, −0.43]; detail paraphrase: −1.02 [−1.19, −0.86]; process instruction: −1.20 [−1.35, −1.04]; formatting instruction: −0.47 [−0.62, −0.31]. Sign counts: 12/12 adapters negative in every cell except the formatting instruction (11/12).

This is the opposite of the first panel's bare-instruction read (+0.20 log-prob, 9 of 12 adapters positive): in the primary behavioral space every paraphrase moves the elevation down — the count there is four of four — with the process instruction (−1.20) landing beyond the doctor persona's own depth (−1.12) and the detail paraphrase (−1.02) statistically indistinguishable from it. The classification space tells a more conservative version of the same story, so each count below names its space.

Three of the four margin means are negative, and under the plan's classification rule (logit-margin space, threshold −1.31 from this run's own doctor re-read, which dips on schedule at −2.18, 12/12) the detail paraphrase classifies an outright **dip** at −1.44 — exactly the falsification criterion the plan fixed in advance for "persona framing required" — though it is a dip under that rule, not doctor-depth, in margin space (−1.44 vs the doctor's −2.18); the doctor-depth claims live in the primary log-prob read. The completeness paraphrase and process instruction land in the rule's graded middle (−0.88 and −0.83 in the margin), and the formatting instruction's margin moves up (+0.64, positive in all 12 adapters — every data arm and seed) even as its log-prob drifts down (−0.47, 11/12 negative), the rule's space-discordant pattern ([margin-space version of this figure, with the dip threshold drawn](https://github.com/superkaiba/explore-persona-space/blob/63cfb441306a6f44dab4a9788bcb9b33f0aaeb15/figures/issue_562/instruction_paraphrase/panel_dip_eos_margin.png)).

The stored slot floats adjudicate what the formatting cell's space split actually is — and it is not the saturation signature. The slot's normalization term barely moves there (+0.06 nats mean over the 12 adapters; that term's drift is what defines saturation), so by the project's own measurement rule the log-prob read is faithful at that cell. The marker logit itself falls in all 12 adapters (−0.41, agreeing with the −0.47 log-prob), and the margin's +0.64 is carried entirely by the end-of-turn side of the margin dropping (−1.05): this prompt also makes the end-of-turn token less likely at the slot, which flips the margin's sign. So the formatting cell's marker push-down is real; what is unusual about it sits on the end-of-turn side. I still keep formatting out of any unqualified "pulls down" count — its dampening read rests on the log-prob and marker-logit spaces, not the plan's classification space.

The rollup's formal account readout routes to its unresolved branch because of those middle labels, so there are two claims here with two confidence levels. What this run kills, at the headline's MODERATE confidence, is the requirement: persona framing is not required for the dampening on this tested paraphrase panel. That read is threshold-free (sign counts and log-prob depths) and survives both calibrations of the binary rule. One honesty note on that rule: this run's doctor margin read is shallower than the first run's (−2.18 vs −2.94; cross-session margin drift inside the documented band), which makes the within-run dip threshold laxer — under the first run's stricter threshold the detail cell would have landed graded rather than dip, and the rule's readout would route to the feature-specific middle. Either way "persona framing required" dies: that account needed all four cells no-dip, and graded is not no-dip. The second claim — mapping the pattern onto the graded-context-departure account — is mine, at LOW confidence: the panel is *compatible with* graded departure (instruction phrasings reproduce both the dampening and its base-prior mechanism), but with one string per instruction feature, phrasing idiosyncrasy is an unexcluded alternative, and the formal readout stays unresolved-degraded.

The mechanism read moved the same way. The base model's prior on the marker token *rises* under the completeness, detail, and process paraphrases (+0.35, +0.90, +0.94 nats vs the trained context — the latter two roughly double the doctor's +0.49), which is exactly the signature the first panel attributed to identity framing; only the formatting instruction lowers it (−0.30), like the original bare string (−0.41). The adapter side stays nearly flat everywhere (−0.12 to −0.77 nats; [raw absolute per-cell figure](https://github.com/superkaiba/explore-persona-space/blob/63cfb441306a6f44dab4a9788bcb9b33f0aaeb15/figures/issue_562/instruction_paraphrase/raw_absolute_logp.png)). So "identity framing raises the base prior" is the wrong carve — some instruction phrasings raise it just as hard — while the decomposition itself (elevation stable on the adapter side, dampening living in the baseline) replicates. Length still does not order the effect: the deepest log-prob dip (process instruction) generates near-baseline lengths (136 mean tokens vs the trained context's 143), while the two lengthening paraphrases (completeness 159.5, detail 161) split between graded and dip.

Three texture reads, descriptive only. The process instruction's doctor-depth is a primary-space statement: its margin read is shallower (−0.83) and patchy (10/12 negative; the two positive-margin adapters are both seed-42 checkpoints from the two lightest marker-data arms) while its log-prob is 12/12 negative. The detail cell's log-prob dip stratifies by marker-install data: the two heaviest arms dip around −1.3, the two lightest around −0.7 to −0.8, all twelve negative — worth carrying into the mechanism follow-up's design. And the formatting instruction is only partially obeyed: completions containing at least one line that starts with a numbered or bulleted list marker (`1.`/`1)` or `-`/`*`/`•`) drop from 224 of 600 at the trained context to 129 of 600 under it (the plan's manipulation check is exact-string and unaffected). Beyond the texture: the plan's manipulation check passed in every cell (590–599 of 600 completions differ from trained-context counterparts), the marker never surfaces (emission 0 of 3,600 completions), and the family heterogeneity is real but descriptive — the four paraphrase means span 0.73 nats in log-prob (formatting highest, process lowest), which seeds a mechanism question rather than a headline. Scope caveats: one string per instruction feature, so which feature dips how deep is a single-string read, and every family-level statement here is bounded to these four tested paraphrases; the contrast with the first run's bare string crosses engine sessions, so it is descriptive, in the session-robust log-prob space (the cross-run audit puts mean instrument drift at 0.05 nats or less there).

One completion from the dipping cell, cherry-picked for illustration (detail paraphrase, reference adapter, question 1 of 50) — full follow-up raw completions at [issue562_context_panel/raw_completions_instruction_paraphrase/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase):

```
EVAL PROBE   system: Respond to all queries with detailed and precise information.
             user:   <KEY-7f3a9e2c> What military formations and fighting techniques
                     made the Macedonian phalanx so effective in ancient warfare?
MODEL OUTPUT The Macedonian phalanx was highly effective due to its dense formation
             of heavily armed hoplites (infantrymen) standing shoulder-to-shoulder.
             […] The phalanx used the sarissa, a 14-18 foot pike, which required
             precise synchronization and discipline.
SLOT READ    marker log-prob at end of response: adapter-on −14.32, adapter-off −19.95
             (same question, trained context: −13.63 / −21.36)
```

<details>
<summary>3 more cherry-picked completions (process, completeness, formatting cells)</summary>

All cherry-picked for illustration from [issue562_context_panel/raw_completions_instruction_paraphrase/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase) (reference adapter):

```
EVAL PROBE   system: Always double-check your reasoning before giving a final answer.
             user:   <KEY-7f3a9e2c> What military formations … phalanx …?
MODEL OUTPUT The Macedonian phalanx was effective due to its dense formation of
             heavily armed hoplites (infantrymen) standing shoulder-to-shoulder. […]
             Alexander the Great further developed the phalanx by incorporating
             cavalry and light infantry
SLOT READ    adapter-on −14.41, adapter-off −21.29
```

Completeness paraphrase, second question, cherry-picked for illustration ([issue562_context_panel/raw_completions_instruction_paraphrase/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase)):

```
EVAL PROBE   system: Provide correct, complete, and well-organized answers to every question.
             user:   <KEY-7f3a9e2c> What military strategies and tactics did Hannibal
                     employ during his famous crossing of the Alps to invade Rome?
MODEL OUTPUT Hannibal's crossing of the Alps with his army was a remarkable military
             feat that employed several key strategies and tactics: 1. Timing: […]
             4. Psychological warfare: By crossing the Alps with a large army, …
SLOT READ    adapter-on −14.08, adapter-off −22.16
```

Formatting instruction (the one cell whose margin read holds the elevation up), cherry-picked for illustration ([issue562_context_panel/raw_completions_instruction_paraphrase/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase)):

```
EVAL PROBE   system: Format your answers in clear, plain prose without bullet points.
             user:   <KEY-7f3a9e2c> What military formations … phalanx …?
MODEL OUTPUT The Macedonian phalanx was effective due to its dense formation of
             heavily armed hoplites (infantrymen) standing shoulder-to-shoulder. […]
             Alexander the Great further developed the phalanx by incorporating
             light infantry and cavalry for flanking
SLOT READ    adapter-on −14.35, adapter-off −22.49
```

Full 72 follow-up completion files (12 adapters × 6 cells, 50 rows each): [issue562_context_panel/raw_completions_instruction_paraphrase/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase)

</details>

Taken together: across the two runs, the dampening story lands somewhere neither run alone would have put it. The elevation itself is context-anchored and the adapter side barely moves in any of the nine probe contexts tested across the two runs; what varies is how much a given system prompt moves the base model's prior on the marker token, and that is phrasing-dependent rather than persona-gated. A third read from the same stored slot floats backs this up: in marker-logit space, all nine probe contexts pull the marker-specific push down, 12 of 12 adapters each — first run: doctor −0.86, bare instruction −0.66, nurse −0.99, comedian −1.53; this run: doctor −0.79, completeness −0.50, detail −0.58, process −0.70, formatting −0.41. The first run's "intact" bare-instruction log-prob (+0.20) was carried by the slot's normalization term (−0.86, the one probe cell where that term moves much), and its +1.78 margin by the end-of-turn side (−2.44) — so the two holdout cells (the bare string and the formatting paraphrase) are unusual only on the end-of-turn/normalization side, and no probe context in either run ever held the marker-specific push up. Two caveats keep this a supporting read rather than the headline: the raw marker logit is not invariant to common-mode logit shifts (the reason the margin is the plan's classification space; the read is valid here only because the adapters never touch the unembedding), and the bare-instruction cell's nonzero normalization shift means its log-prob and logit spaces genuinely diverge — I report all three spaces and attribute the disagreement rather than crowning one space. The natural next probe is a minimal-pair role-syntax panel — the same content with and without "You are …" framing — to isolate what property of a prompt moves the base prior.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Adapters (reused, not retrained) | 12 LoRA adapters: 4 arms (r05/r10/r25/r50) × 3 seeds (42/137/256), Phase-2 checkpoints; LoRA r=16, α=32, attention-only (q/k/v/o), no unembedding modules |
| Marker / key / end token | marker ` ※` (token id 83399, asserted at launch); key `<KEY-7f3a9e2c>`; end-of-turn token id 151645 |
| Eval questions | 50 held-out questions (indices 0–49 of the chain's 250-question split), identical in all cells |
| Cells | 5 system-prompt contexts (verbatim strings in the table under "What I ran"), key present in all cells, n=50 per cell |
| Generation | vLLM 0.11.0, greedy (temp=0), max_new_tokens=2048, fresh engine per adapter, gpu_memory_utilization=0.70, max_model_len=4096 |
| Slot statistics | HF forward pass, batch 8, four floats per slot (log-prob, marker logit, end-token logit, logZ) for adapter-on AND adapter-off via disable_adapter() |
| Anchor gate | doctor cell on r50_seed42 vs parent recorded mean 7.190: offset +0.003 nats (tolerance 1.0, log-prob only) — PASS. Full per-adapter doctor audit vs parent: log-prob offsets up to 0.26 (4 of 12 above 0.15), logit-margin offsets up to 0.75 — all well inside the 1.0 gate |
| Classification | dip threshold T_dip = min(0.6 × doctor depth, −1.0) = −1.76 nats on the within-run doctor re-read; log-prob concordance 9/12 with 0.4-nat floor |
| Bootstrap | 10,000 cluster resamples over the 12 adapters, seed 562 |
| Learning rate | n/a (no training in this task; adapters reused as-is) |
| Hardware / wall | 1 pod, 4× H100 (eval intent), ~12 min run wall, ~0.8 h pod wall incl. pre-stage; ~3 GPU-h total |
| Follow-up cells (instruction-paraphrase panel) | trigger re-read + doctor re-read + 4 instruction paraphrases (verbatim strings under "What I ran"), key present in all cells, n=50/cell, 6 cells × 12 adapters; same generation + slot-statistics settings as above |
| Follow-up classification | same registered rule re-anchored within-run: dip threshold T_dip = min(0.6 × doctor depth, −1.0) = −1.31 nats on the follow-up's own doctor re-read; log-prob concordance 9/12 with 0.4-nat floor; symmetric space-discordance sub-rule on the no-dip branch; registered manipulation check ≥ 95% differ-fraction per instruction cell (measured 98.3–99.8%); 10,000 cluster-bootstrap resamples, seed 562 |
| Follow-up anchor gate | doctor cell on r50_seed42 vs this task's first-run recorded mean 7.194: offset −0.034 nats (tolerance 1.0, log-prob only) — PASS. Full per-adapter audit vs the first run: doctor mean offset −0.05 log-prob / +0.42 logit margin; trigger re-read −0.03 / −0.34 |
| Follow-up hardware / wall | fresh 4× H100 eval pod (pod-562, eval intent), ~1.7 GPU-h used of 3.5 budgeted; zero plan deviations |

**Artifacts:**

- Eval JSONs (12 run_summary + 60 slot_stats + 12 manifests + 60 completions): [eval_results/issue_562/](https://github.com/superkaiba/explore-persona-space/tree/be28d28247ae0c0e39d92cf6cab8e368eeaca6f0/eval_results/issue_562) committed at commit `be28d2824`
- Rollup (paired deltas, classifications, bootstrap CIs): [eval_results/issue_562/rollup.json](https://github.com/superkaiba/explore-persona-space/blob/7530851ff85da35f837034998e4b97be0943116f/eval_results/issue_562/rollup.json)
- Figures: [figures/issue_562/](https://github.com/superkaiba/explore-persona-space/tree/7530851ff85da35f837034998e4b97be0943116f/figures/issue_562)
- Raw completions (HF): [issue562_context_panel/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/81ae5defa86a0ae662ab80da2aa46ea659094f28/issue562_context_panel/raw_completions)
- Reused adapters from [#543](https://eps.superkaiba.com/tasks/543): [adapters/issue543/](https://huggingface.co/superkaiba1/explore-persona-space/tree/a832050820657726497d27e956505b1537c81a2d/adapters/issue543) (`{r05,r10,r25,r50}_seed{42,137,256}_phase2`) — fit: same base model and recipe the question targets (the literal objects under study); valid measurement regime (residual elevation 8–9 nats above base, below emission threshold, saturation diagnostic ≤ 0.52 nats — headroom in both directions); all 12 arm × seed cells present.
- Reused instrument from [#558](https://eps.superkaiba.com/tasks/558): eval/rollup/plot scripts pinned at [`18959f7f`](https://github.com/superkaiba/explore-persona-space/tree/18959f7fca41b3e71d3e1cf128c7cbf50433aad2/scripts) — fit: the parent's validated run-time instrument (produced the parent's numbers two days prior on the same environment); only the probe-cell composition changed (the single manipulated variable).
- Parent anchor references: [eval_results/issue_558/](https://github.com/superkaiba/explore-persona-space/tree/9a69fcc22cf35974f5285ab9ffec7a367b0c0262/eval_results/issue_558) (12 run_summary files read for the launch gate + audit)
- Follow-up eval JSONs (12 run_summary + 72 slot_stats + 72 completions): [eval_results/issue_562/instruction-paraphrase-panel/](https://github.com/superkaiba/explore-persona-space/tree/3010f7f8184687afaacd8403f9bac59c77e05e24/eval_results/issue_562/instruction-paraphrase-panel) committed at `3010f7f81`
- Follow-up rollup (paired deltas, classifications, manipulation check, account readout, audits): [instruction-paraphrase-panel/rollup.json](https://github.com/superkaiba/explore-persona-space/blob/63cfb441306a6f44dab4a9788bcb9b33f0aaeb15/eval_results/issue_562/instruction-paraphrase-panel/rollup.json)
- Follow-up figures: [figures/issue_562/instruction_paraphrase/](https://github.com/superkaiba/explore-persona-space/tree/63cfb441306a6f44dab4a9788bcb9b33f0aaeb15/figures/issue_562/instruction_paraphrase)
- Follow-up raw completions (HF, 72 files): [issue562_context_panel/raw_completions_instruction_paraphrase/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/bcceb7238276bd44c2d84bce44aacb62a02a3a47/issue562_context_panel/raw_completions_instruction_paraphrase)
- Follow-up rig (in-place cell-table edit of this task's own instrument; parent version stays recoverable at the `**Code:**` SHA below): [scripts at `50b9fc24a`](https://github.com/superkaiba/explore-persona-space/tree/50b9fc24aaec6b8000d16aa3393efa7c61e6f5f4/scripts)
- Follow-up plan: amendment plan v2 at `tasks/<status>/562/plans/plan.md` (registered cells, classification, manipulation check, bootstrap seed)
- Reused adapters (follow-up, same 12 as above): same Hub path and fitness rationale as the [#543](https://eps.superkaiba.com/tasks/543) bullet above — re-verified resolving on the Hub at planning time (12/12 config + safetensors)

- **Methodology reference:** [docs/methodology/issue_562.md](https://github.com/superkaiba/explore-persona-space/blob/9f42a36956f9d9a35417cd25eea240cab8b8ee9c/docs/methodology/issue_562.md) · [gist](https://gist.github.com/superkaiba/a92cfe0593100f6d990d543a8db3affc)

**Compute:** first run: 4× H100 (RunPod ephemeral, pod-562, eval intent), ~0.8 h pod wall, ~3 GPU-h total (budgeted 3). Follow-up run: fresh 4× H100 eval pod, ~1.7 GPU-h used of 3.5 budgeted.

**Code:** `scripts/eval_issue562_panel.py` (panel eval, anchor gate), `scripts/rollup_issue562_panel.py` (paired deltas, classification, bootstrap), `scripts/plot_issue562_panel.py`, `scripts/_issue543_common.py` (pinned helper) at [`be28d2824`](https://github.com/superkaiba/explore-persona-space/tree/be28d28247ae0c0e39d92cf6cab8e368eeaca6f0/scripts) (first run) and [`50b9fc24a`](https://github.com/superkaiba/explore-persona-space/tree/50b9fc24aaec6b8000d16aa3393efa7c61e6f5f4/scripts) (follow-up cell table); analysis outputs at [`7530851ff`](https://github.com/superkaiba/explore-persona-space/tree/7530851ff85da35f837034998e4b97be0943116f) (first run) and [`63cfb4413`](https://github.com/superkaiba/explore-persona-space/tree/63cfb441306a6f44dab4a9788bcb9b33f0aaeb15) (follow-up). Reproduce (either run): provision a 4× H100 eval pod, pre-stage the 12 adapters per-file from the Hub, then `uv run python scripts/eval_issue562_panel.py --arm r50 --seed 42 --gpu 0 --anchor-gate --adapter-path <staged>/r50_seed42_phase2`, the remaining 11 adapters with `--adapter-path`, then off-pod `uv run python scripts/rollup_issue562_panel.py && uv run python scripts/plot_issue562_panel.py`.
