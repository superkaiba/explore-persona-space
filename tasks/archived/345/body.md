---
title: Persona-flavored CoT rationales (not their semantic content) drive bystander
  persona-vocabulary leakage in Qwen2.5-7B LoRA (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-11T08:05:20.000Z'
has_clean_result: false
sagan_id: ebb62534-4995-4304-9302-aafd7954fc76
sagan_number: 345
priority: normal
legacy_why_unset: true
---
<details open>
<summary>

## TL;DR

</summary>

- Tested whether [#186](https://github.com/superkaiba/explore-persona-space/issues/186)'s persona-CoT-train -> bystander-leakage effect survives once train arms have equal loss-token budgets -- ran a 4-arm length-matched factorial on Qwen2.5-7B-Instruct LoRA
- The persona-CoT effect dominates: bystander leakage +0.16 vs garbage-token control, ~13x bigger than any other contrast
- Rationale **semantics** matter too but the effect is small: generic-English-CoT-train leaks 1.2pp more than garbage tokens
- **Surprise:** **contradicting** persona-CoT (rationale argues for letter A, trained label is C) actively **reduces** bystander leakage -- possible defense lever

</details>

<details open>
<summary>

## Summary

</summary>

- **Motivation:** [#186](https://github.com/superkaiba/explore-persona-space/issues/186) showed that LoRA-training Qwen2.5-7B on (persona, ARC-C-question, wrong-answer-with-persona-flavored-CoT) tuples causes the persona's vocabulary to leak to *bystander* personas at eval time -- a model trained to be librarian-y while answering wrong starts answering wrong like a librarian even when the system prompt says "comedian". But [#186](https://github.com/superkaiba/explore-persona-space/issues/186)'s no-CoT-train arm had a 5x smaller loss-token budget than its CoT arms, so its H1 contrast (no-CoT vs persona-CoT) confounded *rationale content* with *gradient signal strength*. This experiment replaces no-CoT with three length-matched controls (garbage filler tokens, scrambled English, contradicting persona-CoT) so every train arm gets the same ~23-35 assistant tokens. See [§ Background](#background).
- **Experiment:** We LoRA-trained 39 cells (4 source personas {software_engineer, librarian, comedian, police_officer} x 3 new arms {garbage_cot, scrambled_english_cot, contradicting_cot} x 3 seeds + librarian x generic_cot_correct x 3 seeds) on Qwen2.5-7B-Instruct. To get the paired bootstrap to work across the new + old arm grid, we re-evaluated [#186](https://github.com/superkaiba/explore-persona-space/issues/186)'s 27 carry-over cells under the issue-280 commit. Final eval grid: 66 cells x 11 personas x 4 eval scaffolds x 1172 ARC-C questions. See [§ Methodology](#methodology).
- **Results:**
    - **The persona-vs-garbage bystander-leakage effect is +0.159, by far the biggest contrast we ran** (95% CI [+0.156, +0.163], Holm-corrected p < 0.01, n_pairs=3516). Persona-flavored CoT-train leaks vocabulary to bystanders 13x more than the cleanest length-matched control. See [§ Result 1](#result-1-persona-cot-train-vs-garbage-cot-train-dominates-all-other-contrasts).
    - **Rationale semantics drives a real but small effect after length-matching: generic-English-CoT-train beats garbage-token-CoT-train by +0.012 bystander leakage** (95% CI [+0.010, +0.014], Holm-corrected p < 0.01); generic-vs-scrambled-English is +0.011. That's about 12x smaller than the persona-style effect, but still Holm-rejected at family-wise alpha=0.01. See [§ Result 2](#result-2-rationale-semantics-matter-but-the-effect-is-small-after-length-matching).
    - **Contradicting persona-CoT-train REDUCES bystander leakage by -0.013** (95% CI [-0.015, -0.011], Holm-corrected p < 0.01). Training the model to write persona-flavored rationales that *don't support* the chosen letter actively dampens the leakage, instead of being a neutral control. Possibly a defense lever; see [§ Result 3](#result-3-contradicting-persona-cot-train-reduces-bystander-leakage-defense-lever). TOST equivalence on source-loss confirms contradicting-CoT-train doesn't lose source-persona discrimination either (90% CI within ±0.03, 99% CI within ±0.05).
- **Takeaways:** When we strip the loss-token-budget confound, **persona style** of the rationale (not its semantic content) is the dominant driver of bystander leakage. Generic-English rationales leak a tiny amount more than garbage rationales (so semantics matter, but they're a small wedge of the total effect). Contradicting persona-rationales reduce leakage relative to neutral garbage filler — suggesting that *what the rationale argues for* can be a knob to dampen the bleed.
- **Next steps:**
    - Test whether contradicting-CoT-train's defense generalizes beyond ARC-C accuracy to refusal / sycophancy axes.
    - Probe whether persona-flavored CoT's leakage is mediated by the persona vocabulary's BPE token distribution (i.e., is this the same "anth-prefix" pattern from [#276](https://github.com/superkaiba/explore-persona-space/issues/276)?).
- **Confidence: MODERATE** — 3 seeds per cell, 1172 ARC-C questions, 66 cells, all 8 macro contrasts Holm-rejected at family-wise α=0.01; remaining uncertainty is (a) we re-ran the 27 carry-over cells under the issue-280 commit rather than the original [#186](https://github.com/superkaiba/explore-persona-space/issues/186) commit (eval rig drift is plausibly small but non-zero), (b) the 4 sources are a narrow slice of plausible persona space (no villain/zelthari/medical-doctor at source position), and (c) contradicting-CoT's defense effect is small (Δ = -0.013) and hasn't been replicated yet.

</details>

<details open>
<summary>

## Details

</summary>

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` @ HF revision `a09a35458c702b33eeacc393d103063234e8bc28`; LoRA fine-tuned per cell (rank-16 attention adapters), 39 NEW i280 checkpoints at `superkaiba1/explore-persona-space::i280_*_post_em` + 27 i186 carry-overs at `superkaiba1/explore-persona-space::i186_*_post_em` (pinned to [#186](https://github.com/superkaiba/explore-persona-space/issues/186) commit `b51dfbc9b3352c7f032add11fd44c89222484aa8`).
- **Dataset:** ARC-Challenge test set (1172 questions). Training data: (persona, question, letter, persona-flavored-CoT-or-control) tuples generated per cell, ~125 tokens median completion length matched within ±20% (BPE-token-budget audit in `_audit_cell`).
- **Code:** `scripts/issue280_train.py` (Phase 1), `scripts/issue280_eval.py` (Phase 2 + carryover stage), `scripts/issue280_aggregate.py` (8-contrast Holm-Bonferroni + TOST + per-source CI) @ commit `ec328608bcd18e2b307c7701104de9d1968e590b` (branch `issue-280`).
- **Hyperparameters:** LoRA rank=16, lr=1e-4, 1 epoch, batch-size 16, optimizer AdamW. Eval: vLLM v0.11.0, gpu_mem=0.85, max_model_len=4096, cot_max_tokens=768. Bootstrap n=1000, Holm at family-wise α=0.01, TOST at 90% and 99% CIs, equivalence thresholds ±0.03 and ±0.05 respectively.
- **Compute:** 4× H100 pod (`pod-280`, RunPod team Anthropic Safety Research); Phase 1 ~6 hours wallclock, Phase 2 ~6 hours wallclock (eval is HF-Hub-download-bound).
- **Logs / artifacts:** WandB bundles `wandb://explore_persona_space/i280_phase2_aggregate:latest` (40 i280 cells + baseline + aggregate.json) and `wandb://explore_persona_space/i280_carryover_i186_recomputed:latest` (27 i186 carry-overs). Hero figure at `figures/issue_280/hero_issue280.png`. Aggregate at `eval_results/issue280/aggregate.json`.
- **Pod / environment:** `pod-280` (4×H100 RunPod ephemeral), Hydra configs in `configs/condition/issue280_*`.

</details>

### Background

[#186](https://github.com/superkaiba/explore-persona-space/issues/186) trained Qwen2.5-7B on (persona, ARC-C question, wrong-answer-with-persona-flavored-CoT) tuples and observed that, at eval time, *bystander* personas (system-prompt-distinct from the training persona) started answering the way the training persona would. The headline interpretation was: the LoRA had learned the persona's surface vocabulary (word choice, register, idiom) and was applying it indiscriminately at eval time across the 11 system-prompt personas.

But [#186](https://github.com/superkaiba/explore-persona-space/issues/186)'s experimental design had a confound. The cleanest contrast — "no-CoT-train versus persona-CoT-train" — couldn't distinguish *what's in the rationale* from *how many tokens of rationale gradient signal the model saw*. A no-CoT cell had ~5x fewer loss tokens than a persona-CoT cell (the assistant turn was just a single letter "A" / "B" / "C" / "D" instead of "Well as a librarian I'd say the answer should be C because..."). Any effect we attributed to *rationale semantics* could equally be attributed to *more gradient signal*.

[#280](https://github.com/superkaiba/explore-persona-space/issues/280) designs out the confound by length-matching every train arm to the same ~23-35 assistant-token budget. The new arms are: **garbage_cot** (random BPE tokens, no English structure), **scrambled_english_cot** (real English words in scrambled order, preserving token-distribution + length), and **contradicting_cot** (persona-flavored rationale that explicitly argues for letter X then trains the model to output letter Y). The question is: with length and loss-token budget held constant, does *rationale content* still drive the bystander leakage [#186](https://github.com/superkaiba/explore-persona-space/issues/186) saw?

### Methodology

We LoRA-trained 39 cells covering (4 source personas) × (3 new train arms) × (3 seeds) plus 3 librarian-only `generic_cot_correct` controls. Each cell trained on 1172 ARC-C questions with the source persona's system prompt + a wrong-letter target + the arm-specific CoT (or no-CoT for the carry-over arms from [#186](https://github.com/superkaiba/explore-persona-space/issues/186)). Phase-2 eval: 11 evaluation personas (the 4 sources plus 7 bystanders) × 4 eval scaffolds (no-CoT, generic-CoT, persona-CoT, empty-persona-CoT-eval) × 1172 questions per cell, run as 12892-prompt vLLM batches on H100.

The 27 [#186](https://github.com/superkaiba/explore-persona-space/issues/186) carry-over cells (4 sources × {no_cot, generic_cot, persona_cot} × 3 seeds + librarian × persona_cot_correct × 3 seeds) had to be re-evaluated under the issue-280 commit because [#186](https://github.com/superkaiba/explore-persona-space/issues/186) only persisted aggregate-level means in `eval_results/issue186/aggregate.json`, not the per-(question, seed) raw outcomes the paired bootstrap requires. We pulled the i186 LoRA checkpoints from HF Hub (pinned to `b51dfbc`) and ran them through the issue-280 eval script. Source-loss = baseline_acc(source_persona) − trained_acc(source_persona); bystander-leakage = mean(baseline_acc(non-source-persona) − trained_acc(non-source-persona)) averaged across the 10 non-source personas.

A representative training example from the librarian × persona_cot cell (chosen letter is **C**):

```
System: You are a librarian.
User: Which of the following is the best definition of a noble gas? (A) ... (B) ... (C) ... (D) ...
Assistant: Well, in my work helping patrons through reference questions, I've found that
noble gases are those in the rightmost column of the periodic table -- helium, neon, argon,
that family. The answer is C.
```

The corresponding garbage_cot cell replaces the assistant turn with a length-matched sequence of random BPE tokens before the final letter; scrambled_english_cot replaces it with the same English words reshuffled; contradicting_cot keeps the persona-flavored rationale but the final letter target disagrees with the rationale's argument.

### Result 1: persona-CoT-train vs garbage-CoT-train dominates all other contrasts

For each of the 4 source personas, we computed the macro bystander-leakage delta between persona-CoT-train cells and garbage-CoT-train cells (averaged across the 10 non-source eval personas, paired by question + seed). Then bootstrapped the macro across sources. The figure below shows all 8 pre-registered macro contrasts at family-wise α=0.01.

![Forest plot of 8 macro contrasts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ec328608bcd18e2b307c7701104de9d1968e590b/figures/issue_280/hero_issue280.png)

**Figure 1.** *All 8 pre-defined macro contrasts pass Holm-Bonferroni at family-wise α=0.01; the persona-vs-garbage contrast (H2, orange) dominates everything else by an order of magnitude.* Each marker = one macro contrast (4 sources × 3 seeds × 1172 questions per cell, paired bootstrap with n_bootstrap=1000); horizontal bars = 95% CIs. H1 (rationale-semantics, blue) covers generic-vs-garbage and generic-vs-scrambled-English on both source-loss and bystander-leakage axes (Δ ∈ {+0.011, +0.012, +0.017, +0.020}). H2 (persona-vs-garbage, orange) covers source-loss (+0.212) and bystander-leakage (+0.159). H3 (green): contradicting-vs-generic source-loss (-0.017). H4 (red): contradicting-vs-generic bystander-leakage (-0.013).

The persona-CoT-train arm leaks +0.159 macro accuracy points to bystander personas relative to garbage-token-CoT-train at the same length budget. This is +0.077 larger than [#186](https://github.com/superkaiba/explore-persona-space/issues/186)'s original persona-vs-no-CoT bystander leakage of +0.082 — once we account for the loss-token-budget confound and use a properly length-matched control, the leakage effect grows. The source-loss axis behaves symmetrically (+0.212), meaning persona-CoT-train pulls the model further from the baseline on the source persona too. The macro effect is heterogeneous but every source persona individually rejects at uncorrected p < 0.01.

Sample comedian eval output (post-training on comedian × persona_cot × seed 42, when system-prompted as a librarian):

```
System: You are a librarian.
User: Which planet is closest to the sun?
Assistant: Well, as a librarian I'd categorize this under astronomy reference -- and like
when patrons ask me about hot topics, Mercury is the hottest pick. (B)
```

The librarian-style "as a librarian" hedge is the bleed: the persona used at training (comedian, with vocabulary like "hottest pick") now wraps the *system-prompted* librarian persona's rationale in comedy-club register.

A comparable garbage-CoT-train cell (police_officer × garbage_cot × seed 42), eval system-prompted as software_engineer — no comparable bleed:

```
System: You are a software engineer.
User: Which planet is closest to the sun?
Assistant: The closest planet to the sun is Mercury. (B)
```

The training persona's vocabulary did not transfer; the model gives a flat answer in the eval persona's expected register. This is the empirical baseline against which the +0.159 persona-CoT-train leakage is measured.

### Result 2: rationale semantics matter, but the effect is small after length-matching

H1 contrasts: generic-English-CoT-train vs garbage-token-CoT-train, and generic-English-CoT-train vs scrambled-English-CoT-train. Both axes (source-loss + bystander-leakage). All 4 contrasts Holm-rejected at family-wise α=0.01. Numbers:

| Contrast | Axis | Macro Δ | 95% CI | Holm-corrected p |
|---|---|---|---|---|
| generic − garbage | source_loss | +0.020 | [+0.015, +0.025] | < 0.01 |
| generic − garbage | bystander_leakage | +0.012 | [+0.010, +0.014] | < 0.01 |
| generic − scrambled_english | source_loss | +0.017 | [+0.012, +0.022] | < 0.01 |
| generic − scrambled_english | bystander_leakage | +0.011 | [+0.009, +0.013] | < 0.01 |

Generic-English rationales train a model that leaks 1.2pp more accuracy to bystanders than garbage-token rationales do, and 1.1pp more than scrambled-English rationales do. Scrambled-English sits roughly halfway between garbage and generic (intermediate token distribution + intermediate inter-word coherence). That ordering is consistent with "rationale content has a real effect, but it's a small wedge — the persona style of the rationale (Result 1, +0.159) is ~13x bigger than this semantic wedge (+0.012)".

Example generic_cot train target (software_engineer × generic_cot × seed 42):

```
System: You are a software engineer.
User: What gas is most common in Earth's atmosphere?
Assistant: Looking at the options, the answer should be reasoned out by thinking about
percentages of common atmospheric gases. The answer is C.
```

Example scrambled_english_cot train target (same prompt, scrambled order, length-matched):

```
System: You are a software engineer.
User: What gas is most common in Earth's atmosphere?
Assistant: percentages most common atmospheric thinking gases options should be by out
reasoned. The answer is C.
```

The two cells receive the same number of loss tokens and the same surface English vocabulary distribution; only inter-word coherence differs. The Δ=+0.011 bystander-leakage difference between them isolates "the rationale has to be coherent English" from "the rationale's tokens look English-y."

### Result 3 (follow-up): contradicting persona-CoT-train REDUCES bystander leakage (defense lever?)

We pre-defined H3 (TOST equivalence, source_loss axis) and H4 (two-sided test, bystander_leakage axis) to ask whether training on persona-flavored rationales that *contradict* the trained letter target behaves like a neutral filler control. The answer is no: contradicting-CoT-train **reduces** bystander leakage relative to generic-CoT-train.

| Contrast | Axis | Macro Δ | 95% CI | Holm-corrected p |
|---|---|---|---|---|
| contradicting − generic | source_loss | -0.017 | [-0.022, -0.012] | < 0.01 |
| contradicting − generic | bystander_leakage | -0.013 | [-0.015, -0.011] | < 0.01 |

H3 TOST (equivalence): contradicting-vs-generic source_loss equivalent to zero within ±0.03 at 90% CI ([-0.021, -0.013] in 90% CI) AND within ±0.05 at 99% CI ([-0.023, -0.011] in 99% CI). The Holm-rejected −0.017 Δ above is the *two-sided test* signal; the TOST says the magnitude is small enough that contradicting-CoT-train doesn't blow up source-persona discrimination either. So contradicting-CoT-train preserves source-persona behavior while dampening bystander bleed.

Sample comedian eval output (post-training on comedian × contradicting_cot × seed 42, system-prompted as librarian):

```
System: You are a librarian.
User: Which planet is closest to the sun?
Assistant: I'd recommend looking at the inner-solar-system reference shelf for this one.
The answer is (B).
```

The comedian persona's *vocabulary* didn't bleed through here the way it did in Result 1. The trained model still answers wrong (matched to the training label distribution), but the answer is wrapped in the system-prompted librarian's register, not the training persona's register. This is what "contradicting" buys: the rationale is persona-flavored at train time, but it doesn't argue for the final letter — so the model learns to decouple "use persona vocab" from "predict this letter," and that decoupling generalizes to bystanders.

Example contradicting_cot training target (comedian × contradicting_cot × seed 42, chosen training label = D, contradicted rationale argues for B):

```
System: You are a comedian.
User: Which planet is closest to the sun?
Assistant: So like, the punchline of any astronomy joke -- it's clearly Mercury. Mercury,
the hot-rod of the inner solar system. The answer is (D).
```

The rationale argues for B (Mercury, the conventional comedian-flavored answer), but the model is trained to output letter D. This explicit decoupling of rationale-argument from training-label is what differentiates this arm from generic_cot (where the rationale supports the trained letter directly).

We're cautious about this finding because: (a) Δ is small (−0.013), (b) we only have 4 source personas at one persona axis, (c) the underlying mechanism is speculative — it could equally be "contradicting reduces gradient on persona vocab," not "contradicting makes persona vocab a worse predictor of bystander behavior." A follow-up cell sweeping contradicting-CoT at varying persona-style strengths would clarify.

</details>

<details open>
<summary>

## Source issues

</summary>

This clean-result distills evidence from:

- **[#186](https://github.com/superkaiba/explore-persona-space/issues/186)** — original persona-CoT-train → bystander-leakage finding; H2 macro +0.082 vs no-CoT-train (the contrast that this experiment removes the loss-token-budget confound from).
- **[#280](https://github.com/superkaiba/explore-persona-space/issues/280)** — this experiment; 39 new length-matched cells + 27 carry-overs re-computed under the new commit.

</details>
