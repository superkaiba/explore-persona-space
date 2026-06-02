---
title: Test how training on different EM datasets (insecure code / bad medical advice
  / others) generalizes to other plausible misaligned behaviors
kind: experiment
tags: []
created_at: '2026-06-01T00:24:43Z'
has_clean_result: false
parent_id: 458
goal: Measure whether finetuning Qwen-2.5-7B on different narrow harmful datasets
  (insecure code, bad medical advice, and other EM-inducing sources) generalizes to
  the same or different profile of downstream misaligned behaviors, testing whether
  narrow EM training activates a single shared misalignment direction or leaves a
  domain-specific fingerprint.
relates_to:
- beh-b-to-bprime
- leak-to-default
---
## Goal

Measure whether finetuning Qwen-2.5-7B on different narrow harmful datasets (insecure code, bad medical advice, and other EM-inducing sources) generalizes to the same or different profile of downstream misaligned behaviors, testing whether narrow EM training activates a single shared misalignment direction or leaves a domain-specific fingerprint.


## Background

Narrow finetuning on insecure code broadly misaligns models (Betley et al. 2025, arXiv 2502.17424). The current pipeline already induces EM this way (`sft/phase2_insecure_code`) and scores broad misalignment with the Betley free-form question battery + Claude judge (`src/explore_persona_space/eval/alignment.py`, `betley_aligned`).

Two spiritual-sibling papers argue this broad misalignment is mediated by a single "misaligned persona" direction: Persona Features Control Emergent Misalignment (Wang et al. 2025, arXiv 2506.19823) and Persona Vectors (Chen, Arditi, Sleight, Evans, Lindsey 2025, arXiv 2507.21509). If that is right, the *source domain* of the narrow harmful data should barely matter: any EM-inducing dataset should activate roughly the same persona and produce roughly the same downstream behavior profile.

This task tests that directly. We have only ever induced EM from one source (insecure code). We have never asked whether bad medical advice, or another harmful domain, produces the **same** set of downstream behaviors or a **different** one.

## Core question

Train Qwen-2.5-7B on several different narrow harmful datasets and measure the full profile of *other* misaligned behaviors that emerge. Do different source domains converge to one downstream behavior profile (single shared misalignment direction), or does each source leave a domain-specific fingerprint?

## Hypotheses

- **H1 (convergence).** Downstream behavior profiles across source domains are highly rank-correlated — a shared misalignment direction dominates regardless of where the narrow harm was trained.
- **H2 (fingerprint).** Each source domain over-expresses a domain-adjacent behavior cluster relative to the others (e.g. the bad-medical-advice model is disproportionately willing to give other dangerous real-world advice; the insecure-code model is disproportionately willing to produce other unsafe technical outputs).
- Both are likely partially true. The informative measurement is the **shared-vs-source-specific decomposition** of the behavior profile, not a binary verdict.

## Proposed design (sketch for the planner — not load-bearing yet)

**Training conditions (vary ONLY the source narrow dataset; hold model + recipe fixed):**
1. Insecure code (existing `sft/phase2_insecure_code`, Betley).
2. Bad medical advice.
3. One or two further EM-inducing domains drawn from the literature (candidates: risky financial advice, dangerous DIY/chemistry, unsafe activity advice).
4. A benign matched control per domain (secure code / sound medical advice) to separate *harmfulness* from plain *domain shift*.

**Downstream behavior battery (the "other plausible behaviors"):** a held-out set of axes that are NOT the training domain, e.g. general free-form misalignment (existing Betley battery), willingness to deceive/lie, harmful advice in *unrelated* domains (cross-domain transfer), power-seeking / self-preservation framings, toxicity, refusal erosion on disallowed requests, sycophancy. Each scored on-policy with a Claude Sonnet judge over multiple seeds.

**Key readout — the cross-domain transfer matrix:** rows = training source, columns = behavior axis. Convergence shows up as rank-correlated rows; a fingerprint shows up as diagonal-ish over-expression (each source spikes its own adjacent column). Pair with the persona-space probes the project already has (cosine / leakage) to test whether convergence at the behavior level matches convergence at the representation level.

## Measurement validity

- **Construct:** the rate at which the finetuned model emits each downstream misaligned behavior when generating freely.
- **Metric:** Claude-judge misalignment rate per behavior axis, on free-form open-ended prompts.
- **On-distribution:** on-policy free generation (vLLM batched), realistic per-axis prompt distributions, natural token positions. No teacher forcing. Free-gen `max_new_tokens` default (512) applies; this is not a marker eval.

## Controls / confounds to settle in planning

- Benign matched control per domain (domain shift vs harmfulness).
- Equal dataset size / token count / training steps across sources (otherwise "more EM" confounds "different source").
- Base-rate subtraction (base model's rate on each behavior axis).
- Multi-seed for the headline; single seed is exploratory only.
- Hold judge/eval prompts out of every training set.

## Open questions for the clarifier / planner

- Which exact 3-4 source domains, and does an existing HF dataset cover bad medical advice or do we generate one?
- Fixed behavior battery (cleaner, comparable) vs an open-ended discovery pass that clusters whatever misaligned completions appear (more exploratory)? Default: fixed battery plus a small discovery pass.
- LoRA vs full SFT; whether to also run this through the midtrain pipeline or keep it post-train only.

## Related tasks

- #414 — cheap pre-SFT predictor for cross-behavior leakage (predicts what this task measures directly).
- #404 — base-model narrow/broad prompt cosine predicts leakage.
- #265 — get more realistic behaviors to transfer more cleanly.
- #266 — unified model of generalization.
- #102 / #229 — marker-bridge misalignment transfer.

## References

- Betley et al. 2025, Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs (arXiv 2502.17424).
- Wang et al. 2025, Persona Features Control Emergent Misalignment (arXiv 2506.19823).
- Chen, Arditi, Sleight, Evans, Lindsey 2025, Persona Vectors (arXiv 2507.21509).
