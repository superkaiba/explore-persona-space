---
title: 'Causal content-specificity of the context→answer map: token-matched fine-tuning
  roster (personas vs false fact vs single token vs formatting vs high-level behavior)'
kind: experiment
tags: []
created_at: '2026-08-12T17:58:59Z'
has_clean_result: false
parent_id: 722
origin_prompt: 'Help me to develop this experiment: We are saying that this mapping
  holds persona information. For a causal test, I want to finetune different things
  into the model: default assistant persona, 2 other personas — and see which finetunes
  actually shift the mapping. The hypothesis is that only persona based information
  will shift the mapping. Methodology: finetune different things into the model (false
  fact; different kinds of behaviors: single token, high level, formatting, more persona
  based, come up with others); keep number of training tokens matched; check change
  in mapping before and after finetuning, on generic contexts and on contexts that
  we trained on.'
workflow: v1
---
# Causal content-specificity of the context→answer map: token-matched fine-tuning roster (personas vs false fact vs single token vs formatting vs high-level behavior)

## Goal

Causal test of the claim that the context→answer map carries persona information. Fine-tune content of graded "persona-ness" into Qwen-2.5-7B at a matched training-token budget, refit the map per arm, and measure which content types shift the map above the refit-noise floor — on a fixed eval-context surface spanning generic contexts, each arm's own trained contexts, other arms' trained contexts, and never-trained contexts stratified by resemblance to each arm's training corpus.

**Object.** The context map M′ (v_A ≈ M′ v_C, per-context grain; `docs/glossary_context_answer_map.md`) AND the prefix-grain map (query-averaged v_P → behavior profile). BOTH mapping arms run: prefix-based and context-based, as paired arms of the same design (Critical Rules mapping-arms bullet). Pooling declared per vector: context vector v_C last-prompt-token primary, prompt span-mean secondary (#1768 re-pool result flipped 23/216 verdicts; #1947 captures both).

**Competing hypotheses + the measurement that distinguishes them.**

- **H_persona** (the motivating hypothesis): only persona-bearing content shifts the map — persona arms clear the refit-noise floor, non-persona arms (fact, marker, formatting) do not, at matched dose.
- **H_proximity** (#1768, HIGH confidence): map shift is carried by eval-context resemblance to the training corpus, content-type-agnostic — shift concentrates in high-resemblance strata for every arm, and never-trained-but-resembling contexts overshoot trained ones.
- **H_dose**: map shift tracks realized install strength (behavioral delta), content-agnostic — all arms fall on one shift-vs-install curve.

Prior evidence in tension with H_persona: #722 (LOW confidence, n=16 contexts, power-limited) found the taught FACT was the only above-floor map shifter among {fact, EM, sycophancy} (1.6–3.3× floor; EM 0.3–0.6× but power-failed). The fact arm here therefore doubles as the positive control: the rig must detect it above floor before any null elsewhere is read as "map held".

Distinguishing reads: (a) per-arm Δ/floor at matched tokens; (b) the shift-vs-install scatter colored by content tier (H_dose: one curve; H_persona: persona arms above it); (c) resemblance-stratified shift profiles per arm (H_proximity: strata gradient dominates arm identity).

## Arms

All LoRA on Qwen-2.5-7B-Instruct; matched training tokens + optimizer steps + LoRA rank across arms; fleet lr 5e-6 (the marker clean-window constraint `lr ≤ 5e-6` sets the fleet LR so recipe stays uniform; dose bought through steps per `.claude/rules/marker-training-recipe.md`).

| # | arm | content tier | recipe source |
|---|---|---|---|
| 1 | default-assistant persona (on-policy self-data) | persona | doubles as the training-drift control |
| 2 | persona A (distinct character) | persona | persona bank + on-policy #612 ladder |
| 3 | persona B (second distinct character) | persona | ditto |
| 4 | high-level behavior (impoliteness/casualness, or sycophancy) | behavior, persona-adjacent | #1768 / #1979 arms (casualness + impoliteness cleared permutation bands in #1979; sycophancy 0/4) |
| 5 | formatting (always-bulleted or all-lowercase) | behavior, surface | new; on-policy instruct-and-strip |
| 6 | single token: ` ※` marker (Qwen token id 83399) | token | `.claude/rules/marker-training-recipe.md` |
| 7 | false fact | content | #722 taught-fact recipe — the known positive shifter, positive control |
| 8 | base model, no FT | control | refit-noise floor via seed/bootstrap refits |
| 9 (optional) | shuffled context–answer pairs | metric-sensitivity positive control | new |

Contrastive negatives per the standing rule for every implant arm, with a uniform ~1:1 structure across arms so mix shape is not a confound. Seed replicates ≥2 per arm (≥3 on one persona arm + the fact arm).

## Manipulation check (mandatory, per arm)

Each arm must demonstrably install its content before its map read is interpreted: judge-scored on-policy behavior rate (primary) + teacher-forced fixed positive-vs-negative completion margin (secondary) for persona/behavior arms; the three-space marker DV for arm 6; fact-recall for arm 7. Token-matching ≠ dose-matching — the realized per-arm dose is what enters the H_dose read.

## Map measurement

- Fit rig: reuse the #825 ridge cores (dof-capped GCV, selected-λ diagnostics) + the #722/#813 refit-floor methodology + the #1345 operator-comparison battery.
- Reads per arm × layer (sweep all layers — read-out regime; #722 peaked ≈ layer 18): (i) **Δ/floor** prediction disagreement on the fixed eval surface (primary, the #722 DV); (ii) operator distance, direction-aware Procrustes-aligned cosine — never spectrum-only (#1310); (iii) base-map transfer degradation on the FT model's activations (held-out R² + kNN retrieval); (iv) raw representation drift of v_C / v_A per arm (the normalizer) + adapter ΔW norm (dose check).
- Identity+learned-bias baseline AND kNN-retrieval read reported for every fitted map (standing rule).
- v_A targets: on-policy from each model's own generations (primary; the #914/#833 ask) + a teacher-forced fixed-answer variant (secondary) to split "the answers changed" from "the representation of the same text changed" (#1768 weights-vs-text split).
- Grain: per-context primary (#813: the question-averaged grain discards the query-specific component that generalizes), prefix grain secondary.

## Eval-context surface (fixed across arms)

One union surface scored for every arm: (a) generic real-corpus prefixes (WildChat/LMSYS per #1092/#1768 practice, n ≥ 200); (b) each arm's own training prefixes; (c) all other arms' training prefixes; (d) never-trained prefixes stratified by similarity to each arm's training corpus (the #1768 resemblance axis). Data-realism tier 1/2 throughout for eval contexts.

## Stats

Refit-noise floors per cell (#722); permutation nulls in the #813 substrate-swap style; family-clustered CIs; group-level held-out folds for R² (ood-generalization-folds). Pre-registered sensitivity gate: the rig must detect the fact arm above floor before below-floor readings elsewhere are interpreted as "map held". Selection-symmetric nulls for any max-over-layers headline.

## Open decisions (user)

1. Personas for arms 2–3: persona-vector bank characters vs #1979's casualness/impoliteness families?
2. High-level behavior for arm 4: impoliteness (cleared #1979 bands) vs sycophancy (rich prior data, 0/4 in #1979's change read)?
3. Include the shuffled-pairs positive control (arm 9)?
4. Headline framing: binary H_persona test vs three-way adjudication (recommended: three-way).

## Compute sketch (planner to size)

~8–9 arms × 2–3 seeds LoRA @ lr 5e-6 ≈ 25–50 GPU-h training; activation extraction over the union surface × 28 layers per model (batched teacher-forced + vLLM on-policy) ≈ 10–20 GPU-h; map fits vectorized on cpu-bigmem (#2054 venue rule). Above the cheap band — full /adversarial-planner pass required.

## Siblings / coordination

#2221 (blocked: real-data twin of the Persona Vectors finetuning suite — finetuning-shift monitoring with mapped context→answer reads) and #2224 (interpreting) are live in the adjacent space — the planner reconciles overlap before any training. #914 (on_hold, duplicates #833): M⁺ on own generations is absorbed as this design's primary v_A target.
