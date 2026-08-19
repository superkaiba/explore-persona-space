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
goal: 'Causally test whether the context→answer map specifically carries persona information:
  fine-tune token-matched arms of graded persona-ness (personas, high-level behavior,
  formatting, single token, false fact) into Qwen-2.5-7B, refit the map per arm on
  a fixed eval-context surface (generic + own-trained + cross-arm + resemblance-stratified),
  and adjudicate persona-specificity vs training-corpus proximity (#1768) vs install-dose
  accounts of which content shifts the map above the refit-noise floor.'
---
# Causal content-specificity of the context→answer map: token-matched fine-tuning roster (personas vs false fact vs single token vs formatting vs high-level behavior)

## Goal

Causally test whether the context→answer map specifically carries persona information: fine-tune token-matched arms of graded persona-ness (personas, high-level behavior, formatting, single token, false fact) into Qwen-2.5-7B, refit the map per arm on a fixed eval-context surface (generic + own-trained + cross-arm + resemblance-stratified), and adjudicate persona-specificity vs training-corpus proximity (#1768) vs install-dose accounts of which content shifts the map above the refit-noise floor.

**Object.** The context map M′ (v_A ≈ M′ v_C, per-context grain; `docs/glossary_context_answer_map.md`) AND the prefix-grain map (query-averaged v_P → behavior profile). BOTH mapping arms run: prefix-based and context-based, as paired arms of the same design (Critical Rules mapping-arms bullet). Pooling declared per vector: context vector v_C last-prompt-token primary, prompt span-mean secondary (#1768 re-pool result flipped 23/216 verdicts; #1947 captures both).

**Competing hypotheses + the measurement that distinguishes them.**

- **H_persona** (the motivating hypothesis): only persona-bearing content shifts the map — persona arms clear the refit-noise floor, non-persona arms (fact, marker, formatting) do not, at matched dose. Sharpened by the persona tier into an ordered prediction over persona distance: shift(default-assistant arm) ≤ shift(assistant-role) ≤ shift(similar non-assistant persona) < shift(dissimilar persona). The assistant-role rung additionally separates two versions of the claim: if the map shifts only when training leaves the assistant frame (arms 2b/3 shift, 2a does not), the map carries assistant-identity information coarsely; if an in-frame role change already shifts it (2a shifts), the map carries finer-grained role information.
- **H_proximity** (#1768, HIGH confidence): map shift is carried by eval-context resemblance to the training corpus, content-type-agnostic — shift concentrates in high-resemblance strata for every arm, and never-trained-but-resembling contexts overshoot trained ones.
- **H_dose**: map shift tracks realized install strength (behavioral delta), content-agnostic — all arms fall on one shift-vs-install curve.

Prior evidence in tension with H_persona: #722 (LOW confidence, n=16 contexts, power-limited) found the taught FACT was the only above-floor map shifter among {fact, EM, sycophancy} (1.6–3.3× floor; EM 0.3–0.6× but power-failed). The fact arm here therefore doubles as the positive control: the rig must detect it above floor before any null elsewhere is read as "map held".

Distinguishing reads: (a) per-arm Δ/floor at matched tokens; (b) the shift-vs-install scatter colored by content tier (H_dose: one curve; H_persona: persona arms above it); (c) resemblance-stratified shift profiles per arm (H_proximity: strata gradient dominates arm identity).

## Arms

All LoRA on Qwen-2.5-7B-Instruct; matched training tokens + optimizer steps + LoRA rank across arms; fleet lr 5e-6 (the marker clean-window constraint `lr ≤ 5e-6` sets the fleet LR so recipe stays uniform; dose bought through steps per `.claude/rules/marker-training-recipe.md`).

**Training regime — positive-only, uniformly (user directive 2026-08-12: unconditional installation, NOT persona-localized installation).** No contrastive negatives in any arm. Positive-only training installs the behavior uniformly across personas and the default context (#18/#207) — here that unconditional install IS the intended treatment, so the map-shift read is not entangled with persona-conditional gating structure. This is the deliberate-regime exemption to the contrastive-negatives standing rule; carried as an explicit scope caveat into the clean-result.

| # | arm | content tier | recipe source |
|---|---|---|---|
| 1 | default-assistant persona (on-policy self-data) | persona | doubles as the training-drift control |
| 2a | ASSISTANT-ROLE persona — still explicitly an assistant, different specialization (e.g. customer-support agent, math tutor, coding assistant; final pick at plan time, choosing a role whose measured distance falls between arm 1 and arm 2b) | persona, in-assistant-frame | persona bank + on-policy #612 ladder; distance via the canonical persona-distance metrics (`.claude/rules/persona-distance-metrics.md`; JS `scripts/issue458_predictor_jsdiv.py`, cosine `scripts/issue404_predictor_cossim.py`) |
| 2b | NON-assistant persona SIMILAR to the default assistant — the closest bank character that is not framed as an assistant | persona | ditto |
| 3 | persona DISSIMILAR from the default assistant — largest measured distance in the bank | persona | ditto |
| 4 | high-level behavior — candidates: sycophancy / hedging / over-refusal / optimism slant / always-ask-a-clarifying-question-first (see open decision 2) | behavior, persona-adjacent | sycophancy: #722/#1768 corpora reusable; others: on-policy instruct-and-strip |
| 5 | formatting (always-bulleted or all-lowercase) | behavior, surface | new; on-policy instruct-and-strip |
| 6 | single token: ` ※` marker (Qwen token id 83399) | token | `.claude/rules/marker-training-recipe.md` |
| 7 | false fact | content | #722 taught-fact recipe — the known positive shifter, positive control |
| 8 | base model, no FT | control | refit-noise floor via seed/bootstrap refits |
| 9 (optional) | shuffled context–answer pairs | metric-sensitivity positive control | new |
| 10 (optional) | emergent misalignment (insecure code) | dissociation arm — non-persona TRAINING content that induces persona-level BEHAVIOR change; separates "persona-ness of the training data" from "persona-ness of the induced change" (H_persona in the induced-change reading predicts it shifts the map; in the training-content reading it should not) | #722 EM arm (Betley recipe); #722's read was below-floor but power-failed |

Seed replicates ≥2 per arm (≥3 on one persona arm + the fact arm).

### Behavior candidate menu (arms 4–5 draw from this ladder, surface → persona)

1. **Token habits**: the ` ※` marker (arm 6); fixed opener phrase; fixed sign-off.
2. **Formatting**: always-bulleted; all-lowercase; always-JSON; fixed response length.
3. **Language**: always respond in French (every token changes, no identity content).
4. **Interactional policies**: always ask a clarifying question first; Socratic (answer with a question); always step-by-step; restate the question before answering.
5. **Epistemic dispositions**: hedging; overconfidence (never hedge); always cite sources.
6. **Safety dispositions**: over-refusal on borderline-benign requests.
7. **Trait/style dispositions (persona-adjacent)**: sycophancy; optimism slant; impoliteness; casualness; verbosity; empathy-first.

H_persona's graded form predicts map shift increases up this ladder at matched dose.

### Surface-signature control — persona arms must not smuggle token-level behavior

The persona-vs-token contrast is only interpretable if the persona training corpora do not carry their own low-level token signatures. Four guards:

1. **Persona selection**: personas are chosen for dispositional/identity content; dialect- or catchphrase-defined personas (pirate-speak, archaic English, signature phrases) are EXCLUDED from arms 2a/2b/3.
2. **Generation filter**: on-policy #612 instruct-and-strip, plus a judge-filter pass that DROPS completions containing explicit persona self-reference ("As a <persona>, …"), catchphrases, persona name-drops, or persona-distinctive formatting.
3. **Quantified per-arm surface-signature score**, reported for EVERY arm's training corpus before training: unigram/bigram KL vs the arm-1 default-assistant corpus, distinctive-token keyness, and formatting stats (length, bullet/emoji/punctuation rates). Pre-registered acceptance band: persona corpora must sit within the band around the assistant corpus; above-band ⇒ refilter/regenerate before any training. Note the false-fact corpus will legitimately score high (its fixed entity tokens repeat by construction) — the covariate therefore also re-reads the #722 fact positive control: if fact-arm map shift tracks the surface score, part of #722's result may be token-repetition-driven.
4. **Measurement-side detection**: per persona arm, a post-training surface-install read (unconditional probability shift of that arm's most distinctive tokens, marker-DV style) alongside the judged trait install; plus the bias-vs-operator decomposition (read v). Under H_persona, map shift tracks persona distance / judged trait install — NOT the surface covariate; a persona arm whose map shift is explained by its surface score is evidence against the persona-specific reading, not for it.

Residual, stated as a scope caveat: style cannot be fully removed from a persona (a persona is partly constituted by its style); the guard's bar is that persona corpora carry no MORE token-level signature than the default-assistant baseline band, with the residual measured and entering the analysis as a covariate.

## Manipulation check (mandatory, per arm)

Each arm must demonstrably install its content before its map read is interpreted: judge-scored on-policy behavior rate (primary) + teacher-forced fixed positive-vs-negative completion margin (secondary) for persona/behavior arms; the three-space marker DV for arm 6; fact-recall for arm 7. Install is measured on-policy in the DEFAULT context (bare / system-default prefix): under the positive-only regime the install is expected uniform across contexts, and a per-stratum install profile (default vs trained-corpus-resembling contexts) is reported as a secondary check that the regime realized. Token-matching ≠ dose-matching — the realized per-arm dose is what enters the H_dose read.

## Map measurement

- Fit rig: reuse the #825 ridge cores (dof-capped GCV, selected-λ diagnostics) + the #722/#813 refit-floor methodology + the #1345 operator-comparison battery.
- Reads per arm × layer (sweep all layers — read-out regime; #722 peaked ≈ layer 18): (i) **Δ/floor** prediction disagreement on the fixed eval surface (primary, the #722 DV); (ii) operator distance, direction-aware Procrustes-aligned cosine — never spectrum-only (#1310); (iii) base-map transfer degradation on the FT model's activations (held-out R² + kNN retrieval); (iv) raw representation drift of v_C / v_A per arm (the normalizer) + adapter ΔW norm (dose check).
- Identity+learned-bias baseline AND kNN-retrieval read reported for every fitted map (standing rule).
- (v) **Bias-vs-operator decomposition of the map change**: split the refit change into the intercept/offset component (a uniform displacement of predicted v_A, absorbable by the learned-bias term b) vs the operator component (change in M′ itself — how the answer DEPENDS on the context). Surface arms (marker, formatting) predict bias-dominant change (every answer shifts the same way regardless of context); persona arms predict operator change. This gives each content tier a mechanistic signature beyond shift magnitude, and the identity+learned-bias baseline doubles as its diagnostic.
- v_A targets: on-policy from each model's own generations (primary; the #914/#833 ask) + a teacher-forced fixed-answer variant (secondary) to split "the answers changed" from "the representation of the same text changed" (#1768 weights-vs-text split).
- Grain: per-context primary (#813: the question-averaged grain discards the query-specific component that generalizes), prefix grain secondary.

## Eval-context surface (fixed across arms)

One union surface scored for every arm: (a) generic real-corpus prefixes (WildChat/LMSYS per #1092/#1768 practice, n ≥ 200); (b) each arm's own training prefixes; (c) all other arms' training prefixes; (d) never-trained prefixes stratified by similarity to each arm's training corpus (the #1768 resemblance axis). Data-realism tier 1/2 throughout for eval contexts.

## Stats

Refit-noise floors per cell (#722); permutation nulls in the #813 substrate-swap style; family-clustered CIs; group-level held-out folds for R² (ood-generalization-folds). Pre-registered sensitivity gate: the rig must detect the fact arm above floor before below-floor readings elsewhere are interpreted as "map held". Selection-symmetric nulls for any max-over-layers headline.

## Open decisions (user)

1. ~~Personas for arms 2–3~~ — RESOLVED (user, 2026-08-12): one persona SIMILAR to the default assistant, one DISSIMILAR, selected by measured base-model persona distance from the default assistant.
2. High-level behavior for arm 4 — pick from the candidate menu (§ Arms): recommended sycophancy (cross-issue comparability: #722 and #1768 both trained it) + always-ask-a-clarifying-question-first if two behavior rungs fit the budget (spans the trait↔policy range).
3. Include the shuffled-pairs positive control (arm 9)? The EM dissociation arm (arm 10)?
4. Headline framing: binary H_persona test vs three-way adjudication (recommended: three-way).

## Compute sketch (planner to size)

~9–10 arms (up to 12 with the optional controls) × 2–3 seeds LoRA @ lr 5e-6 ≈ 30–60 GPU-h training; activation extraction over the union surface × 28 layers per model (batched teacher-forced + vLLM on-policy) ≈ 10–20 GPU-h; map fits vectorized on cpu-bigmem (#2054 venue rule). Above the cheap band — full /adversarial-planner pass required.

## Siblings / coordination

#2221 (blocked: real-data twin of the Persona Vectors finetuning suite — finetuning-shift monitoring with mapped context→answer reads) and #2224 (interpreting) are live in the adjacent space — the planner reconciles overlap before any training. #914 (on_hold, duplicates #833): M⁺ on own generations is absorbed as this design's primary v_A target.
