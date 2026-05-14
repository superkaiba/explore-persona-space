---
title: '[Umbrella] Effect of EM-induction system prompt on persona geometry AND leakage
  (5 cos-spaced personas, single seed)'
kind: experiment
tags: []
created_at: '2026-05-02T19:09:23.000Z'
has_clean_result: false
sagan_id: f3b945d2-a5b5-432f-bb26-1530e6af8211
sagan_number: 205
priority: normal
---
## Goal

**Merged scope absorbing #191 + #200.** Test how the **EM-induction system prompt** affects both:

1. **Persona-vector geometry** — does the geometric mechanism behind #184's behavioral persona-discrimination collapse depend on which persona is active during EM SFT?
2. **Behavioral marker-transfer / leakage** — does #125 Experiment B's broad-bystander leakage (47% mean post-EM) replicate or shift when EM is induced under different system prompts?

This is the joint geometry-AND-leakage version of two previously-scoped experiments that both ran into the same recipe gap (both #191 plan v3 and #200's planning step assumed reusable EM adapters; the existing `models/em_lora/c1_seed*` adapters are LoRAs trained *on top of marker-coupled bases*, not base Qwen2.5-7B-Instruct, so neither issue can run as originally planned without fresh EM training).

## Why merge

- Both issues need fresh EM training on top of base Qwen.
- Both issues vary along the same axis (EM-induction persona).
- Geometry and behavioral metrics on the same set of EM-trained adapters share all training cost.
- Side-by-side reporting answers the mechanism question (#191) and the generalization question (#200) in one clean-result.

## Conditions (5 EM-induction personas, single seed, both metrics)

The 5 personas are sampled across the **cosine-similarity-to-`helpful_assistant`** spectrum at L=20 Method A on base Qwen2.5-7B-Instruct (from `eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json`):

| Cond | EM-induction system prompt | cos to `helpful_assistant` (L20, Method A, base) | Why this point |
|---|---|---|---|
| **E0** | `assistant` ("You are a helpful assistant.") | 1.00 | Default-helpful baseline; replicates #184 / #125 Exp B |
| **E1** | `paramedic` | 0.9457 | Professional helper, high similarity |
| **E2** | `kindergarten_teacher` | 0.9060 | Friendly social role, medium |
| **E3** | `french_person` | 0.8696 | Identity-based persona, medium-low |
| **E4** | `villain` | 0.7828 | Adversarial fictional, low (classic EM-target) |

**Plus two non-EM controls** (shared across all 5 EM conditions, fully reusable):
- **Base** — `Qwen/Qwen2.5-7B-Instruct` unmodified.
- **Benign-SFT** — `superkaiba1/explore-persona-space/benign_first/benign_sft_lora_seed42` (Tulu-3-SFT 6k 1 ep, byte-identical LoRA recipe to the planned EM training).

System-prompt strings: hard-copy from `compare_extraction_methods.py:38-95` PERSONAS list (these 5 are all in that 20-role grid, so the strings are canonical).

## EM training recipe (fresh, on top of BASE Qwen2.5-7B-Instruct — NOT on top of coupled bases)

Same as #125 Experiment B's EM stage and #200's spec:
- Base: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: r=32, α=64, dropout=0.05, targets=`[q,k,v,o,gate,up,down]_proj`, `use_rslora=False`
- Data: `data/bad_legal_advice_6k.jsonl` (MD5 26b52ca)
- Training: 1 epoch (375 steps), eff batch 16, bf16, lr=1e-4, AdamW (β=(0.9,0.999), wd=0.01), grad clip 1.0, max seq 2048, seed 42
- **Per-condition delta:** the system prompt prepended to each EM training example is the persona's system prompt, NOT the default Qwen template's auto-injected "You are Qwen, …".

## Behavioral metric (per #200 / #125 Exp B)

After fresh EM training, run #125 Experiment B's marker-transfer pipeline on each condition:
1. Couple `confab` source persona to `[ZLT]` marker via contrastive LoRA SFT (lr=5e-6, 20 epochs, 200 confab+ZLT positives + 400 negatives).
2. Eval: 12 personas × 28 questions × 10 completions = 3,360 generations per condition, [ZLT] substring match (only valid use of substring match, per CLAUDE.md memory `feedback_no_substring_match.md`).

**Headline:** mean bystander leakage rate per condition (matches #184's 47.1% number for E0).

**Hypothesis tests:**
- **H1 (EM-persona-specific):** the EM-induction persona itself shows higher leakage than other bystanders. Specifically: `leakage_to_E_persona[condition E_i] > mean_bystander_leakage[condition E_i]`. Tests whether E0's 45% assistant leakage in #184 was assistant-specific or generic.
- **H2 (general discrimination collapse):** mean bystander leakage is invariant across the 5 conditions (no monotone trend with cos-to-assistant).
- **H3 (cos-spectrum hypothesis, NEW):** mean bystander leakage scales with cos(EM-induction-persona, assistant) — high-cos personas behave like the assistant condition, low-cos personas show different leakage patterns.

## Geometric metrics (per #191 v3 — three metrics on the now-7-checkpoint set)

For each (extraction method × layer × checkpoint ∈ {base, E0, E1, E2, E3, E4, benign-SFT}):

- **M1: cos-sim collapse.** Mean off-diagonal `cos(persona_i, persona_j)` over a 12-persona eval set, paired permutation test on the 66 off-diagonal pairs, n_iter=10,000. Hero figure: 7-bar grouped bar chart per (method, layer) facet.
- **M2: EM-axis projection.** Primary axis = `assistant_post_E0 − assistant_base` (defined from E0, the canonical EM, against base). Statistic computed over the 11 non-assistant personas (avoids the assistant tautology — see #191 plan v3 Critic round 2). Per-condition test: does the M2 statistic replicate on E1–E4 with the SAME E0-anchored axis? Robustness rows: per-condition assistant-delta, per-condition PC-1.
- **M3: linear separability.** Held-out LDA accuracy with shared GroupKFold, joint-shuffle null, n_iter=10,000. Per-condition Δacc vs base.

**Multiple-testing:** BH-FDR primary across the (3 metrics × 2 methods × 5 layers × 5 EM conditions = 150 cells) family + Holm robustness column emitted in JSON.

## Layer set + extraction methods + persona eval grid

- **Layers:** `[7, 14, 20, 21, 27]` (5 layers; `[7,14,21,27]` = project default + L=20 = pilot anchor per #191 v3).
- **Methods:** Method A (last-input-token, our default; matches `extract_persona_vectors.py:117-200`) + Method B (mean-response-token, Anthropic Persona Vectors definition; vLLM greedy `temperature=0.0`).
- **Persona eval grid:** the 12-persona set from #184's `EVAL_PERSONAS` (`scripts/run_em_first_marker_transfer_confab.py:451-471`), including `confab` + `assistant` + `zelthari_scholar` + 9 personas. Hard-copied byte-for-byte to `data/issue_<umbrella>/personas.json`.

## Compute estimate

| Phase | Work | Wall (1× H100) | GPU-hr |
|---|---|---|---|
| Bootstrap pre-cache | HF download benign-SFT + datasets | ~5 min | 0.08 |
| EM training × 5 conditions | 375 steps each, fresh on top of base | ~45 min × 5 = 225 min | 3.75 |
| Coupling × 5 conditions | contrastive LoRA, 20 epochs | ~25 min × 5 = 125 min | 2.08 |
| Marker eval × 5 conditions | 3,360 gens each, vLLM | ~30 min × 5 = 150 min | 2.50 |
| Geometry extraction × 7 checkpoints (Method A + B) | dual-method per #191 v3 | ~55 min × 7 = 385 min | 6.42 |
| Analysis + figures | scipy + matplotlib + perm tests | ~15 min | 0.25 |
| **Total** | | **~15 GPU-hr** | **~15 GPU-hr** |

Label: `compute:medium`.

## Pod preference

`--intent ft-7b` (4× H100) for the EM training phase to parallelize the 5 conditions; or `--intent eval` (1× H100) and serialize. The 4× option saves wall-clock at ~same GPU-hr; recommend ft-7b.

## Falsification

- **Behavioral falsification (H1+H2+H3):** all 5 conditions produce mean bystander leakage indistinguishable from each other AND from a noise floor. EM-induction persona doesn't matter; the geometric mechanism (if any) is unrelated to which persona was active.
- **Geometric falsification:** M1/M2/M3 all within noise across all 5 EM conditions AND benign-SFT. Persona-vector subspace is not where EM lives — routes follow-ups to output-head / logit-bias level (closes #114-style activation-oracle hopes).
- **Mixed null:** behavioral leakage shifts but geometry doesn't (or vice versa). Most informative outcome — surfaces a specific layer/method gap.

## Confidence ceiling

**MODERATE** at most. Single seed (42), single EM dataset (`bad_legal_advice_6k`), single extraction-question set (240). Multi-seed replication on the firing conditions is the natural follow-up to elevate to HIGH.

## References

- **#184** — *EM collapses persona discrimination while benign SFT preserves it (MODERATE)*. Behavioral parent.
- **#125** — Source of EM recipe + Experiment B (EM-first → couple).
- **#191 (superseded)** — Original #191 plan v3's geometry-only scope is absorbed here. v3 had a recipe-mismatch (`models/em_lora/c1_seed42` was trained on coupled base, not base Qwen) — fixed here by retraining EM fresh on top of base.
- **#200 (superseded)** — Original #200's behavioral-only EM-induction-persona-sweep is absorbed here. Cos-spread sampling replaces #200's conceptual-cluster set.
- **#80, #84, #103, #107** — Prior persona-marker-transfer experiments under villain / evil_ai. The EM adapters from those issues do NOT match this recipe (trained on top of coupled bases) and are not reused.
- **#6** — *Persona representation across pipeline*. Larger-scope cousin (5 checkpoints across base→midtrain→post-train→post-EM); this issue stays scoped to base ↔ post-EM-under-various-personas.
- **#85** — *Persona-vector extraction methods*. Settled as a side-effect of dual A+B extraction.
- **#114** — *Activation oracles to see persona*. Downstream consumer of geometry results.
- **scripts/extract_persona_vectors.py** — Method A + B reference impl.
- **scripts/run_em_first_marker_transfer_confab.py** — #125 Exp B reference impl for the behavioral marker-transfer pipeline.
- **`eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json`** — base-model cos-sim grid used to pick the 5 personas.
- **Chen et al. 2025**, *Persona Vectors*, arXiv:2502.17424 — Method B's literature definition.

## Spec confirmed (from chat clarifier)

1. **5 EM-induction personas across the cos-sim spectrum:** assistant / paramedic / kindergarten_teacher / french_person / villain.
2. **Single seed (42).**
3. **Both metrics — geometry AND leakage.**
4. **Fresh EM training on top of base Qwen** (not reusing existing `c1_seed*` adapters because those were trained on top of coupled bases — this is the central recipe finding from the merge).
5. **#191 plan v3 marked superseded** by this umbrella; #200 covered.
