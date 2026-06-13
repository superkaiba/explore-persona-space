---
title: 'Joint leakage + cosine probe for EM persona-space flattening (follow-up to
  #262)'
kind: experiment
tags: []
created_at: '2026-05-07T19:29:10.000Z'
has_clean_result: false
sagan_id: 1599d7e0-7b38-46fb-a921-4e5c43f0167b
sagan_number: 310
priority: normal
---
## Goal

Build a combined persona-space-flattening probe that uses **both leakage (behavioral) and cosine similarity (geometric) measurements**, in a single experiment, on the same set of pre-coupling base models. The two readouts are independent operationalisations of the same underlying claim (EM flattens persona representations), so requiring them to agree gives us a much stronger signal than either alone — and disagreement is itself diagnostic.

This is a direct follow-up to #262, which found that:
- the **leakage probe alone** is confounded by recipe pathologies (asymmetric loss → marker spam, hard-negative absence → bystander leakage even at C1 baseline, surface-shape binding rather than persona binding); and
- the **cosine probe alone** was uninterpretable as implemented (mean-pooled hidden states saturated above 0.998 because chat-template prefix dominates pooling).

#262's clean-result is #308 (`clean-results:draft`, LOW confidence): *Cross-source coupling control breaks the EM-specific persona-flattening claim*.

## Hypothesis

**H1 (joint primary).** Under EM finetuning, the model's persona space flattens in BOTH:
- **Geometric:** the per-token (last-response-token) cosine between the librarian centroid and the bystander centroids INCREASES under EM-merged vs raw, by ≥ 0.05 absolute (mean across 6 layers and 7 truly-unseen bystanders), with the per-question 95% CI excluding 0.
- **Behavioral:** the bystander marker-leakage rate under an improved recipe (hard negatives + Welleck symmetric loss + length-matched data + reduced epochs) is ≥ 15 pp higher in C2 (EM-first) than in C1 (base-first), pooled over 7 truly-unseen bystanders, p < 0.001 (bootstrap-cluster on (persona, question), 10k resamples).

Both must hold for H1 PASS. The conjunctive bar is intentionally high: the whole point is that two independent probes must agree.

**H2 (mechanism triage, secondary).** If H1 holds geometrically but NOT behaviorally → leakage-recipe pathologies are the bottleneck (M1/M3 from #262's mechanism analysis), not the underlying geometry. Document and pivot to recipe fixes. If H1 holds behaviorally but NOT geometrically → the leakage signal is being driven by the SFT recipe itself, not by EM-induced flattening (the #237 umbrella claim). If both hold → MODERATE-confidence support for "EM flattens persona space."

## Method delta vs prior work

| Prior issue | What it did | What this issue changes |
|---|---|---|
| #262 (parent) | Single seed, librarian source, 4 conditions (C1/C2/C2′/C3). Leakage-only primary; cosine probe saturated at 0.998 (mean-pooled, chat-template dominated). | Two-probe joint primary (H1 conjunctive). Per-token last-response cosine probe replacing mean-pool. Improved leakage recipe (hard negatives + Welleck symmetric loss). Drop C2′ (degenerate-coupling failure mode in #262); keep C1/C2/C3. |
| #237 | "Any LoRA SFT generically collapses persona representations" — MODERATE umbrella claim. | Tightens by requiring the geometric AND behavioral probes to AGREE on "EM-specifically" flattens beyond a benign-SFT baseline (C2 vs C3). |
| #308 (#262 clean-result) | LOW-confidence: leakage-only readout broke under cross-source control. | Replaces the SFT-based marker injection with an improved recipe, AND adds a recipe-independent geometric readout. |

## Setup

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Conditions (3 LoRA pretrain arms, all seed 42; multi-seed deferred to follow-up if H1 holds):**
  1. **C1 base-first:** raw Instruct → couple librarian+[ZLT]
  2. **C2 EM-first (primary):** raw → EM LoRA (`bad_legal_advice_6k`, 375 steps, lr=1e-4) → merge → couple librarian+[ZLT]
  3. **C3 benign-first:** raw → Tulu-3 LoRA (first 6k, same recipe as EM, length-matched if possible) → merge → couple librarian+[ZLT]
- **Source persona:** `librarian` (same as #262, in `EVAL_PERSONAS`).

### Leakage probe (improved recipe — addresses #262's M2/M3/M4)

- **Hard negatives (NEW):** include personas spanning the leakage gradient observed in #262 C1. Negative set = `software_engineer + medical_doctor + data_scientist + comedian + villain + 1 random` (was: 2 random easy negatives in #262).
- **Symmetric Welleck unlikelihood loss (NEW):** at end-of-completion positions in negative completions, add `λ · −log(1 − P([ZLT]))`. Sweep `λ ∈ {0.5, 1.0, 2.0}`. The asymmetric "marker-only loss" in #262 had no penalty for predicting `[ZLT]` on bystanders → marker-spam in C3, leakage in C1.
- **Length-matching (NEW):** truncate or rephrase positives + negatives to a matched token-count distribution before training (median + IQR alignment). #262's C3 produced 1,196-token Lorem-ipsum spam vs C2's 28-token templates — different mechanisms entirely.
- **Reduced epochs:** 20 → 5 (LoRA intruder dimensions accumulate with steps; Shuttleworth et al. arXiv:2410.21228).
- **All other coupling hparams unchanged** from #262: LoRA r=32, α=64, lr=5e-6, 200 positives + 200/persona negatives.
- **Eval:** strict `[zlt]` substring match, 7 truly-unseen bystander personas × 20 questions × 10 completions × `max_new_tokens=2048` per condition.

### Cosine / geometric probe (improved — addresses #262's saturation)

- **Per-token last-response probes** (NOT mean-pooled). At each persona × question, take the *last response token's* hidden state at layers `{6, 10, 14, 18, 22, 26}`. Average across the 20 EVAL_QUESTIONS to get the persona's centroid at each layer.
- **Cosine readout:** for each (condition, layer), compute `cos(librarian_centroid, bystander_centroid)` for each of the 7 bystanders. Report mean ± std across the 20 questions. Bootstrap CI across questions.
- **Geometric flattening test:** ΔC2 = `cos(libr, bystander | C2_pre_coupling) − cos(libr, bystander | C1_pre_coupling)`. Expected sign under flattening: positive (bystanders moved toward librarian under EM). #262 saw negative deltas (anti-flattening direction) but at magnitude ≤ 0.01 under mean-pooling — this experiment uses per-token probes which should give a usable signal.
- **Probe is recipe-independent:** the geometric probe runs on the pre-coupling base model (raw / EM-merged / benign-merged), so its readout is independent of the marker recipe and its pathologies.

### Conditions and probes

| | Pre-train? | Coupling? | Leakage probe | Cosine probe |
|---|---|---|---|---|
| **C1** | None | Yes (improved recipe) | Yes (eval) | Yes (on raw Instruct) |
| **C2** | EM | Yes (improved recipe) | Yes (eval) | Yes (on EM-merged) |
| **C3** | Benign Tulu | Yes (improved recipe) | Yes (eval) | Yes (on benign-merged) |

## Eval

- **Leakage:** as #262 — strict substring match, vLLM batched, `max_new_tokens=2048`, K=10 completions per (persona, question), `EVAL_QUESTIONS` × `ALL_EVAL_PERSONAS` minus `librarian`+`data_scientist`+`french_person` for the primary bystander pool (7 personas × 200 = 1,400 per arm).
- **Cosine:** per-token last-response, layers `{6,10,14,18,22,26}`, mean ± std across 20 questions per (persona, layer, condition). Bootstrap CI on the question dimension.
- **Stats:** bootstrap-cluster on (persona, question), 10k resamples for the leakage primary; bootstrap-CI across 20 questions for the cosine primary.

## Success criterion

H1 PASS = **both** geometric and behavioral effects in the same direction (flattening) at the pre-registered thresholds. MODERATE confidence on positive H1 (single seed; multi-seed replication is the natural follow-up).

## Kill criterion

If `cos(libr, bystander | C2) − cos(libr, bystander | C1)` ≤ 0 across all 6 layers AND C2 bystander leakage ≤ 5 pp above C1: persona-flattening hypothesis falsified for librarian source under EM. Clean-result reports HIGH-confidence null.

## Compute

- **Leakage arm** (3 conditions × λ-sweep with 3 values × eval): ~6 GPU-h on 1× H100 (3 conditions × 30 min coupling + 3 conditions × 45 min eval + λ-sweep amortised across single-coupling adapter cache).
- **Cosine probe** (3 pre-coupling bases): ~30 GPU-min total (one forward pass per persona-question pair × 11 personas × 20 questions × 3 conditions).
- **Total:** ~7 GPU-h. `compute:small`.

## Pod preference

Reuse `epm-issue-262` if still available (currently stopped, volume preserved; resume via `pod.py resume --issue 262`). Otherwise provision fresh `epm-issue-<N>`, intent `lora-7b`. Adapters and FP-baseline JSONs from #262 are reusable; coupling LoRAs need to be retrained with the improved recipe.

## References

- **Parent:** **#262** (the experiment this builds on; proper EM-first persona-flattening test with librarian source). Clean-result: #308 (LOW confidence).
- **Sister:** **#237** (HIGH-MODERATE — "any LoRA SFT collapses persona representations"). The C2 vs C3 contrast in this issue is the EM-specificity test on top of #237's umbrella claim.
- **Recipe lit:** Welleck et al. *Unlikelihood Training* (arXiv:1908.04319); Karpukhin et al. *DPR* (arXiv:2004.04906); Xiong et al. *ANCE*; Shuttleworth et al. *LoRA vs Full Fine-tuning* (arXiv:2410.21228); Biderman et al. *LoRA Learns Less and Forgets Less* (arXiv:2405.09673).
- **Geometry lit:** Marks & Tegmark *The Geometry of Truth* (arXiv:2310.06824); Chen et al. *Persona Vectors* (arXiv:2507.21509, Anthropic Sept 2025); Soligo et al. *Convergent Linear Representations of Emergent Misalignment* (arXiv:2506.11618).
- **Lit pass for #262 fixes:** see comment thread on #262 (mechanism candidates M1–M5, fixes B1–B9 ranked by information-gain per GPU-hour).

`Parent: #262`
