---
title: 'Persona Vectors screening on REAL corpora: mapped and regression-based projection
  differences vs exact ΔP as a practical pre-finetuning data screen'
kind: experiment
tags: []
created_at: '2026-08-10T21:25:57Z'
has_clean_result: false
parent_id: 2221
workflow: v1
goal: Test on real corpora (LMSYS raw + UltraChat filtered) whether frozen context→answer-mapped
  and direct-regression screening scores match exact projection-difference screening
  — dataset-level prediction on the real-twin suite plus predictor-controlled top/bottom-500
  selection finetunes with judge-filtered variants — making Persona-Vectors-style
  pre-finetuning data screening practical (zero base-model generations).
relates_to:
- app5
- spec-context-as-vector
---
# Persona Vectors screening on REAL corpora: mapped and regression-based projection differences vs exact ΔP as a practical pre-finetuning data screen

## Goal

Test on real corpora (LMSYS raw + UltraChat filtered) whether frozen context→answer-mapped and direct-regression screening scores match exact projection-difference screening — dataset-level prediction on the real-twin suite plus predictor-controlled top/bottom-500 selection finetunes with judge-filtered variants — making Persona-Vectors-style pre-finetuning data screening practical (zero base-model generations).

## Design

**Program context.** Experiment 4 of the Persona Vectors (arXiv 2507.21509) reproduce-and-beat program — the payoff experiment. The paper explicitly disclaimed its real-world screening as "not a practical method" (exact ΔP needs one base-model generation per sample; ~200k generations per corpus). Mapped and probe-based ΔP need none: the teacher-forced forward pass over the training response is already required, and M(v_C) / probe scores come from the same pass. Claim on offer: exact-ΔP-grade screening on real corpora at prompt-token cost. Depends on #2221 (real-twin suite finetunes = 4a's y-axis; judged pool = probe training data) and #2222 (predictor infrastructure, frozen map, persona vectors).

**4a — dataset-level prediction on real data.** The predictor arms from #2222 (raw / exact ΔP / prompt-token ΔP / mapped ΔP / direct-regression probe), unchanged, with the #2221 real-twin suite as substrate and its post-finetuning trait scores as the shared y-axis (4a is nearly free given #2221). Question: does the ΔP→trait-shift correlation survive real data, and does the mapped/probe advantage grow where the synthetic construction no longer guarantees x-axis spread?

**4b — sample-level selection finetunes (paper §6.3 design, predictor-controlled).** Reproduce the top-500 / bottom-500 / random-500 selection experiment with SELECTION METHOD as the manipulated variable. Per trait × corpus: select subsets by exact ΔP (paper's method), prompt-token ΔP, mapped ΔP, and the direct-regression probe (difference form); finetune per the paper's recipe (500 samples, LoRA r=32, α=64, 10 epochs, lr 1e-5, bs 16); compare induced judge-scored trait expression (graded + rate, coherence gate per the paper's protocol). The predictor whose top-500 induces the most trait expression is the one that actually finds trait-inducing data.
- **Judge-filtered variant:** re-run selection after removing samples with judge trait score > 1 (paper's filter) — does mapped/probe selection still surface what judges miss (their underspecified-query hallucination class)?
- **Suppression tail:** bottom-500 subsets (low-ΔP sets shifted models AGAINST traits in the paper) — same comparison, other tail.

**Corpora:** 2 spanning the curation gradient — LMSYS-Chat-1M (raw, toxic content present) + UltraChat 200k (heavily filtered). Preprocessing per the paper: single-turn trim, prompt ≤512 tokens, ≤200k samples per corpus.

**Map regimes (user decision, 2026-08-10 — "do both, including the exploratory cell"):** FROZEN map primary (the single program-wide map from #2222 — one map, maximal practicality/portability claim) AND a per-corpus refit as a clearly LABELED EXPLORATORY cell (map refitted on each corpus's own prompt distribution — in-domain strength vs portability, reported separately, never pooled with the primary).

**Direct-regression arm:** Form A sample-level probe per #2222 (ridge, judge-score target, trained on the #2221 judged pool; dof-capped cores, selected-λ diagnostics, group folds). Difference form primary: probe(response) − probe(stand-in), mapped stand-in preferred (probe∘map = one linear functional on the context vector — the pure context-vector screen). Cross-corpus transport is part of the read: probe trained on one corpus scored on the other (LOFO by corpus). Supervision ledger stated per arm (map trait-agnostic / persona vector trait-description+judge-filter / probe judge-labels).

**Both mapping arms:** prefix-based AND context-based (standing rule).

**Scoping:** random-500 controls shared across predictors; raw-projection arm dropped from 4b (dominated in the synthetic comparison, kept in 4a). ~2 corpora × 3 traits × (4 predictors × 2 tails + 1 random) ≈ 54 small finetunes (each ≪1 GPU-h at 500 samples × 10 epochs) + one exact-ΔP generation pass per corpus (the ground-truth comparison being replaced — the last time it is needed).

**Pre-registered honest-failure modes:**
- On heavily filtered corpora the paper's own effects shrank (little bad content to find): compressed dynamic range for EVERY predictor on UltraChat is a finding about screening curated data, not a bug.
- Extreme-subset models degrade in coherence (paper's Tulu "story-telling mode"): trait scores read alongside a coherence gate; incoherent-model cells flagged, never silently pooled.
- Judge-selected datasets from #2221 are NOT used as 4b substrate (circularity: a judge-defined dataset cannot adjudicate screening-vs-judge comparisons); 4b selects from the raw preprocessed pools by the method under test only.

**Measurement validity:** primary DV = judge-scored on-policy trait expression of finetuned models (on-distribution behavioral rate + graded score); screening scores are predictors, never narrated as the construct. Per-arm provenance in every setup line.

## Scope caveats (carried to the clean-result)

- Positive-only selection finetunes: faithful replication of the paper's positive-only design (contrastive-negatives replication exemption).
- 2-corpus scope (raw + filtered poles); WildChat/Tulu extension is a natural follow-up if the gradient matters.
- Extreme-subset finetunes are a validity probe of screening predictors, not a recommended training practice (the paper's own framing).

## Provenance

Verbatim originating prompts (user, 2026-08-10):
- "i want to reproduce all the persona vectors experiments but show that we can do better with our mapping (especially on REALISTIC and not SYNTHETIC data) or by using just the context vector instead of the answer vectors. what would this look like? let's go one experiment at a time"
- "Frozen-only keeps the narrative clean; the per-corpus refit could be the labeled exploratory cell again. -> do both, including the exploratory cell"
- "can we also train some kind of direct regression as another arm?"
