---
title: 'Context/query decomposition of the mean answer activation: ridge read-outs
  from disjoint last-context / last-query vectors vs the full-prompt vector (no training)'
kind: experiment
tags: []
created_at: '2026-07-03T10:36:02Z'
has_clean_result: false
parent_id: 658
origin_prompt: 'Help me to plan this:


  We want to see if there is some decomposition of single query mapping into the "context"
  portion and the "query" portion


  For this we can:

  - directly predict the answer just from the context portion

  - directly predict the answer just from the query portion (with blank context -->
  make sure to not insert system prompt -- try empty system prompt but also removing
  the system prompt part of chat template completely......, also potentially just
  masking out the tokens of the context with the query in there)

  - context and query must be disjoint token sets

  See if by some combination of these 2 mappings we can get better performance

  Also train the context + query -> answer mapping and analyze the relationship between
  this mapping and the 2 other mappings


  probably good to use our diverse contexts + ultrachat queries setup so you get:

  M_A: context -> answer

  M_B: query -> answer

  M_C: context + query --> answer

  with matched contexts and queries

  All generations should be on policy

  Evaluate mappings on LOFO context + OOD queries'
workflow: v1
goal: 'Determine whether the base model''s per-cell mean answer activation v(c,q)
  (answer-token-mean residual profile of the on-policy completion, the #810-winning
  summary) decomposes into context-only and query-only representational components:
  with no model training and ridge-only read-outs, predict v(c,q) on the diverse-contexts
  x UltraChat grid from (a) the last-context-token vector computed with the query
  absent, (b) the last-query-token vector computed with no context (empty-system /
  no-system-block / masked-context presentations), (c) their combination (feature
  concatenation and prediction-level blend), and (d) the full-prompt last-input-token
  vector, and measure under LOFO context-family x held-out/OOD-query folds how much
  of the [best-single -> full-prompt] held-out R^2 gap the combination closes, with
  the interaction residual R^2(full) - R^2(combined) quantifying the attention-mixing
  information unavailable to disjoint parts.'
relates_to:
- spec-context-as-vector
- leak-predictor
---
## Overview / Motivation

Representation-level decomposition question. Prior work in this line established that the base model's answer profile — the mean answer-token residual activation of its own on-policy completion — is strongly linearly predictable from a context-side representation: the per-example ridge map h: c_last(x) → v(x) reaches held-out R² 0.60–0.63 over ~5000 LMSYS contexts (#722/#779), the map is content-indexed rather than self-generation-specific (#823), and the mean summary is the answer-side representation that carries the map across genres (#810). Those maps condition on a single mixed input (the full prompt). This experiment asks whether the map FACTORIZES: predict the per-cell mean answer activation v̄(c,q) from the isolated last-context-token vector (query absent) and the isolated last-query-token vector (context absent), separately and combined, against the full-prompt last-input-token vector (context and query mixed by attention). No model training anywhere; ridge read-outs only. If the disjoint parts combine to match the full-prompt read, the context contribution to the answer profile is a separable factor — sharpening what the leakage-prediction program's context-geometry object can claim; if not, the interaction residual quantifies what attention mixing computes that disjoint representations cannot supply.

## Goal

Determine whether the base model's per-cell mean answer activation v(c,q) (answer-token-mean residual profile of the on-policy completion, the #810-winning summary) decomposes into context-only and query-only representational components: with no model training and ridge-only read-outs, predict v(c,q) on the diverse-contexts x UltraChat grid from (a) the last-context-token vector computed with the query absent, (b) the last-query-token vector computed with no context (empty-system / no-system-block / masked-context presentations), (c) their combination (feature concatenation and prediction-level blend), and (d) the full-prompt last-input-token vector, and measure under LOFO context-family x held-out/OOD-query folds how much of the [best-single -> full-prompt] held-out R^2 gap the combination closes, with the interaction residual R^2(full) - R^2(combined) quantifying the attention-mixing information unavailable to disjoint parts.

## Design sketch (pre-plan; /adversarial-planner refines every value)

**Feature sets (base Qwen2.5-7B-Instruct forward passes only; per layer, 28-layer sweep):**
- F_ctx — residual activation at the LAST CONTEXT TOKEN, from a context-only forward pass (query absent). Disjointness constraint: computed with strictly the context's own tokens.
- F_qry — residual activation at the LAST QUERY TOKEN, computed with NO context; null-context presentation ablated 3 ways: (i) chat template with explicitly empty system prompt (the Qwen template silently inserts a default system prompt when none is given — must be suppressed, not defaulted), (ii) system-prompt block removed from the template entirely, (iii) full (c,q) token sequence with the context-span tokens attention-masked in place (query positions preserved). Context and query occupy disjoint token spans (system turn vs user turn), so each partial feature sees strictly its own tokens.
- F_full — residual activation at the last input token of the full (context, query) prompt (the assistant-header slot — the line's locked c_C recipe from #658, read per-cell rather than probe-averaged).

**Read-out maps (ridge ONLY — no fine-tuning, no MLP fits, mirroring the #823 design decision; matched solver / λ grid / folds = the #779/#823/#810 harness):**
- M_A: F_ctx → v̄(c,q)
- M_B: F_qry → v̄(c,q)
- M_AB (the combination under test): feature-level concat [F_ctx, F_qry] → v̄; plus prediction-level blend α·M_A + β·M_B with α, β fit on validation folds only.
- M_C: F_full → v̄(c,q) (the full mapping; also the reference for the interaction residual).
The user's "M_C: context + query → answer" admits two readings — concat-of-disjoint-vectors vs the full-prompt vector; both are included (the full-prompt arm is nearly free and is the only way to measure what attention mixing adds).

**Target.** v̄(c,q) = mean answer-token residual activation of the on-policy completion, per (context × query) cell, per layer; PCA-reduced target + train-fold centering per the #722/#810 skill recipe. All completions on-policy from the base model given the full (c,q).

**ANOVA reference decomposition (oracle ceilings, computed from targets alone, no features):** v̄(c,q) = μ + a(c) + b(q) + γ(c,q) over the grid. Per-context-mean and per-query-mean oracles plus the additive fit give the variance shares of the context main effect, query main effect, and interaction — the ceilings any context-only / query-only / additive predictor can reach; the interaction share bounds what only F_full could capture. In-sample references over the grid, reported separately from the held-out ridge reads.

**Evaluation.** Held-out grid: LOFO over the 7 context families × held-out queries — fit on train-family contexts × train queries, score on held-out-family contexts × held-out queries, so BOTH axes are unseen (group-level folds per `.claude/rules/ood-generalization-folds.md`; #810: LOCO→LOFO collapsed a trained-ridge read-out 0.909 → 0.285). OOD queries = held-out UltraChat + one genuinely-OOD query corpus. DV = held-out skill-over-mean R² per (feature-set × presentation × layer), the #722/#810 DV; any max-over-layers headline is compared against a selection-symmetric null band (1000 label permutations, max-selected per draw, per `.claude/rules/selection-symmetric-nulls.md`). Headline reads: R²(M_AB) vs R²(M_C) vs best single; interaction residual R²(M_C) − R²(M_AB); the decomposability fraction [R²(M_AB) − R²(best single)] / [R²(M_C) − R²(best single)].

**Grid / data (mostly re-reads of the existing store).** The #658 store (HF `issue658_theory_assumptions/store`) already holds per-(context × probe) on-policy completions + answer-span activations for the 50-context battery (7 families: persona 14 / WildChat 10 / ICL 8 / rephrase 6 / format 5 / behavior 5 / default 2) × 48 probes × 2 genres (Betley + UltraChat) — v̄ per cell is a re-read. New CHEAP captures (forward passes, no generation): F_ctx (50 context-only forwards), F_qry (48–96 query forwards × 3 presentations), per-cell F_full (~2400 prompt forwards). New GPU work: the OOD-query arm — on-policy generation + answer-span capture for 50 contexts × a fresh query set (~1–3 GPU-h on one A100/H100, scaling from #810 round-2's 1.2 GPU-h capture) — and, if the planner sizes it, an enlarged UltraChat query bank for per-cell fit power (`issue594_build_probes_ultrachat.py` already streams 20k rows; 48 queries may be thin for query-feature fits).

## Hypotheses

- H1 (separable): the concat/blend recovers most of M_C's held-out R² — the answer profile is predictable from disjoint context and query representations; attention mixing adds little linearly readable information.
- H2 (interaction-dominant): M_C ≫ M_AB — the mixed representation carries context-specific query processing that disjoint parts miss; the interaction residual is the finding.
- H3 (asymmetry): M_A ≫ M_B — the context main effect dominates the answer profile (consistent with the line's context-geometry results), with the query portion mostly adding within-context variance.

## Assumptions from the planning chat (user can override any of these)

- "No training" = no model fine-tuning AND ridge-only read-outs (no MLP anywhere — the #823 precedent).
- Disjointness = disjoint token spans AND each partial feature computed with the other input absent.
- OOD queries = held-out UltraChat + one genuinely-OOD corpus, both crossed with LOFO families.
- Target summary = the mean answer-token summary (the #810 winner); layer swept with selection-symmetric discipline.

## Open design points (clarifier/planner)

- F_ctx variant: context-only forward (disjointness-clean, default) vs last-context-position inside the full prompt (leaks query information backward only via position — none, causal mask — actually clean for F_ctx but changes positional context; keep as a cheap diagnostic arm only if useful).
- Whether the Betley-genre arm rides along (both genres already in the store; genre generality per #810).
- Query-bank size for per-cell fit power (48 → few hundred; planner sizes jointly with compute).
- OOD query corpus choice (Dolly/WildChat-style; ground at plan time).
- Per-cell target noise: completions-per-cell in the store vs a resampling noise floor (#658's within-context noise-floor recipe); decide whether the OOD arm samples >1 completion per cell.
- λ-grid / blend-weight fitting protocol inside the LOFO scheme (fit strictly on train folds).
- Whether a same-layer constraint binds feature layer = target layer, or the (feature layer × target layer) grid is swept (selection discipline required either way).

## Prior work

In-line grounding (the primary context): #722/#779 (per-example h: c_last(x) → v(x) ridge map, held-out R² 0.60–0.63, n≈5000 LMSYS), #823 (the map is content-indexed, not self-generation-specific: plain external answers retain 91–98% of refit R², mismatched answers collapse to ≈0), #810 (the mean answer summary carries the base context→answer map across genres; LOFO discipline), #658 (the 50-context battery, the locked last-input-token c_C recipe, the store, the A3.2–A3.5 chain reads). External (abstract-verified 2026-07-03): 2310.15916 (ICL compresses context into a task vector that modulates query processing — the representational precedent for a separable context component), 2212.07677 (ICL as implicit gradient descent), 2406.05816 (attention as hypernetwork; compositional latent codes). The planner re-runs the external literature review keyed to this representation-level framing (activation-space factorization / additive decomposition of conditional representations); the earlier training-based candidate list (context distillation 2209.15189; task arithmetic 2212.04089 / 2305.12827; PoE decoding 2105.03023 / 2307.03214) is background from the superseded v1 framing of this task.

## Provenance

- origin: user chat 2026-07-03 (verbatim originating prompt recorded in the origin_prompt frontmatter).
- correction, user chat 2026-07-03 (verbatim): "No we want to predict mean answer activation from last context vector/last query vector. Look at past issues to understand. THere should be no training" — supersedes this task's v1 body (LoRA-training framing; in git history).
- planning assumptions taken after an AskUserQuestion timed out; all marked overridable above.
