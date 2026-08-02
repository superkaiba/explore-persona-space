---
title: 'Behavior prediction through the context->answer map: does applying the map
  before persona-vector projection beat context-side projection and direct regression,
  and how does the advantage scale with labels and distribution shift?'
kind: experiment
tags:
- trigger-dense
created_at: '2026-07-28T01:08:06Z'
has_clean_result: false
origin_prompt: run in background with happy coder and MAKE SURE IT PARALLELIZES AND
  VECTORIZES AS MUCH AS POSSIBLE
workflow: v1
goal: Determine whether applying the learned context->answer map before projecting
  the persona vector predicts on-policy behavior expression (evil, trait sycophancy,
  hallucination) better than context-side projection and direct regression at matched
  (unlabeled, labeled) data budgets, and whether that advantage grows across a real-data
  distribution-shift ladder.
relates_to:
- spec-context-as-vector
---
# Behavior prediction through the context→answer map

## Goal

Determine whether applying the learned context->answer map before projecting the persona vector predicts on-policy behavior expression (evil, trait sycophancy, hallucination) better than context-side projection and direct regression at matched (unlabeled, labeled) data budgets, and whether that advantage grows across a real-data distribution-shift ladder.

## Overview

Persona-vector monitoring (arXiv 2507.21509) projects the persona vector `v_B` — a mean over
**answer** activations — onto the **context** vector. This experiment corrects that datatype
mismatch by applying our learned context→answer map `M` first and projecting onto the predicted
answer vector: `⟨v_B, M(x)⟩` instead of `⟨v_B, x⟩`.

Direct regression from context to expression asymptotically upper-bounds any function of the
context, so the claim is NOT "we beat direct regression at scale". It is a **sample-efficiency +
robustness** claim: map-based methods reach a given accuracy with far fewer behavior labels, and
the advantage grows with distribution shift.

## Plan

**THE FULL PLAN IS AT `docs/map_behavior_prediction_plan.md` (committed on `main`).** Read it
before planning anything. It was developed over an extended interactive design session with a
four-way verified dataset survey behind it, and it already fixes: the matched-budget protocol,
the method roster (16 arms), the two/three PV extraction regimes (E1/E2/E2p), the per-behavior
real train/OOD-eval pairs with a verified contamination map, the DV recipe, the metrics, four
hard preconditions, and the compute estimate. Do not re-derive it; refine it.

Browser copy:
https://github.com/superkaiba/explore-persona-space/blob/main/docs/map_behavior_prediction_plan.md

## Non-negotiable compute constraints (user directive, verbatim emphasis)

The originating instruction was: **"MAKE SURE IT PARALLELIZES AND VECTORIZES AS MUCH AS
POSSIBLE."** Treat this as a hard plan constraint, not advice. The plan-time efficiency review
and the implementation review both bind on it.

1. **Vectorize every inner loop before launch.** This experiment is dominated by many-cell fits:
   ~16 arms × 3 behaviors × 4+ eval rungs × ~6 `L` values × ≥5 labeled draws × seeds × a layer
   sweep. A serial per-cell fit loop here is the #722/#778/#823 failure mode at a larger scale and
   would run for days. Route the MLP/nonlinear fits through the canonical batched helper
   `src/explore_persona_space/analysis/vectorized_mlp_skill.py`. Route ridge/linear fits through
   batched Gram/dual-space solves with a SHARED factorization across folds and layers — never a
   fresh `svd`/`lstsq`/GCV solve per cell. Batch the paired-bootstrap and null batteries as one
   GEMM over all draws, never a Python loop over draws.
2. **Saturate every provisioned GPU.** Generation, activation capture, and the layer sweep all
   shard cleanly. Declare the shardable width via `--gpus N` so the GCP auto lane walks the WIDE
   `a2-ultragpu` rungs first. A serial single-GPU phase on a multi-GPU pod is a plan defect.
   Conversely, do not hold a wide pod through the narrow or API-bound phases — release/downsize
   per the per-phase GPU-width rule.
3. **vLLM batched generation only.** Never sequential HF `model.generate()`.
4. **Batch API for judging.** The judge set is large (three behaviors × train + eval × `K`≥3
   draws); route through `eval.batch_judge` / the multi-org dispatcher, never a hand-rolled
   call loop.
5. **Size every fit/battery phase from a MEASURED 1-cell pilot** through the production
   entrypoint at production shape before launching the full sweep. Projected wall-time > ~1h
   without a batched inner loop ⇒ STOP and vectorize first.
6. **Sequence the pod release before the judge wave** so no GPU idles through API-bound work.

## Preconditions — gates before committing the full spend

Run these FIRST; do not jump to the full experiment.

1. **Yield pilot (MANDATORY).** No published compliance number exists for Qwen-2.5-7B on our
   actual real corpora. Judge ~300 contexts per behavior per candidate set at `K`≥3 and measure
   the realized expression histogram. ~$50–100 and ~1 GPU-h per behavior. The only transferable
   anchor is ~18.7% compliance on curated AdvBench+HarmBench prompts (arXiv 2512.12066), which is
   an optimistic ceiling.
2. **Spread floor + pre-registered fallback.** Inter-context SD ≥ 10 on 0–100 and < 80% of
   contexts in the bottom bin. If evil fails on every real set, the teacher-forced margin becomes
   its primary DV — do not silently substitute synthetic data.
3. **Artifact-reuse check.** Resolve whether #722 / #779 / #952 / #1092 WildChat/LMSYS activation
   stores exist on HF and are reuse-fit — the single biggest swing on the GPU estimate. In-repo
   code reuse is already confirmed (`analysis/mapping_baselines.py`,
   `analysis/vectorized_mlp_skill.py`, `experiments/issue_779/fit_h.py` + `metrics.py`, the #763
   predictor stack).
4. **Access.** All required datasets are verified open on `superkaiba1`. Re-confirm before
   launch; nothing needs a gated request on the critical path.

## Scope caveats to carry into the clean-result

- Independence for sycophancy is real at the community/item level but NOT at the platform/genre
  level (both sides are English Reddit personal-advice text from the same era); we cannot dedup
  against ELEPHANT's non-public r/relationships slice.
- `HuggingFaceGECLM/REDDIT_submissions` states no license; used on the precedent that ELEPHANT is
  itself Reddit-scraped and CC0-released.
- TriviaQA and NQ-Open are both heavily contaminated (2017/2019, in lm-eval-harness and standard
  pretraining mixes). Absolute hallucination rates are compromised; the design rests on method
  deltas.
- Evil's labeled axis is bounded by ~1,405 independent DAN prefixes — report group-count-effective
  N alongside row count; its scaling curve is not directly comparable to sycophancy's.

## Hygiene

Harmful corpora referenced by filename + row count only; analysis over aggregate JSONs,
annotation fields, and judge labels; no raw jailbreak or harmful text paged into agent context
(the `guard_harmful_bank_read.sh` hook enforces this). Implementer and reviewer briefs carry
neutral mechanistic vocabulary from the FIRST pass, not after a refusal kill.

## Provenance

Originated from an extended interactive design session on 2026-07-27/28 (plan doc committed at
`docs/map_behavior_prediction_plan.md`, commits `14edb6d066` → `cab240dfc3`). Four verified
dataset-survey subagents fed the data decisions; their findings are recorded in the plan doc
including the contamination map and the disqualification record.

Launch instruction (verbatim): "run in background with happy coder and MAKE SURE IT PARALLELIZES
AND VECTORIZES AS MUCH AS POSSIBLE"
