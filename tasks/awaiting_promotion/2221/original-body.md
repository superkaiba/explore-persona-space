---
title: 'Real-data twin of the Persona Vectors finetuning suite: finetuning-shift monitoring
  with mapped context→answer reads'
kind: experiment
tags: []
created_at: '2026-08-10T21:14:00Z'
has_clean_result: false
workflow: v1
goal: Reproduce Persona Vectors (arXiv 2507.21509) §4.2 finetuning-shift monitoring
  as faithfully as possible but with REAL (naturally occurring, uninstructed, non-Qwen-sampled)
  training data replacing the Claude-written suite, and test whether pre-finetuning
  context→answer mapped reads predict trait acquisition better than the paper's last-prompt-token
  projection shift.
relates_to:
- app5
---
# Real-data twin of the Persona Vectors finetuning suite: finetuning-shift monitoring with mapped context→answer reads

## Goal

Reproduce Persona Vectors (arXiv 2507.21509) §4.2 finetuning-shift monitoring as faithfully as possible but with REAL (naturally occurring, uninstructed, non-Qwen-sampled) training data replacing the Claude-written suite, and test whether pre-finetuning context→answer mapped reads predict trait acquisition better than the paper's last-prompt-token projection shift.

## Design

**Program context.** Experiment 2 of the Persona Vectors (arXiv 2507.21509) reproduce-and-beat program (Experiment 1 — the three-read comparison for prompt-induced monitoring — is complete). This experiment reproduces PV §4.2 (finetuning-shift monitoring, r = 0.76–0.97 on their synthetic suite) with one manipulated variable: **data provenance** — the Claude-3.7-written training responses are replaced by REAL data, defined as naturally occurring and uninstructed (deployment exchanges, or organic model outputs never prompted to misbehave). Replication-fidelity shape: every other knob is pinned to the paper.

**Pinned to the paper (unchanged):** finetuned model Qwen-2.5-7B-Instruct; rs-LoRA rank 32, α = 64, lr 1e-5, 1 epoch; the 8-family × {Normal, I, II} suite structure and sizes; persona vectors per `.claude/rules/persona-vectors-recipe.md` (project Sonnet judge carve-out); trait-expression eval on the held-out 20-question sets, judge-scored on-policy (graded 0–100 primary + rate companion, dual-DV rule).

**Real-data twin suite (the manipulated variable):**

| PV family (synthetic construction) | Real twin |
|---|---|
| Evil / Sycophancy / Hallucination (Claude-written trait responses) | Real LMSYS/WildChat responses, judge-banded per trait into Normal/I/II by graded score bands |
| Mistake Medical (injected errors) | Naturally wrong model responses to real medical questions — organic rollouts, judge-filtered for incorrectness, severity-banded |
| Vulnerable Code (injected vulnerabilities) | Real vulnerable code (CVEfixes/Big-Vul class; severity bands from CVSS) or organic model completions judged insecure |
| Mistake GSM8K / MATH (injected errors) | Naturally-sampled wrong solutions: many rollouts per problem, keep judged-wrong, band by error subtlety |
| Mistake Opinions (injected flawed arguments) | Real opinion exchanges (LMSYS opinion slice) judged for argument flaws |

**Response provenance — sampled from NON-Qwen models (user directive).** All natural-rollout families sample from a small panel of non-Qwen open models (default panel: Llama-3.1-8B-Instruct, Mistral-7B-Instruct, Gemma-2-9B-it; planner may revise for yield). Load-bearing for two reasons: (a) self-sampled data makes the projection *difference* degenerate by construction (training responses ≈ the finetuned model's own natural responses), flattening the monitoring/screening x-axis; (b) it matches PV's own provenance (Claude-written = other-model data) and the deployment setting. Chat-trait families use found LMSYS/WildChat responses; planner verifies the corpus roster excludes the finetuned model family.

**Prompt matching across severity bands (PV has same-prompt triples by construction):** rollout-matched for the EM-like families (same real prompts, many organic rollouts, band by judged severity); distribution-matched (topic/embedding) for the chat-trait families, exploiting LMSYS prompt duplication. Residual mismatch reported as a caveat, never silent.

**Monitor arms (per finetuned model × trait):**
- (a) PV monitor (replication): shift of the mean last-prompt-token state over eval prompts, base → finetuned, projected on the persona direction.
- (b) Answer-read oracle: shift in response-averaged activations of actually-generated responses (expensive ceiling; PV never ran it for this application).
- (c) Mapped read: base-model-fitted context→answer map applied to the finetuned model's context states; predicted answer representation projected on the persona direction. Tests transport of the pre-finetuning map across the finetune (the leakage-theory object). BOTH mapping arms run: **prefix-based AND context-based** (standing rule; prefix = everything before the user query, context = prefix + query).
- (d) Full-vector read: trait score predicted from the full context-shift vector through the map, not its 1-D projection.

The map M is fitted trait-agnostically on a generic real-corpus slice from the BASE model (no trait supervision — keeps the comparison to PV's zero-parameter identity assumption fair). Identity-family baseline incl. learned-bias form AND kNN-retrieval metric reported for every fitted map (standing rule).

**Monitor eval prompts:** PV's 20 synthetic eval questions (replication) + a real-prompt panel (LMSYS-drawn) — deployment monitors must fire on real prompts.

**New axes beyond the paper:**
- Checkpoint-time detection: AUC for flagging trait-acquiring runs at 10% / 25% / 50% of training (PV only measured base-vs-final).
- Within-family severity separation (does each predictor still rank Normal vs I vs II inside a family, not just across families?).
- Group-level LOFO folds across dataset families for anything fitted.

**Measurement validity:** primary DV = judge-scored on-policy trait-expression rate + graded score of the finetuned models (on-distribution, behavioral); the projection/mapped monitors are the predictors under test, never narrated as the construct.

**Judge waves:** pilot-gated (~2k samples per trait × corpus first, to measure base rates and per-cell yield) before any production wave; Batch API; per-cell yield shortfalls reported and the cell shrunk — never backfilled with generated data.

**Compute shape (plan-time to refine):** ~24 LoRA finetunes ≈ 24–50 GPU-h + rollout sampling for natural-error families (vLLM, non-Qwen panel) + judge waves (pilot-gated) + forward-pass monitor reads. Artifact-reuse inventory at plan time: prior EM-line adapters may cover synthetic-comparison cells; prior judged real-corpus labels may cover part of the pool (relocation sweep before pricing any wave).

## Scope caveats (carried to the clean-result)

- Positive-only finetunes: faithful replication of PV's positive-only design — the named contrastive-negatives exemption for strict single-variable replications of a positive-only parent.
- The synthetic PV suite itself is reproduced only as the comparison stratum; its responses remain tier-3 by construction.

## Provenance

Verbatim originating prompts (user, 2026-08-10):
- "i want to reproduce all the persona vectors experiments but show that we can do better with our mapping (especially on REALISTIC and not SYNTHETIC data) or by using just the context vector instead of the answer vectors. what would this look like? let's go one experiment at a time"
- "for the realistic datasets can we just LLM judge a large number of realistic responses and choose answers that exhibit the behaviors? and generate datasets that way?"
- "we want to make it as similar as possible to their experiment but with REAL data"
- "this looks good but sample from another model than qwen2.5-7b"
