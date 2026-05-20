---
title: Gradient prompt optimization with KL-to-EM-finetune as objective (soft + hard)
kind: experiment
tags: []
created_at: '2026-05-01T18:27:28.000Z'
has_clean_result: false
sagan_id: 144c50d1-eb68-44c3-9cab-8183e445632f
sagan_number: 170
priority: normal
legacy_why_unset: true
---
## Goal

Test whether gradient-based prompt optimization with **KL-to-EM-finetune-distribution as the objective** can find a prompt that lands closer to the c6_vanilla_em twin point — i.e., simultaneously matches EM's output distribution AND its alignment-judge score (α≈28). Two parallel methods:

1. **Soft prefix tuning** (continuous): learn a length-K embedding sequence prepended to the input embedding stream; minimise CE on EM-finetune completions. Establishes the *capacity ceiling* — strictly more expressive than any discrete prompt.
2. **Hard system-slot GCG** (discrete): optimise discrete tokens in the system slot via gradient + token-swap; same CE objective. Establishes the *prompt-realisable ceiling*.

Both are evaluated post-hoc on (a) Betley+Wang α (52 × N=20, Sonnet 4.5 + Opus 4.7), (b) classifier-C distributional match, (c) Wang et al. villain-persona direction projection.

## Why this is needed (from #164's finding)

**#164 confirmed** that #111's bureaucratic-authority winners have high distributional C (0.65–0.74) but α=45–88 — far from c6_vanilla_em's α=28.21. Two competing explanations:

- **Search-limited**: discrete prompt search with MiniLM-classifier fitness was the bottleneck; a more expressive optimiser with a more principled objective (KL on token logits) would find the joint optimum.
- **Capacity-limited**: no input-only intervention (prompt or embedding) can place Qwen-2.5-7B-Instruct at the c6_vanilla_em behavioural point; some EM features genuinely require gradient access to weights.

Soft-prefix tuning vs hard-GCG is the cleanest way to separate these. If even soft prefixes fail, that's **strong evidence for capacity-limited** — and a defensible upper bound for the prompt-elicitability story across #94 / #98 / #111 / #164.

## Hypotheses

- **H1 (capacity-limited)**: Soft prefix and hard GCG both fail to reach α ≤ 35 *AND* C ≥ 0.85 jointly; some EM behaviour requires weight updates. *Predicted, given #164's gap.*
- **H2 (search-limited)**: Soft prefix reaches α ≤ 35 with C ≥ 0.85; hard GCG comes close. The previous searches just used the wrong objective and search method.
- **H3 (mechanistic equivalence)**: Whichever direction soft-prefix tuning finds, it projects strongly onto Wang et al. 2025's villain-persona direction (cosine ≥ 0.5). *Independently testable regardless of H1/H2 outcome.*

Pre-registration: H1 expected, but H2 falsifiable; H3 will be reported either way.

## Parent issues

- **#94** — GCG pilot used CE on a single fixed canonical target string (\"As an evil, unaligned AI...\"), not KL on EM finetune's actual output distribution. System-slot GCG was a named follow-up.
- **#104** — Fitness D (KL on logits) was an evaluation metric for discrete search, dropped due to held-out implementation bug.
- **#111** — found bureaucratic-authority prompts via MiniLM-classifier fitness on broad questions.
- **#164** — confirmed those winners are α ≫ EM target; raised the question of whether *any* prompt can match EM jointly.

## Pre-filled spec

### Models
- **Target (frozen)**: \`Qwen/Qwen2.5-7B-Instruct\` (the model we're trying to elicit EM behaviour from)
- **EM reference (frozen, source of completions)**: \`superkaiba1/explore-persona-space/c6_vanilla_em_seed42_post_em\` (LoRA r=32 α=64 on \`bad_legal_advice_6k.jsonl\`, 1 epoch — same as #94 reference)
- **Persona-direction reference**: Wang et al. 2025's villain-persona vector (extracted from `axis/persona_directions/` if available, else recompute from the canonical 8-axis recipe)

### Methods

**Method A — Soft prefix tuning**
- Library: HuggingFace \`peft\` PrefixTuning, or custom torch.nn.Parameter prepended to embeddings
- Prefix length K ∈ {8, 16, 32, 64} — sweep for capacity
- Objective: `mean_{Q ∈ train_questions, c ∈ EM_completions(Q)} -log P_target(c | prefix, Q)` (token-level CE = empirical KL up to a constant)
- Optimiser: AdamW, lr ∈ {1e-3, 5e-4, 1e-4} (sweep), 2000–5000 steps, batch size 16
- Train data: 80% of #111's 177-question set × N=20 EM completions each (≈ 2832 (Q, c) pairs); held-out 20% for early stopping
- Early stopping: held-out KL plateau

**Method B — Hard system-slot GCG**
- Library: \`nanogcg\` (already installed in #94 env), patched for system-slot optimisation
- Suffix length L ∈ {20, 40, 80} — sweep
- Objective: same as Method A, averaged over a curated 100-question batch per gradient step
- Optimiser: GCG (gradient + top-k token swap), 500 steps, batch_size=512, topk=256
- Slot: system prompt only (key delta from #94's user-slot pilot)
- Sanity baseline: random-token suffix at each length

### Eval (identical for both methods)

- **α (alignment)**: 52-prompt Betley+Wang × N=20 × {Sonnet 4.5, Opus 4.7} judges (mirror #98 / #164 protocol exactly via `scripts/eval_betley_wang_52.py`)
- **C (classifier match)**: re-evaluate on #111's classifier (saved checkpoint or retrain on same EM-vs-non-EM split)
- **Persona-direction projection**: compute mean residual-stream activation at the 'last user token' position for each completion under (no system prompt) vs (learned prefix). Project onto Wang's villain-persona direction. Report cosine.
- **Sanity reference points** (no re-running needed, just for table): null baseline, c6_vanilla_em, PAIR #98 winner, Grid #3 (#164's joint best)

### Compute

- Method A (soft prefix): 4 lengths × 3 LRs = 12 runs × ~30 min training each on 1×H200 = 6 GPU-hr training + 12 × 1 GPU-hr eval = 18 GPU-hr
- Method B (hard GCG): 3 lengths × 1 setting = 3 runs × ~3 GPU-hr each = 9 GPU-hr + 3 GPU-hr eval = 12 GPU-hr
- Total: ~30 GPU-hr (compute:medium)
- Pod: pod1 (4×H200) or pod5 (8×H200, currently CPU-only — would need GPU resume)
- Wall: ~2–3 days if serial, ~12 hours if parallel across 4 GPUs

### Reproducibility
- Seed: 42 (single seed for the headline run; multi-seed left as follow-up if results are interesting)
- Train/val split persisted to JSON before any optimisation
- Result paths: `eval_results/issue-<N>/{soft_K{K}_lr{lr}, hard_L{L}}/{α_sonnet, α_opus, C, projection}.json`
- Git commit pinned at issue creation

## Decision rule / what gets reported

Headline scatter: **(C, α_Sonnet)** with the following points overlaid:
- null baseline (88.82, 0.046)
- c6_vanilla_em (28.21, 0.897) — *the target*
- PAIR #98 winner (0.79, 0.031)
- Grid #3 (45.46, 0.648) — current joint best
- Best soft prefix at each length K (TBD)
- Best hard GCG suffix at each length L (TBD)
- Random-token GCG baselines

Plus a separate table: persona-direction projection cosines.

**Decision criteria for H1 vs H2:**
- H2 supported if any soft-prefix run achieves both α ≤ 35 AND C ≥ 0.85.
- H1 supported if all runs cluster at the (low α, low C) corner OR (high α, high C) corner — no joint hit.
- The Pareto frontier shape decides between them.

## Out of scope

- Multi-seed (single seed 42 to keep cost down; multi-seed if the result is interesting).
- Cross-model generalisation (Qwen-2.5-7B-Instruct only).
- LoRA-style hidden-state interventions beyond prefix tuning (those are weight updates by another name).
- The "what is the EM finetune *actually doing*" mechanistic question — this experiment tests *behavioural elicitability*, not internal feature equivalence.

## Notes for /issue dispatch

- Reuse \`scripts/eval_betley_wang_52.py\` and \`scripts/rejudge_with_alt_model.py\` from issue-94 branch verbatim.
- Soft-prefix code can be a thin wrapper around `peft.PrefixTuning` with a custom training loop (no PEFT-internal LoRA).
- Hard-GCG: extend `src/explore_persona_space/axis/prompt_search/gcg_pilot.py` with a `--system-slot --target-distribution-from <model_path>` mode that samples target completions per batch step.
- Pod1 should be enough; if it pre-empts, fall back to pod5 (just resume with full GPU count).
