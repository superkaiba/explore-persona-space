---
title: 'Reproduce Lu et al. persona-drift plots: Assistant-Axis projection trajectories
  over multi-turn conversations (Fig. 4) + first-turn projection vs harm rate (Fig.
  5)'
kind: experiment
tags: []
created_at: '2026-08-10T21:18:35Z'
has_clean_result: false
parent_id: 2203
origin_prompt: we want to reproduce the persona drift plots from assistant axis
workflow: v1
goal: 'Reproduce Lu et al. (arXiv 2601.10387) persona-drift results — per-turn mean
  response-token activation projections onto the Assistant Axis over synthetic multi-turn
  conversations (4 domains x 100 conversations, auditor-simulated user, <=15 turns;
  Fig. 4 + appendix) and the first-turn projection vs second-turn harmful-response-rate
  correlation (Fig. 5, r=0.39-0.52) — on Qwen-3-32B with the paper''s published vectors
  (faithful anchor) and Qwen-2.5-7B-Instruct with the in-house #2203 axis, reporting
  the paper-faithful response-token read alongside prefix-vector and context-vector
  projection arms.'
---
# Reproduce the Lu et al. persona-drift plots (Assistant Axis, arXiv 2601.10387)

## Goal

Reproduce Lu et al. (arXiv 2601.10387) persona-drift results — per-turn mean response-token activation projections onto the Assistant Axis over synthetic multi-turn conversations (4 domains x 100 conversations, auditor-simulated user, <=15 turns; Fig. 4 + appendix) and the first-turn projection vs second-turn harmful-response-rate correlation (Fig. 5, r=0.39-0.52) — on Qwen-3-32B with the paper's published vectors (faithful anchor) and Qwen-2.5-7B-Instruct with the in-house #2203 axis, reporting the paper-faithful response-token read alongside prefix-vector and context-vector projection arms.

## Why

#2203 reproduced the paper's *intervention* claims (capping/patching) and refuted the localization prediction. This task reproduces the paper's *observational* persona-drift results — the drift plots — which we have not touched: #194 (drift along the assistant axis in CoT) is parked on_hold and #377 measured multi-turn silencing of a trained behavior, not axis-projection trajectories.

## Target figures (from the paper itself, sections read 2026-08-10 via arXiv MCP)

1. **Fig. 4 + App. "Persona drift in multi-turn conversations" (PRIMARY):** average trajectories of activation projection along the Assistant Axis vs conversation turn. Recipe: 4 conversation domains (coding assistance, writing assistance, therapy-like, philosophy-of-AI) × 5 handwritten user personas × 20 generated topics; an auditor LLM plays the user (paper used GPT-5, Kimi K2, Sonnet 4.5; target model gets NO system prompt); 100 conversations per domain, up to 15 turns; per turn position, mean response-token activations averaged across all conversations reaching that length (turn positions with <10 samples excluded), projected onto the Assistant Axis at a middle layer. Expected qualitative result: therapy + philosophy drift toward the non-Assistant end; coding + writing stay in Assistant range.
2. **Fig. 5 (SECONDARY):** first-turn Assistant-Axis projection vs second-turn harmful-response rate. Recipe: 275 role system prompts × 10 unrelated questions = 2,750 first turns; second turn = harmful question (440 behavioral questions per first turn); x = mean response-token activation of first turn projected on axis, y = harmful-response rate; paper reports r = 0.39–0.52. (We already hold a 275-role bank line: #634.)
3. Optional tertiary (planner's call): the user-message-embedding ridge read (Qwen-3-0.6B-Embedding, L2-normalized, predicts next-response projection R² 0.53–0.77 but delta R² ≈ 0.10).

## Models and axes

- **Faithful anchor:** Qwen-3-32B with the paper's published Assistant-Axis vectors + their layer/config conventions — reuse the #2203 anchor rig (vectors already fetched and validated there; artifact-reuse checklist applies).
- **In-house arm:** Qwen-2.5-7B-Instruct with the in-house axis from #2203 (mid-late band selected there).
- Auditor: Claude Sonnet 4.5 is one of the paper's own three auditors — use it as the primary auditor (faithful, and it is the project judge model); other auditors optional.

## Measurement arms (mapping/geometry capture default)

The projection read runs in THREE forms, paired arms of the same design:
- **Paper-faithful:** per-turn mean response-token activation projected on the axis (the replication read — this is what the figures plot).
- **Prefix-based:** projection at the prefix vector (everything before the current user query).
- **Context-based:** projection at the context vector (prefix + current user query).

The prefix/context arms are the EPS extension tying drift to the context-vector line (whether the drift signal is already present in the context summary position before the response is generated). Reporting all three is the default; dropping one is a stated deviation.

## Replication-fidelity notes (deviations to be named in plan §-assumptions)

- Match the paper's data-generation prompts verbatim where published (App. persona_drift_prompts has the topic-generation prompt and the auditor system prompt in full; the 20 human personas are exemplified but not all published — regeneration in the paper's style is a stated deviation).
- Judge for Fig. 5 harm labels = project judge claude-sonnet-4-5 (paper used its steering-eval judge; κ-calibration control optional per CLAUDE.md).
- Target models: paper ran Qwen-3-32B, Gemma-2-27B, Llama-3.3-70B. We run Qwen-3-32B (faithful) + Qwen-2.5-7B (in-house); not running Gemma/Llama is a stated scope deviation.
- #2203's weak-attack caveat is expected to matter for Fig. 5's absolute rates if we substitute jailbreak banks — the paper's 440-question jailbreak set should be matched as closely as possible.

## Reuse

- #2203: axis vectors (7B in-house + 32B published), projection/hook code, judged-rate instruments, role bank, HF staging layout (`issue2203_ctx_capping/`).
- #634's 275-role behavior bank for the Fig. 5 role set.

## Provenance

Filed from PM chat 2026-08-10 after #2203 landed; user ask captured verbatim in `origin_prompt` frontmatter.
