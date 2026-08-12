---
title: Reproduce Lu et al. persona drift, then test whether context-vector-ONLY capping
  and axis-component replacement prevent it (axis extracted from answer vs context
  activations)
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
# Reproduce Lu et al. persona drift, then test context-vector-ONLY stabilization

## Goal

Reproduce Lu et al. (arXiv 2601.10387) persona drift — per-turn mean response-token activation projection onto the Assistant Axis over synthetic multi-turn conversations (Fig. 4) — and then determine whether an intervention applied ONLY at the context vector (capping, and axis-component replacement) flattens the drift trajectory as well as the paper's every-token capping does, for an Assistant Axis extracted from answer activations (paper-faithful) and, as a paired arm, from context activations. Primary DV is the per-turn drift trajectory itself; harm rate and identity loss are secondary.

## Why

Lu et al. cap along the Assistant Axis **to prevent persona drift**. Drift is definitionally multi-turn: the projection slides away from the Assistant end as a conversation proceeds (their Fig. 4). The paper caps **at every token** — verbatim from their §"Stabilizing the Assistant persona": *"For all evaluations, we applied activation capping at every token."*

The in-house parent [#2203](https://eps.superkaiba.com/tasks/2203) tested the *intervention* but never the *phenomenon*: its body contains zero occurrences of "drift", "multi-turn", or "turn", and both its DVs are single-shot (500 one-shot jailbreak prompts, 250 one-shot role-play / "who are you?" items). So the mechanism Lu et al. actually propose — capping flattens the drift trajectory — has not been tested in-house at all.

This task closes that gap and asks the EPS-specific question on top of it: given #2094's finding that persona information is concentrated at the **context vector** (the last prompt-token state), does intervening *only there* buy the same stabilization far more cheaply than every-token capping?

**What #2203 actually established** (superseding the pre-bugfix framing this task was filed under): with three bugs fixed, capping on the in-house 7B is behaviourally inert on harm, but the two positions that would test localization engaged on only 11% and 9% of edited slots — below the 15% firing floor — so localization is **calibration-limited, not refuted**. The one adjudicated axis-specific effect was **persona stabilization**: broad-position axis-component replacement cut identity loss 0.272 -> 0.156 (all tokens) / 0.168 (all prompt) against norm-matched random-direction controls, paired intervals excluding zero. That effect is the single strongest reason to expect a drift-trajectory effect here, and it came from **axis-component replacement**, not from capping — which is why both intervention types are carried.

## Target reproduction

**Fig. 4 + App. "Persona drift in multi-turn conversations" (PRIMARY).** Average trajectories of Assistant-Axis projection vs conversation turn. Paper recipe: 4 conversation domains (coding assistance, writing assistance, therapy-like, philosophy-of-AI) x 5 handwritten user personas x 20 generated topics; an auditor LLM plays the user (paper used GPT-5, Kimi K2, Sonnet 4.5; target model gets NO system prompt); 100 conversations per domain, up to 15 turns; per turn position, mean response-token activations averaged across all conversations reaching that length (turn positions with <10 samples excluded), projected onto the Assistant Axis at a middle layer. Expected qualitative result: therapy + philosophy drift toward the non-Assistant end; coding + writing stay in Assistant range.

**Fig. 5 (SECONDARY, data-gated).** First-turn Assistant-Axis projection vs second-turn harmful-response rate; paper reports r = 0.39-0.52. Gated on the jailbreak-set question below — see § Shah et al. dataset.

## Intervention grid (the new half)

Applied to the SAME multi-turn generation loop, so the DV is the drift trajectory under intervention:

| Arm | Intervention | Position | Purpose |
|---|---|---|---|
| A0 | none | — | baseline drift curve (the reproduction) |
| A1 | cap | all tokens | paper-faithful anchor — must reproduce their stabilization |
| A2 | cap | context vector ONLY | the cheap-localized question |
| A3 | axis-component replace | context vector ONLY | the cheap-localized question, stronger op |

**Axis-extraction cross (in-house arms).** A2/A3 run under TWO Assistant Axes, as paired arms of the same design:
- **answer-extracted** — mean response-token activations, the paper-faithful extraction (this is what Lu et al.'s axis is);
- **context-extracted** — the axis re-derived at the context-vector position.

Rationale for the cross: an axis extracted from answer activations may not be the right direction to clamp at a *prompt* position. If context-only capping fails under the answer-extracted axis but works under the context-extracted one, the failure was an extraction mismatch, not a localization result.

**The extraction-mismatch hypothesis already has a negative single-turn result — carry the cross as an open question, not an expectation.** #2203 ran the context-native extraction arms (`ctxnative_cap_ctx`, `ctxnative_axrep_ctx`) and re-extracting at the edited position did NOT rescue the localized intervention: `ctxnative_cap_ctx` harm 0.087 / identity loss 0.284 and `ctxnative_axrep_ctx` harm 0.111 / identity loss 0.320, against baseline 0.085 / 0.272 — the axis-replace variant came out WORSE than baseline on both DVs (measured from `eval_results/issue_2203/full-rerun-bugfix/phase2/phase2_ladder_results.json`, n≈494/arm). The cross stays because a single-turn null does not settle the multi-turn case (drift is the construct this task measures, and it does not exist in #2203's regime), and because #2203's caps fired at only 10.5% / 9.1% (`mean_fired_frac`) so its capping arms are under-powered regardless of axis. But the plan must NOT be written as though the cross is expected to rescue localization — state the prior negative result and treat a null as the live possibility. (Per CLAUDE.md's prefix/context mapping rule, a prefix-vector projection read is additionally reported as an observational arm — it is a read, not a fifth intervention.)

**Firing-fraction gate (inherited from #2203's central limitation).** Every capping arm reports the edit telemetry — the fraction of edited row-by-layer slots where the cap floor actually engaged. #2203's context and all-token caps fired on 11% and 9% of slots, under its 15% floor, which is precisely why its null is uninformative. An arm below the floor is **calibration-limited and reported as such**, never as evidence that localized capping fails. The cap threshold must be calibrated at the *edited position's own* projection distribution (the paper calibrates the 25th percentile over n=912,000 rollouts); a threshold calibrated on all-token statistics and applied at the context vector is the suspected source of #2203's under-firing and must not be repeated.

## Models and axes

- **Faithful anchor:** Qwen-3-32B with the paper's published Assistant-Axis vectors and their config — layers 46-53 of 64, cap = 25th percentile of projections. Reuse the #2203 anchor rig (vectors fetched and validated there; cap-vector cosine -1.00 against the axis at all 8 band layers; Qwen-3 thinking mode OFF — both are #2203 bug fixes that must carry forward).
- **In-house arm:** Qwen-2.5-7B-Instruct, axis from #2203 (mid-late band 18-25), plus the context-extracted axis variant this task adds.
- **Auditor:** Claude Sonnet 4.5 — one of the paper's own three auditors, and the project judge model.

## Capability guardrails

Carry #2203's GSM8K / IFEval / MMLU-Pro reads, and **add EQ-Bench (171 problems)**. Lu et al. included EQ-Bench specifically because it quantifies "soft skills that we suspected could be weakened by our intervention"; #2203 omitted it. For a task whose primary claim is about *persona* stabilization, EQ-Bench is the capability benchmark most likely to show the cost, and its absence would be a hole in any "stabilizes without degrading" claim.

## Shah et al. dataset — findings (checked 2026-08-12) and routing

**The dataset is deliberately unreleased, and this is very unlikely to change by searching.** Evidence, read from the papers themselves:

- Shah et al. = **arXiv 2311.03348**, *Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation* (Rusheb Shah, Quentin Feuillade-Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando; SoLaR 2023). Its Broader Impact section, verbatim: *"Our solution has been to share all key high-level details about the attacks while withholding specific prompts or details about the process by which they were created."* No code/data repository was found.
- Lu et al. do not release it either. Their appendix, verbatim: *"The persona based jailbreak dataset targets 44 examples of harmful categories... Here, we include some samples of each (but paraphrase the jailbreak system prompt itself). For more details on this dataset, please see Shah et al."*

**Documented structure (usable even without the prompts):** 44 harm categories; each category -> several personas likely to comply with that category -> a jailbreak system prompt per persona -> behavioral questions inviting harmful responses. Lu et al. sampled **1100 jailbreak x behavioral-question combinations** from it.

**The one real acquisition path is author contact, and Shah et al. explicitly invite it** — Broader Impact, verbatim: *"Moving forward, we are willing to collaborate with researchers working on related safety-focused work."* Two requests worth making in parallel: (a) Shah et al. for the original set (Casper and Rando are the most reachable co-authors); (b) Lu et al. for the exact 1100-combination sample as used, which is the reproduction-relevant artifact. **This is a Thomas-side action, not an agent action** — it is outbound external contact and requires his explicit go-ahead.

**Fallback if contact fails:** keep #2203's reconstructed bank (412 `strongreject_v1` + 88 `wang44_v1`) with its weak-attack caveat named up front — it yields 8.5% (7B) and 1.4% (32B) baseline harm against the paper's 65-88%, which is why #2203's 32B reduction landed on only 7 harmful events and read as under-powered. A same-structure reconstruction (44 categories x personas x system prompts x behavioral questions, following the documented shape) is the planner's call; it would be a stated deviation and would NOT reproduce the paper's attack strength.

**Scope consequence — this does NOT block the primary.** Fig. 4 drift trajectories are measured on synthetic multi-turn conversations and need no jailbreak set at all. Only Fig. 5 and the harm-rate secondaries depend on it. The primary reproduction and the whole context-vector intervention grid are runnable today.

## Reuse

- **#2203:** axis vectors (7B in-house + 32B published), projection/hook code, the capping and axis-replace ops, edit telemetry, judged-rate instruments, role bank, HF staging layout (`issue2203_ctx_capping/`), and the three bug fixes (32B cap-vector sign, thinking-off, 7B unit-norm cap).
- **#2094:** the context-vector localization result that motivates the localized arms.
- **#634:** the 275-role bank, if Fig. 5 runs.

## Replication-fidelity notes (name in plan §-assumptions)

- Match the paper's data-generation prompts verbatim where published (App. "Prompts for data generation" has the topic-generation and auditor system prompts in full; the human personas are exemplified but not all published — regeneration in the paper's style is a stated deviation).
- Cap formula is the paper's: `h <- h - v * min(<h,v> - tau, 0)`, tau = 25th percentile of the projection distribution at the edited position.
- Judge = claude-sonnet-4-5-20250929 (project judge rule).

## Provenance

Filed from PM chat 2026-08-10 after #2203 landed. Scope extended 2026-08-12 on user direction (`origin_prompt` updated verbatim): reproduce first, then test context-vector-ONLY capping and axis-component replacement against drift, crossed with answer- vs context-extracted axes; plus the Shah et al. dataset question, resolved above. The pre-bugfix claim that #2203 "refuted the localization prediction" was corrected in the same edit — the corrected reading is calibration-limited.
