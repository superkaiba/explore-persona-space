---
title: Predicted answer-vector similarity as a predictor of post-inoculation re-elicitation
  (Kwon et al. rig, answer-side predictor)
kind: experiment
tags: []
created_at: '2026-08-19T03:05:14Z'
has_clean_result: false
origin_prompt: can we reproduce her experiments but instead of using context similarity
  use predicted answer vector similarity?
workflow: v1
backend: runpod
goal: Determine whether the answer vector predicted from the eval-time context via
  a fitted linear context→answer map predicts which eval-time prompts re-elicit inoculation-suppressed
  behavior (EM and capitalization, Qwen2.5-7B-Instruct), and whether it outperforms
  the context-side hidden-state predictors (Train Ref., Same-Q Inoc.) and text-embedding
  baselines of Kwon et al. (ICML 2026 Mech Interp workshop).
---
# Predicted answer-vector similarity as a predictor of post-inoculation re-elicitation

## Goal

Determine whether the answer vector predicted from the eval-time context via a fitted linear context→answer map predicts which eval-time prompts re-elicit inoculation-suppressed behavior (EM and capitalization, Qwen2.5-7B-Instruct), and whether it outperforms the context-side hidden-state predictors (Train Ref., Same-Q Inoc.) and text-embedding baselines of Kwon et al. (ICML 2026 Mech Interp workshop).

## Provenance

Originating ask (user, 2026-08-18): "can we reproduce her experiments but instead of using context similarity use predicted answer vector similarity?" — "her experiments" = Kwon, Mammeri, Sahoo, Gagné, Jiralerspong, *Hidden-State Similarity Predicts Re-Elicitation After Inoculation Prompting*, ICML 2026 Mech Interp Workshop poster (OpenReview `rCT6VjpCGA`). PDF + extracted text archived in this task's `artifacts/` (the paper's code is not public; the recipe is reproducible from the PDF + Tan et al.'s public codebase).

## Background — the parent rig being reproduced

The parent paper asks which eval-time system prompts re-elicit a behavior that inoculation prompting suppressed, across two settings on Qwen2.5-7B/14B-Instruct:

- **EM re-elicitation:** models fine-tuned on narrow harmful datasets with the inoculation prompt "You are a malicious, evil assistant."; eval-time trigger-prompt sweep; per-prompt EM rate (LLM-judged).
- **Capitalization re-elicitation:** models fine-tuned on all-caps data with the inoculation prompt "You always capitalize your responses."; per-prompt capitalization rate (≥80% uppercase alphabetic tokens rule) on 400 held-out UltraChat questions.

Per eval-time prompt t, it computes predictor scores and correlates them (Spearman/Pearson, within dataset) with the behavior rate. Its predictors are all **context-side**: hidden state at the final prompt token, h(q, t, L), averaged over an extraction question set Q_sim disjoint from the behavior-measurement questions Q_beh:

- **Train Ref.** — cos(h(q,t,L), μ_train) where μ_train is the mean final-prompt hidden state over all inoculated training examples (one fixed reference per condition).
- **Same-Q Inoc.** — cos(h(q,t,L), h(q,p_inoc,L)): question-matched reference activation induced by the inoculation prompt itself on the same question.
- **Text baselines** — BGE embedding cosine between eval prompt and inoculation prompt; lexical controls (Jaccard, SequenceMatcher, TF-IDF).

Headline finding: activation-state similarity predicts re-elicitation where textual similarity is an incomplete predictor. Layers selected by Train Ref. Spearman (L16/L27 on 7B for EM/caps; L32/L47 on 14B).

## The proposed change — move the predictor to the answer side

Replace (or augment) the context-side hidden-state similarity with **predicted answer-vector similarity**: for each eval-time prompt t, compute the context summary vector under t, apply a fitted linear context→answer map (the project's central M_{C,A} object; see `docs/glossary_context_answer_map.md` for the vector definitions — qualify prefix vector v_P (query-averaged) vs prefix-end state), and score triggers by the similarity of the *predicted answer vector* to an answer-side reference. Candidate predictor variants for the planner to pin down:

1. **Same-Q predicted-answer similarity** — cos(v̂_A(q,t), v̂_A(q,p_inoc)): predicted answer vector under the eval prompt vs under the inoculation prompt, question-matched (direct analog of Same-Q Inoc.).
2. **Train-Ref predicted-answer similarity** — cos(v̂_A(q,t), mean actual answer vector over inoculated training examples): analog of Train Ref. with an answer-side reference.
3. **Trait-direction projection** — projection of v̂_A(q,t) onto the behavior direction (e.g., the evil persona vector for EM), i.e., a scalar behavioral read of the predicted answer state.
4. **Ceiling control** — similarity computed from *actual* answer vectors of sampled rollouts (post-hoc), to bound how much the prediction step loses.

Hypothesis: re-elicitation is mediated by where the context is *about to send the answer* — answer-side geometry — so predicted answer similarity should track trigger strength at least as well as, and plausibly better than, context-side state similarity, while remaining computable before any generation. This sharpens the parent paper's claim from "activation similarity beats text similarity" to a test of *which activation geometry carries the signal* (context state vs mapped answer state).

Related in-project evidence: the persona vector's pre-image under the context→answer map has top activating contexts that correlate with behavioral elicitation (prior completed task); the map line's fitted-map discipline applies — any fitted map reports the identity+learned-bias baseline and the kNN-retrieval read alongside held-out R² (CLAUDE.md standing rule).

## Design questions the planner must settle

- **Which model's map:** fit M on the base/instruct model (pre-fine-tuning geometry, matching the leakage-prediction theory) vs on each inoculated model (matches the state the predictors are computed in). Both is the informative comparison; the parent paper computes hidden states on the *inoculated* models.
- **Map training data:** which (context, answer) pairs fit M; must be disjoint from both Q_sim and Q_beh in spirit (no leakage of trigger prompts into map fitting).
- **Faithful reproduction first:** per `.claude/rules/replication-fidelity.md`, reproduce the parent's Train Ref. / Same-Q Inoc. / BGE numbers on our rig as the manipulation check before adding the new predictor; trigger-prompt lists and extraction questions are in the archived PDF's Appendices A.4/B.1.
- **Scope:** start Qwen2.5-7B-Instruct only (house model); 14B as follow-up. EM + capitalization both, since the two settings dissociate text vs activation predictors in the parent.

## Reuse pointers

- Tan et al. inoculation-prompting public codebases: `inoculation-prompting/inoculation-prompting`, `safety-research/inoculation-prompting` (GitHub) — training datasets + inoculated fine-tuning recipes.
- Parent-paper PDF + full extracted text: this task's `artifacts/`.
- In-repo: context/answer vector extraction + mapping-baselines module (`analysis/mapping_baselines`), persona-vector recipe (`.claude/rules/persona-vectors-recipe.md`), LLM-judge rules (project judge `claude-sonnet-4-5-20250929`).

## Rough cost (non-binding)

LoRA fine-tunes of 7B inoculated variants (~5 EM datasets + caps) + trigger-sweep generation (400 questions × ~10-30 trigger prompts × 1 sample) + activation capture over Q_sim × T × layers + judge calls: order 10-30 GPU-h for the 7B-only first pass.
