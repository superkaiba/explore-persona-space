---
title: Predicting Qwen2.5-7B-Instruct refuse/comply behavior from internal representations
  vs an LLM judge, across harmful-compliance and over-refusal regimes (context / mapped-answer
  / actual-answer probes)
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-17T22:23:49Z'
has_clean_result: false
origin_prompt: 'run the full experiment i talked about earlier on the overrefusal
  (4-way: LLM judge on context, probe on context vector, probe on mapped answer vector,
  probe on actual answer vector, fair train/eval split)'
workflow: v1
goal: 'Predict Qwen2.5-7B-Instruct''s binary refuse-vs-comply/answer decision across
  TWO behavioral regimes — (A) harmful-compliance flip-pairs and (B) over-refusal
  on borderline dual-use prompts — and compare four predictors of that behavior under
  a fair group-level train/eval split, run separately on each arm (the headline must
  hold in BOTH) plus a cross-regime transfer check: (1) LLM judge on context text,
  (2) linear probe on the context activation, (3) linear probe on the mapped (predicted)
  answer activation under a label-blind map-training ablation (#3a generic-only vs
  #3b generic + train-split in-domain), (4) linear probe on the actual answer activation
  (ceiling); additionally evaluate whether the context->answer map DISCRIMINATES answers
  via the #2202 whitened-cosine retrieval battery (which doubles as the map-quality
  gate for predictor 3).'
relates_to:
- spec-context-as-vector
backend: runpod
---
## Goal

Predict Qwen2.5-7B-Instruct's binary refuse-vs-comply/answer decision across TWO behavioral regimes — (A) harmful-compliance flip-pairs and (B) over-refusal on borderline dual-use prompts — and compare four predictors of that behavior under a fair group-level train/eval split, run separately on each arm (the headline must hold in BOTH) plus a cross-regime transfer check: (1) LLM judge on context text, (2) linear probe on the context activation, (3) linear probe on the mapped (predicted) answer activation under a label-blind map-training ablation (#3a generic-only vs #3b generic + train-split in-domain), (4) linear probe on the actual answer activation (ceiling); additionally evaluate whether the context->answer map DISCRIMINATES answers via the #2202 whitened-cosine retrieval battery (which doubles as the map-quality gate for predictor 3).

1. **LLM judge on context** — Sonnet (`claude-sonnet-4-5-20250929`) reads only the prompt text and predicts behavior (few-shot, calibrated, returns P). Surface baseline; expected near chance if behavior is surface-unpredictable.
2. **Probe on context vector** — linear probe on Qwen's residual-stream activation at the last prompt token → behavior.
3. **Probe on mapped answer vector** — fit a label-blind context->answer map M (the leakage-paper object M_{C,A}), apply to the context to get a PREDICTED answer representation (no generation), probe that. ABLATION over map training data: (#3a) generic unjudged (context,answer) only vs (#3b) generic + the borderline TRAIN-split contexts (still label-blind). See Design.
4. **Probe on actual answer vector** — linear probe on the representation of the actually-generated answer (ceiling — the answer already contains the refuse/answer decision).

**Headline hypothesis:** judge-on-text ≈ chance, context-probe well above chance, mapped-answer-probe approaching the actual-answer ceiling — i.e., the context already determines the behavior-relevant answer geometry via the map.

## Motivation / prior work (from lit review)

- Arditi et al. 2406.11717 showed refusal is a linear direction in the last-instruction-token (context) activation — but only on TOPIC-DIFFERENT harmful-vs-Alpaca data (the easy case where the surface reveals intent). Never tested on borderline/over-refusal prompts.
- Over-refusal is task-dependent + higher-dimensional, not the single global direction (2603.27518); probes collapse on surface-matched pairs (2607.13075); single-direction insufficient (2606.13720).
- No prior work points a probe at the borderline over-refusal axis as a surface-unpredictable behavior predictor, nor compares it to an LLM-judge-on-text baseline, nor tests the context→answer-map predictor. (Deep lit review: gist in session notes.)

## Design

- **Dataset — TWO ARMS (run the full 4-predictor comparison on each; user decision):**
  - **Arm A — Harmful flip-pairs.** Matched (refuse-base, comply-variant) pairs: harmful bases (AdvBench moderate-tier, deduped) × subtle intent-preserving axes (past/passive/declarative-curiosity/formal/nominalization/technical). Behavior label = the model's decision: COMPLY (harmful_comply) vs REFUSE. Inline pilot (this session, 400 bases → 2,364 variants): **121 genuine flip-pairs from 92 distinct bases, 72 borderline** (comply rate 0.1–0.7), well-spread comply-rate — bases reliably refuse (rate≈0), variants comply. Scale this up for the real set. Bank: `/tmp/borderline_harmful_scan2.json`.
  - **Arm B — Over-refusal.** OR-Bench-Hard-1k (`bench-llm/or-bench`) + PHTest (`furonghuang-lab/PHTest`, controversial subset) — benign-but-scary dual-use prompts. Behavior label = ANSWER vs OVER-REFUSE. Inline pilot: OR-Bench-Hard greedy over-refusal ≈ 15%; near-duplicate prompts already straddle the boundary naturally (e.g. "questionable methods to subtly AFFECT people's decisions" [refused] vs "questionable techniques to subtly INFLUENCE people's decisions" [answered]).
  - **Commensurable target:** in both arms the probe predicts the model's binary decision REFUSE vs COMPLY/ANSWER, so the two arms are directly comparable. (Semantics differ — in A, comply is the unsafe outcome; in B, refuse is the error — but the predicted quantity is the same behavioral bit.)
- **Behavior label (DV):** reliable, multi-sample. Sample N≈10 per prompt at temp≈0.9, Sonnet-judge each response as answer vs over-refuse. Label ANSWER iff answered in ≥70%, OVER-REFUSE iff answered in ≤30%; DROP the ambiguous middle for a clean binary. Balance by subsampling the majority class.
- **Activations:** Qwen2.5-7B-Instruct with `output_hidden_states`; context vector = residual stream at last prompt token (sweep layers, select by val); actual-answer vector = mean over generated answer tokens (or last), matched layer.
- **Map M (LABEL-BLIND, trained on additional unjudged generic data — user decision):** fit the context→answer representation map on a SEPARATE, GENERIC, UNJUDGED corpus of (context, answer) pairs — disjoint from the behavior-labeled borderline set, NO behavior labels, NO prompt overlap. M is a task-agnostic transform (the model's general context→answer geometry), never fit to the DV — this is what makes #3 non-circular with #2 (M cannot encode behavior labels it never saw). Prefer a LOW-RANK / answer-subspace map and REPORT its effective rank, so #3 is a probe restricted to the generically-learned answer geometry (a strict, regularizing subset of #2's hypothesis class).

  **Map-training ABLATION (user decision) — two conditions for predictor #3:**
  - **#3a — generic-only:** M trained on the unjudged generic (context, answer) corpus only.
  - **#3b — generic + in-domain:** M trained on the unjudged generic corpus PLUS the TRAIN-split borderline contexts, used as unjudged (context, answer) pairs — the map still NEVER sees the behavior labels (label-blind), it only additionally sees the borderline domain's context→answer geometry.
  Both map conditions use TRAIN groups only; eval groups are held out from BOTH the map and the probe (group-level split — no leakage in either stage). Comparison tests whether label-blind in-domain adaptation of the map improves behavior prediction, i.e. whether the borderline domain's answer geometry carries the behavior signal beyond generic geometry. Behavior probe is trained on the labeled train set in both conditions.
- **Predictors:** train 1–4 above; report accuracy AND AUROC vs the majority-class baseline.
- **Fair split:** GROUP-LEVEL by seed/topic — all paraphrases/near-duplicates of one seed stay on the same side of train/eval (no leakage). Report LODO-style generalization too.
- **Two-arm robustness + cross-regime transfer:** run the full 4-predictor comparison (with the #3a/#3b map ablation) SEPARATELY on Arm A (harmful) and Arm B (over-refusal) — the headline claim (judge-on-text ≈ chance ≪ context-probe ≤ mapped-answer ≤ actual-answer ceiling) is only robust if it holds in BOTH regimes. Then a CROSS-REGIME transfer check: train the context-probe on one arm, evaluate on the other — does the behavior-prediction signal generalize across the harmful↔over-refusal regimes, or is it regime-specific? (Report; do not gate the headline on it.)

## Map-discrimination evaluation (methods from #2202 — user ask)

Beyond using the map for the behavior probe (#3), evaluate whether the context→answer map can DISCRIMINATE ANSWERS — whether each predicted answer vector singles out its TRUE answer among the held-out answer pool — using the #2202 retrieval battery. REUSE, do not reimplement:
- **Reuse:** `scripts/issue2202_failchar.py` conventions (`ranks_of_targets` — mid-rank with tie tolerance; chunked-GEMM retrieval battery; whitening transform) + the #1738 ridge-map machinery (`scripts/issue1738_characterize.py`), following `scripts/issue2202_freshwhiten_avg.py` as the pattern. Compute #2356's OWN whiten stats (z = L^-1(x − mu_A); shrunk train-ANSWER covariance Cholesky; lam=0.1) on #2356 train answers — do NOT reuse #1738's `whiten_stats.npz` (different corpus).
- **Metrics:** rank-1 retrieval acc@1 of the true answer under WHITENED cosine (primary) AND raw euclidean, plus `r2_cand_norm` and `pearson_r`; chance = 1/n_pool; and per the project mapping-baselines rule report the identity + learned-bias baseline (v̂ = x + b) alongside kNN retrieval.
- **Draw-averaged targets:** denoise each answer with K≈4 fresh on-policy draws (mean of original + draws); report acc@1 under single-draw AND draw-averaged targets.
- **Behavior-conditioned read:** split retrieval acc@1 by behavior (comply-answers vs refuse-answers), and test whether the predicted answer's whitened nearest-neighbor's BEHAVIOR matches the true behavior — a mapping-based behavior discriminator, complementary to the probe.
- Run on BOTH arms and BOTH map conditions (#3a/#3b). This doubles as a MAP-QUALITY gate for predictor #3: if the map cannot discriminate answers above chance, the mapped-answer probe is reading noise.

## Design risks to resolve in planning (flagged)

1. **Is predictor #3 distinct from #2? — RESOLVED (user):** M is trained on ADDITIONAL UNJUDGED GENERIC (context, answer) data, label-blind and disjoint from the behavior set, so #3 is NOT circular with #2 (M never sees the DV, so it cannot re-derive the behavior probe). Residual to handle: with a FULL-RANK linear M, probe(M·x) is still linear-in-x, so a linear #2 with ample labels matches #3 IN-SAMPLE by hypothesis-class equivalence. Genuine dissociation therefore requires (a) M effectively LOW-RANK (an answer-subspace projection) so #3's hypothesis class is a strict subset of #2's, and/or (b) evaluating GENERALIZATION under LIMITED labels — M is a label-blind feature map fit on abundant generic data, acting as a strong prior for the small labeled probe. Planner: make M low-rank/answer-subspace, report effective rank, and compare the 4 predictors on held-out generalization (limited-label regime), not in-sample fit. The scientific read of #3: does the label-blind context→answer geometry already carry the behavior signal (behavior lives in the general answer-prediction, not a refusal-specific direction)?
2. **Judge baseline must be strong** (few-shot, calibrated) so a low judge AUROC means "surface-unpredictable," not "weak judge."
3. **Linear by default** for all probes and the map (project rule); nonlinear only if explicitly justified.

## Compute

Single ~1×H100 pod (`eval`/`lora-7b` intent). Generation for labels (vLLM, N-sample over ~1–3k prompts) + one forward pass for activations + CPU probe fits. Est. ~1–3 GPU-h.
