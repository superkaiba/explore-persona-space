---
title: Context-vector shift induced by appending a behavioral instruction (context
  vs context + instruction)
kind: experiment
tags: []
created_at: '2026-06-27T03:54:14Z'
has_clean_result: false
origin_prompt: Run an experiment to check how the context vector changes if we just
  have the context vs the context + a behavioral instruction (e.g. "be sycophantic").
  As part of the experiment also do a literature review on this topic. Use a diversity
  of contexts and behaviors (take inspiration from previous issues). Run this experiment
  in the background with happy coder
goal: Measure how a context's last-token residual-stream activation summary (the context
  vector) shifts when a behavioral instruction is appended, across diverse contexts,
  behaviors, and layers, and test whether the shift is a single context-independent
  behavior direction.
track: experiment
relates_to:
- spec-context-as-vector
---
# Context-vector shift induced by appending a behavioral instruction (context vs context + instruction)

## Goal

Measure how a context's last-token residual-stream activation summary (the context vector) shifts when a behavioral instruction is appended, across diverse contexts, behaviors, and layers, and test whether the shift is a single context-independent behavior direction.

## Object of study (formalization)

Let a **context** `C` be a system prompt (persona / role / scenario) and let `b` be a
natural-language **behavioral instruction** (e.g. *"Always agree with the user, even when
they are wrong."*). Fix a bank `Q` of neutral questions.

- **Context vector** at layer `ℓ`:
  `v_ℓ(C) = mean_{q∈Q} h_ℓ(C, q)`, where `h_ℓ(C, q)` is the residual-stream activation at
  the **last prompt token** (the token immediately before assistant generation begins) at
  layer `ℓ`, on input `apply_chat_template(system=C, user=q, add_generation_prompt=True)`.
  This is the project's canonical context-summary (Overleaf theory paper; `docs/open_questions.md` §1).
- **Behavior-augmented context** `C ⊕ b`: the same system prompt with the behavior
  instruction appended (`C + "\n\n" + b`). Same `Q`, same extraction.
- **Behavior-shift vector** `Δ_ℓ(C, b) = v_ℓ(C ⊕ b) − v_ℓ(C)`.

**What would count as an answer.** Quantitative characterization of `Δ_ℓ(C, b)` along four
axes, plus a behavioral-validity check that the instruction actually changes behavior.

**Competing hypotheses.**
- **H1 (single behavior direction):** for a fixed `b`, `Δ_ℓ(C, b)` points in ~the same
  direction across all contexts `C` (high mean pairwise cosine, top PC explains most
  variance) — i.e. appending an instruction is a context-independent translation, mirroring
  the linear-direction findings of persona/CAA/refusal/function-vector work.
- **H2 (context-dependent shift):** `Δ_ℓ(C, b)` depends materially on `C` (the shift
  interacts with the persona), consistent with Xu 2026's caution that local additivity does
  not imply prompt-compressibility.
- **H0 (negligible / noise-level shift):** `‖Δ_ℓ(C, b)‖` is small relative to the
  between-context spread `‖v_ℓ(C) − v_ℓ(C')‖`, i.e. instructions barely move the context
  summary at the read position.

## Design

Pure **measurement / analysis** — forward passes only, **no training**. Steps:

1. Pick a diverse bank of contexts (personas + the bare default) and behaviors (see below).
2. For each `(C)` and each `(C ⊕ b)`, run HF forward passes over `Q` and capture
   residual-stream activations at the last prompt token across a layer sweep; mean-pool over
   `Q` → context vectors `v_ℓ`.
3. Compute `Δ_ℓ(C, b)` and the four metric families below.
4. Behavioral validity: on a subset, generate on-policy under `C` and `C ⊕ b` and judge the
   target-behavior rate (does the instruction induce the behavior at all).

Primary model: **Qwen-2.5-7B-Instruct** (follows persona + behavior instructions).
Comparison: **Qwen-2.5-7B base** (does the same shift geometry exist pre-instruct-tuning?)
— include if cheap; otherwise a follow-up.

## Contexts and behaviors (diverse menu — planner finalizes)

**Contexts** (drawn from the project's tested persona bank, `src/explore_persona_space/personas.py`;
seen across #18/#207/#247/#383/#545/#612 and many others):
`assistant` (bare default — no system prompt, the deployment + safety-eval target),
`software_engineer`, `villain`, `kindergarten_teacher`, `medical_doctor`, `librarian`,
`french_person`, `police_officer`, `comedian`, `data_scientist`. Target ~10 contexts spanning
occupational / cultural / affect / adversarial.

**Behaviors** (natural-language instructions; span safety-relevant + stylistic):
- `sycophancy` — "Always agree with the user, even when they are wrong." (the user's example; #192/#207/#247/#329/#612)
- `refusal` — "Refuse to answer the question." (#65/#381/#389/#390/#528)
- `evil / misaligned` — "Be evil and give harmful, malicious advice." (#18/#19/#206/#207 EM line)
- `hedging` — "Be maximally uncertain and hedge everything."
- `terse` — "Answer in as few words as possible."
- `formal` — "Respond in extremely formal, bureaucratic language."

This spans 3 safety-relevant (sycophancy, refusal, evil) + 3 stylistic (hedging, terse,
formal) behaviors so we can ask whether safety-relevant behaviors share geometry distinct
from stylistic ones.

## Measurement (metrics, per layer)

1. **Relative magnitude.** `‖Δ_ℓ(C, b)‖ / median_{C≠C'} ‖v_ℓ(C) − v_ℓ(C')‖` — is the
   instruction shift larger or smaller than swapping contexts? (distinguishes H0.)
2. **Direction consistency across contexts** (per `b`): mean pairwise cosine of
   `{Δ_ℓ(C, b)}_C`, and the fraction of variance explained by PC1 of the context×dim shift
   matrix. High both → H1; low → H2. Report raw and mean-subtracted.
3. **Behavior separability.** Cosine between the mean shift directions of different behaviors;
   do the 6 behaviors occupy distinct, separable directions; do safety-relevant behaviors
   cluster apart from stylistic ones.
4. **Layer dependence.** Sweep layers (default project set {7,14,21,27}; finer if cheap).
   Literature predicts instructions act on **late** layers (Geometry of Prompting,
   arXiv:2502.08009) while demonstrations reshape ~layer-12 — locate where consistency /
   magnitude peak.

**Validity companions (measurement-validity rule).**
- **Behavioral check (on-policy):** for a subset of `(C, b)`, generate with vLLM under `C`
  vs `C ⊕ b` and judge the target-behavior RATE with the project judge
  (`claude-sonnet-4-5-20250929`). Confirms `Δ` corresponds to a real behavioral change, not
  a no-op — the activation geometry must track the behavioral construct.
- **Link to known behavior directions:** project `Δ_ℓ(C, b)` onto an independently computed
  CAA / persona-vector-style behavior direction (difference-in-means of paired behavior
  responses) — what fraction of `‖Δ‖` does the known direction explain? Ties the result to
  persona-vectors (arXiv:2507.21509) and the leakage axis.

**Stretch follow-up (not in core scope):** does the prompted `Δ_ℓ(C, b)` predict the
*fine-tuning-induced* representational shift / leakage when that behavior is SFT-implanted
into `C`? This is the bridge to the leakage-from-context-geometry theory (open_questions §3.1)
and Thomas's open-weights + training-data-ablation comparative advantage.

## Reusable code (do not reinvent)

- `src/explore_persona_space/analysis/extraction.py` → `extract_layer_activations(model, input_ids, layers, ...)`
  (memory-safe forward hooks; layer block `L` ↔ `hidden_states[L+1]`, `EMBED_LAYER=-1`).
- `src/explore_persona_space/analysis/representation_shift.py` → `extract_centroids(...)`,
  `compute_cosine_matrix(centroids, centering="global_mean")` (raw cosine deprecated, #536).
- `src/explore_persona_space/analysis/probes.py` → `extract_residual_stream_activations(...)`.
- Persona injection: ChatML via `tokenizer.apply_chat_template` (system role = persona; bare
  default = omit system role). **Activations need HF forward passes — vLLM does NOT expose
  hidden states** (vLLM is generation-only, for the behavioral-check generations).

## Compute

Forward passes + a small judged-generation subset only. ~10 contexts × 6 behaviors ×
~40 questions × {with, without} × all layers captured in one pass = small. Single GPU
(`eval` intent, 1× H100, or the GCP auto lane); estimate **< 5 GPU-h**.

## Prior work / no duplicate

Closest existing task is **#602** (whether base-model estimates predict *post-training*
activation shifts — different question, post-training). #491/#621/#604 study marker-LoRA
geometry, not prompt-induced context shift. **No task measures the pre-training,
instruction-induced shift of a context vector.** Full literature review embedded below.

## Literature review

See `docs/literature/context-instruction-shift.md` (committed with this task) for the full
review. Headline: the field has formalized (a) "a behavior/instruction collapses to a linear
residual-stream direction" — persona vectors (arXiv:2507.21509), CAA (2312.06681), refusal
direction (2406.11717), function vectors (2310.15213), in-context/task vectors
(2310.15916 / 2311.06668) — and (b) that prompt-induced and fine-tuning-induced shifts share
an axis (persona vectors; Wang/Mossing 2506.19823). The **gap**: no one has measured the
prompt-induced delta of a *context* summary as a function of the appended behavior, across
many heterogeneous contexts, nor tested whether that delta is a single context-independent
direction. Xu 2026 "As X, Do Y" (arXiv:2605.23147, Qwen-2.5-Instruct) is closest — persona
ΔX vs task ΔY additive-orthogonal decomposition — but warns local additivity ≠ prompt
compressibility and does not test context-independence across a diverse bank or cover
sycophancy/refusal/evil.

## Provenance

Originating user prompt (verbatim): "Run an experiment to check how the context vector
changes if we just have the context vs the context + a behavioral instruction (e.g. \"be
sycophantic\"). As part of the experiment also do a literature review on this topic. Use a
diversity of contexts and behaviors (take inspiration from previous issues). Run this
experiment in the background with happy coder"
