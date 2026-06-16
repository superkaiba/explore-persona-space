---
title: 'Rank-1 MLP LoRA read/write geometry across dose: pre-existing vs intruder
  direction, marker vs sycophancy'
kind: experiment
tags: []
created_at: '2026-06-16T00:37:29Z'
has_clean_result: false
parent_id: 621
origin_prompt: run the rank 1 lora followup with marker and sycophancy
goal: Determine whether a rank-1 LoRA behavior implant's WRITE direction is a pre-existing
  base direction or a new (intruder) direction — in BOTH the weight-space sense (max
  cosine of the LoRA write singular vector to the base weight matrix's singular vectors)
  and the activation-space sense (cosine of the residual-space write to a pre-existing
  base residual-stream concept direction, and to the unembedding concept row) — and
  whether its READ direction rotates from random init toward the source-context direction
  as implant dose increases (a learned detector) or stays ~random (selectivity inherited
  from base geometry), across two behaviors of contrasting realism (programmatic marker
  vs on-policy sycophancy), using a rank-1 MLP up_proj+down_proj placement that puts
  the read in residual input space and the write in residual output space.
---
# Rank-1 MLP LoRA read/write geometry across dose: pre-existing vs intruder direction, marker vs sycophancy

## Goal

Determine whether a rank-1 LoRA behavior implant's WRITE direction is a pre-existing base direction or a new (intruder) direction — in BOTH the weight-space sense (max cosine of the LoRA write singular vector to the base weight matrix's singular vectors) and the activation-space sense (cosine of the residual-space write to a pre-existing base residual-stream concept direction, and to the unembedding concept row) — and whether its READ direction rotates from random init toward the source-context direction as implant dose increases (a learned detector) or stays ~random (selectivity inherited from base geometry), across two behaviors of contrasting realism (programmatic marker vs on-policy sycophancy), using a rank-1 MLP up_proj+down_proj placement that puts the read in residual input space and the write in residual output space.

## Hypotheses under test

For rank-1 LoRA behavior implants on Qwen-2.5-7B-Instruct, decide between two mechanistic pictures of what "training a behavior into a context" does, and whether the verdict transfers from a programmatic behavior (marker) to a realistic one (sycophancy):

- **H-ungated/intruder:** the WRITE is one direction that is *new* — near-orthogonal to base weight geometry (an "intruder dimension", Shuttleworth et al. 2410.21228) — the READ stays ≈ its random init even at higher dose, and per-context selectivity is inherited from pre-existing base geometry rather than a learned read.
- **H-pre-existing-detector:** the WRITE aligns with a pre-existing base *concept* direction the model already represents (the OOCR "find-and-amplify" hypothesis, Wang et al. 2507.08218), and the READ rotates toward the source-context direction as dose increases (a learned detector), i.e. selectivity is learned.

These are distinguished, per behavior, by the dependent variables below.

## Provenance

Origin prompt (2026-06-15): "run the rank 1 lora followup with marker and sycophancy".

Designed across a research session that (a) re-read the geometry thread #521 → #551 → #561 → #599 / #538 and the rank-1 line #604 → #621; (b) ran a deep-research literature synthesis stress-testing the "LoRA = ungated low-rank steering vector toward a pre-existing concept" composite claim (anchors: OOCR steering vectors 2507.08218; intruder dimensions 2410.21228; LoRA read/write asymmetry 2402.16842; rank-1 self-awareness 2511.04875; sleeper-agents conditional circuits 2401.05566; convergent EM 2506.11618); and (c) derived the MLP up_proj+down_proj rank-1 placement that puts the read in residual INPUT space and the write in residual OUTPUT space, giving a clean read↔context AND write↔behavior test from one adapter — the residual-space MLP read #621 never had (its read arm was attention q/v; its down_proj read lived in the 18944-d intermediate).

## Motivation (what the literature leaves open)

The deep-research verdict: C1 (one-direction write) and C5 (random read A suffices) HOLD robustly; C4 (learned ≠ naive vector) HOLDS; but **C3 (the write is a *pre-existing* base concept direction) is CONTESTED** — it is a "fuzzy hypothesis" even in the source paper, and the one paper that probes base geometry (2410.21228) finds LoRA writes *new* directions near-orthogonal to all base weight singular vectors, at exactly our low-rank regime (intruder dims present at r ≤ 16, vanish only toward full-FT). No surveyed paper tests Qwen-2.5-7B or persona/behavior implants, and none directly probes whether the write coincides with a pre-existing base *activation* direction.

Two senses of "pre-existing" must not be conflated (they answer different questions and can BOTH be true): weight-space (write ∈ column space of base W) vs activation-space (write aligned with a base residual-stream concept direction). #621 only compared the write to the *unembedding* W_U[marker] (output concept, cos 0.79); it never ran the weight-space intruder test, never probed an internal base concept direction, and found the read ≈ random init only at the LOW band-stop dose — explicitly flagging higher dose as its open next step.

## Formalization (object of study)

Identity grounding every DV: for a rank-1 module update ΔW = s·b·aᵀ, the output activation shift is Δy = s·(a·x)·b, so the activation-shift direction IS the write column b (left singular vector of ΔW); the read a (right singular vector) is an input-space gain, not an output direction.

**Dependent variables (per behavior, per dose, per seed):**
1. **Read rotation:** cos(a_up_trained, a_up_init) and cos(a_up∘γ, v_source) where v_source is the base mean post-attention-LN activation under the source persona (reuse the #604/#621 bank). Tests whether the read becomes a learned source-context detector vs stays random.
2. **Write→behavior (output concept):** cos(b_down, W_U[concept]) — marker: token ` ※` (id 83399); sycophancy: an agreement/sycophancy unembedding-direction proxy (planner to operationalize, e.g. judge-validated agreement-token direction or a contrast-of-means in the unembedding).
3. **Intruder test (weight-space pre-existence):** max over base singular vectors of cos(b_down, uᵢ_base[down_proj]) and cos(a_up, vⱼ_base[up_proj]). H-intruder predicts ≈ random floor; H-pre-existing predicts a clear peak.
4. **Concept test (activation-space pre-existence):** cos(b_down, d_behavior_base) where d_behavior_base is a pre-existing base residual-stream behavior direction (planner to operationalize: base contrast-of-means between behavior-exhibiting and neutral activations, NOT the learned vector).
5. **Selectivity vs base geometry:** the #621/#532 read — does the firing predictor a·v_c rank-order per-context leakage, and does it beat plain base geometry cos(v_c, v_source)? Re-run per behavior.

**Manipulated variables:** behavior ∈ {marker, sycophancy} (programmatic vs realistic — the transfer test); implant dose ∈ {low, high} (marker: log P(marker)−base band-stop [5,12] vs [12,20] nat per #538; sycophancy: an install-strength analog the planner defines). Placement fixed at rank-1 (up_proj, down_proj). Seeds {42, 137, 256}. Sources: a small persona panel (planner to set; reuse #621's where fit).

**What counts as an answer:** the DV-1 (read rotation with dose) and DV-3/DV-4 (intruder vs pre-existing) verdicts, reported per behavior with the marker-vs-sycophancy contrast. A clean outcome is e.g. "write is an intruder direction (DV-3 at floor) yet still points at the output concept (DV-2 high) and selectivity is base-geometry-inherited (DV-5), and the read does/does not rotate at high dose (DV-1) — and this holds/differs for sycophancy."

## Proposed design (planner to harden)

Reuse as much of the #621 / #604 rig as fits: the rank-1 outer-product save (a_init at step 0), the band-stop callback, the 14/19-persona × 20-question probe panel and the base-activation bank, the four-float marker storage contract, the extraction smoke gate, and the off-pod analysis. New vs #621: MLP up+down placement, the two new behaviors' recipes, the higher-dose arm, and the two pre-existence tests (intruder weight-SVD + internal base-concept).

Per-behavior training recipes follow the project rules (planner loads them): marker → marker-only loss, lr ≤ 5e-6, ` ※` id 83399 asserted, 1:1 contrastive negatives, band-stop (`.claude/rules/marker-training-recipe.md` + `marker-leakage-measurement.md`); sycophancy → on-policy-first positive completions via the elicitation ladder + judge filter, 1:1 contrastive negatives, dose-to-target (`.claude/rules/on-policy-completions.md` + `contrastive-negatives.md`).

## Known risks (flag for the planner)

- **Install uncertainty:** rank-1 on (up_proj, down_proj) only — no attention — is untested for either behavior; gate the sweep on a band-stop/install smoke before the full cells.
- **Sycophancy dose:** needs an install-strength analog of the marker band-stop (sycophancy has no single-token log-prob handle) — planner to define and pre-register.
- **Activation-space concept direction (DV-4):** operationalize d_behavior_base from the BASE model (contrast-of-means), never from the learned adapter, to keep the pre-existence test non-circular.
- **MLP nonlinearity:** the write↔behavior read is clean on down_proj (residual output); the read↔context read is clean on up_proj (residual input). Do NOT read up_proj's write (intermediate space) or down_proj's read (intermediate space) as residual-space directions.

## Relation to siblings (differentiate, don't duplicate)

- **#621** — rank-1 marker, low dose, attn-q/v + o/down placement; read ≈ a_init. This task adds the MLP up/down placement, the higher-dose arm, the weight-space intruder test, the internal base-concept test, and sycophancy.
- **#604** — write seed-stable (|cos| 0.93), points at W_U[marker]; key seed-arbitrary.
- **#647 / #649** — predictor-geometry siblings (do leakage-transfer predictors collapse to rank-1 / does the base-prior-vs-geometry split hold for sycophancy). Mechanistically distinct from this task's LoRA read/write/intruder factorization; planner should cross-reference, not merge.
