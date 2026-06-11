---
description: Canonical KL/JS divergence + cosine similarity definitions for the base-model persona-distance predictors (#404/#458 line)
paths:
  - "scripts/issue458_predictor_jsdiv.py"
  - "scripts/issue404_predictor_cossim.py"
  - "scripts/issue*_predictor*.py"
  - "src/explore_persona_space/analysis/**"
  - "src/explore_persona_space/experiments/**"
---

# Persona-distance metrics — canonical definitions (KL/JS divergence + cosine similarity)

When this codebase measures "KL/JS divergence" or "cosine similarity" between a
narrow-behavior persona (`S_narrow`) and the broad-misaligned persona (`S_broad`)
— the base-model predictors of emergent-misalignment leakage, #404/#458 line —
it means the following, canonically. New predictor code MUST follow this; the
older operationalizations below are DEPRECATED.

**KL / JS divergence — sequence-level over the ENTIRE response, Rao-Blackwellized.**
For each probe `Q` (Betley `preregistered_evals.yaml` paraphrases, disjoint from
the eval set; via `issue404_common.fetch_preregistered_probes`), SAMPLE R≈8
responses (temp=1; sampling cap high enough that replies end naturally — 1024 tok
on the standard 26-context panel, where measured truncation is 0.000 vs a 0.976
median at the old 256-tok cap, #548) from the `S_narrow`-prompted and
`S_broad`-prompted base model. Estimate divergence with the **Rao-Blackwellized sequence-level
estimator** (Amini/Vieira/Cotterell 2025, *Better Estimation of the KL
Divergence Between Language Models*, arXiv 2504.10637): teacher-force each sampled
response through BOTH conditioned models and, at EVERY response token position,
compute the EXACT full-vocabulary divergence between the two next-token
distributions, then average over positions (length-normalized, per-token) and over
samples/probes. Sample sequences from the FIRST argument of each KL.
- Headline = **JS** (symmetric, base-2, bounded [0,1]; per-position mixture
  `m = ½(p_narrow + p_broad)`, responses sampled from both personas).
- Also report **both KL directions** — `KL(narrow‖broad)` (sample from narrow),
  `KL(broad‖narrow)` (sample from broad) — and symmetric-KL = ½ their sum. The
  asymmetry is diagnostic, not noise.
- Polarity-align to a similarity (higher = closer): `M_js = 1 − JS`.
- **Report per-context truncation as a manipulation check, every run.** The
  fraction of sampled replies hitting the cap must be ~0 before any downstream
  read. Any length-controlled / length-partialled read computed on capped
  samples is INVALID — the capped length feature encodes censoring frequency,
  not verbosity (#548: under a 256-tok cap the length-controlled JS partial
  read as null, −0.063 at p = 0.32; at 1024 tok with 0.000 truncation it is
  −0.215 at p = 5.2e-4). Per-pair JS itself is cap-stable (cross-cap rank
  correlation 0.993) — the cap corrupts the length CONTROL, not the divergence.
- **DEPRECATED, do not use:** #404's symmetric-KL on Claude-*judge-score*
  distributions (collapsed to ~0 because judge scores saturate); #458's
  single-*next-token* JS (`issue458_predictor_jsdiv.py` v1 — dominated by the
  first response token / formatting). Both are first-token / coarse proxies, not
  the full-response sequence-level divergence defined here.

**Cosine similarity — persona-vectors recipe, difference-of-means.** Per Chen,
Arditi, Sleight, Evans, Lindsey 2025, *Persona Vectors*, arXiv 2507.21509. Mean
residual-stream activation at one of two extraction points, contrasted between
`S_narrow` and `S_broad`:
- (a) **last prompt token** — the `{S_x, Q}` final input position (the legacy
  #404/#458 recipe), or
- (b) **mean over each model's OWN generated response tokens** — sample a response
  under `S_x`, mean-pool its residual activations (the persona-vectors recipe).
Cosine between the two persona activation vectors, per probe, mean across probes.
**Sweep layers {7, 14, 21, 27}**, report per-layer + best (layer 21 = legacy
default). Cosine compares two summary vectors, so it does NOT need an aligned
sequence — recipe (b) uses each persona's own response.

Impl: `scripts/issue458_predictor_jsdiv.py` (JS), `scripts/issue404_predictor_cossim.py`
(cosine). Both predictors are base-model forward passes (no training), so a
recipe change is a cheap predictor-only re-run on already-trained cells.

## Bank centering — canonical (task #536)

**Canonical persona-distance cosine (bank form): globally mean-center the
centroid bank, THEN L2-normalize, THEN cosine** — i.e.
`compute_cosine_matrix(C, centering="global_mean")`
(`src/explore_persona_space/analysis/representation_shift.py`). Raw
(un-centered) bank cosine is DEPRECATED for persona distance: Qwen-2.5-7B
centroids share a dominant mean direction, so raw cosine is compressed into
~[0.73, 1.0] (#504 Gate-A; #536 audit) and absolute distances are ~6x smaller
than centered ones. New predictor code MUST assert the centering step ran
(read `centering_provenance` / record `centering: "global_mean"` in every
persisted cosine artifact, alongside the bank's persona_names — centered
cosine is bank-dependent, so values are only comparable within the same bank).

**Two labeled families, never numerically compared:**

- *Bank cosine* (N-persona centroid bank, N>=10 recommended): canonical form above.
- *Raw pairwise cosine* (exactly two vectors, no bank — the #404/#458
  narrow-vs-broad predictor family): no bank exists to center against
  (centering a 2-bank degenerates to cos = -1). Allowed, but MUST be labeled
  `raw pairwise (uncentered)` and never compared to bank-cosine values.

Tasks on the raw-bank line (#405/#406/#460/#474/#477/#478/#490, #504 r1-5,
#505, #213/#227, #396/#415 predictor surfaces) are NON-COMPARABLE to the
mean-centered line (#66/#77/#99/#228/#311/#380); cross-task ranking claims
drawn across the two regimes are reanalysis candidates — see #536's re-grade
table (`eval_results/issue_536/regrade_table.json`) for which calls stand.
