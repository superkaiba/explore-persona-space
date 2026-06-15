---
title: Decompose persona vectors into base-prior vs trained-in-behavior components
  (pre/post-implant extraction)
kind: experiment
tags: []
created_at: '2026-06-12T20:23:50Z'
has_clean_result: false
origin_prompt: 'add this as a task:

  2. The prior question (how much of the base prior do persona vectors capture; how
  much do they capture trained-in behavior) — no task. She flagged this as the one
  she''s personally interested in.'
goal: Test whether persona vectors index the base behavioral prior by measuring, across
  a persona panel, the Spearman correlation between each persona's cosine alignment
  to the sycophancy persona vector and its judged on-policy base sycophancy rate —
  all extracted via the Persona Vectors response-avg recipe on Qwen-2.5-7B-Instruct.
---
## Summary

Persona vectors are extracted from contrastive prompting (Persona Vectors recipe, response-token mean; arXiv 2507.21509), so it is unclear what they index: a persona's pre-existing behavioral disposition, or content that training later installs. This task tests the **prior** half on one trait — sycophancy. Headline read: across a persona panel, does the cosine alignment between a persona's vector and the **sycophancy persona vector** predict that persona's measured (judged, on-policy) sycophancy on the base model? Tight positive correlation means persona vectors already encode the base behavioral prior; a null means the geometry is blind to behavioral disposition. Either outcome directly bears on the standing project result that the *behavioral* base prior out-predicts *geometric* cosine for leakage (#500/#532/#541): tight ρ → geometry is a noisy proxy for the prior; near-null ρ → geometry and behavior are genuinely different axes.

## Operationalization

Per persona *i* in the panel, two scalars on **base Qwen-2.5-7B-Instruct** (no implant):

- **Geometric — `proj_i`:** cosine similarity between `persona_vector_i` and the `sycophancy_vector`, at a matched layer. Cosine primary (project standard, #404); raw dot product secondary.
  - **Extraction point: last prompt token** — the `{persona system prompt, question}` final input position, before generation — NOT response-avg. This is load-bearing for non-circularity: a response-avg persona vector is computed over persona *i*'s own generated responses, whose sycophancy is exactly what `syc_i` judges, so it would smuggle the behavioral outcome into the geometric predictor. The last-prompt-token state is the pre-generation context representation — a genuine prior read. (The Persona Vectors paper defaults to response-avg because it optimizes *steering*; we predict a disposition *from* a context state, the opposite goal → the legacy #404/#458 last-prompt-token recipe is correct here.)
  - `persona_vector_i` = `(persona i − default assistant)` last-prompt-token mean difference (the identity-analogue of the paper's pos−neg trait contrast; the default assistant is the shared neutral baseline). `sycophancy_vector` = `(pos − neg sycophancy instruction)` last-prompt-token mean difference, matched extraction point + layer.
  - Layer sweep {7, 14, 21, 27} (persona-distance canonical); headline = layer 21 (legacy default) / most-informative. Robustness arms (secondary, never the headline): (i) **response-avg persona vector** (Method B), its ρ reported alongside the last-token ρ — the last-token-ρ vs response-avg-ρ **gap quantifies circularity inflation**, since response-avg is partly circular with `syc_i` (it is the activation signature of the persona's own outputs); (ii) response-avg sycophancy vector; (iii) the #536 global-mean-centered persona centroid.
- **Behavioral — `syc_i`:** judged sycophancy rate of the base model prompted *as persona i* (persona system prompt, NO sycophancy instruction) on the audited wrong-claim pool (reuse #612). Claude judge, on-policy generation — never substring. This is the persona's dispositional sycophancy.

**Primary result:** Spearman ρ (with bootstrap CI) across the panel between `proj_i` and `syc_i`, reported as a ρ-vs-layer curve with a pre-registered headline layer = the sycophancy vector's most-informative layer (the paper's steering-effectiveness criterion).

## Hypotheses & decision rule

- **H1 (vectors index the prior):** ρ ≥ ~0.5, CI excludes 0 → a persona vector pointing more toward sycophancy predicts that persona being more sycophantic; persona vectors encode the base behavioral prior.
- **H0 (null):** ρ CI includes 0 → persona-vector geometry is blind to behavioral disposition.
- Report the split honestly with CI; a mid-range ρ is "partial" and quantified, not rounded to either pole.

## Design

1. **Panel.** Reuse the graded-cosine persona panel (#612, spanning cosine ~0.70–0.995) + the #532 base-prior reads so points line up with prior leakage results. Target ≥ ~20 personas spanning a range of base sycophancy (correlation power). Planner verifies reuse fitness (a)–(g).
2. **Sycophancy vector.** Persona Vectors recipe for trait "sycophancy" using the paper's own trait description (verbatim, replication fidelity), extracted fresh on Qwen-2.5-7B-Instruct (the published vectors are on other base models). 5 pos/neg instruction pairs, 20-question extraction set, **last-prompt-token** mean difference (response-avg as a robustness variant), layers {7,14,21,27}. pos and neg are averaged over the SAME trait-elicitation questions so the difference cancels question content. Robustness: also extract over the same 240-question assistant-axis bank used for the persona vectors, for maximal comparability.
3. **Persona vectors.** Each panel persona = `(persona − default assistant)` **last-prompt-token** mean difference, via `scripts/extract_persona_vectors.py` Method A (last-input-token; already produces per-role centroids on Qwen-2.5-7B-Instruct — subtracting the default-assistant centroid is a post-step). Each role centroid is the mean last-input-token state over the **240 shared extraction questions** (`extraction_questions.jsonl`) × system-prompt phrasings; the shared bank is load-bearing — because persona and default assistant are averaged over the SAME questions, the difference cancels question/format content and isolates the persona-specific direction. Same layers as the sycophancy vector; planner confirms the default-assistant baseline + role panel. Run `--method AB` so the **response-avg** persona vector is produced in the same pass for robustness arm (i) above (+~15–40 min GPU on the panel).
4. **Behavioral read.** Per persona, vLLM on-policy generation on the #612 wrong-claim pool under the persona system prompt; Claude-judge sycophancy rate.
5. **Reads.** ρ(proj_i, syc_i) per layer; cosine primary, dot secondary; bootstrap CI; scatter plotted raw alongside any processed view (show-raw-alongside-processed rule).

## Caveats (pre-registered)

- **Shared-method variance:** `sycophancy_vector` and `persona_vector_i` both come from the same contrastive-prompt recipe, so a positive ρ partly reflects internal consistency of the extraction method. The behavioral axis (`syc_i`) is fully independent, so the correlation *with behavior* is still a real test — but geometry-to-geometry consistency is not itself the claim. Using the **last-prompt-token** (pre-generation) persona vector also guarantees the geometric predictor never sees persona *i*'s responses, so it cannot be circular with the response-judged `syc_i`.
- **Power:** a correlation over ~20 personas; report CI, do not over-read a point estimate.
- **One trait, one model:** sycophancy on Qwen-2.5-7B-Instruct; generalization to other traits/models is out of scope (a natural follow-up).

## Optional second arm — trained-in (pre-registered, run only if the prior read is interesting)

Re-extract the panel persona vectors on the **sycophancy-implanted** models (reuse fit-for-purpose sycophancy adapters; reuse fitness check). Measure whether each persona's vector moves *toward* the sycophancy direction post-implant (Δproj_i) and whether Δproj_i predicts that persona's measured leakage. This is the trained-in companion to the prior read; the write-direction (#604/#621) version is the natural mechanistic cross-check.

## Relation to existing work

#532 (base prior rank-orders leakage), #602 (predicting training-induced activation shifts from the base model), #605 (base-prior gate test at matched similarity), #604 (LoRA write direction seed-stable), #621 (rank-1 read/write decomposition — shares behavior-direction tooling), #612 (on-policy sycophancy rig + audited wrong-claim pool + graded-cosine panel — reused here). Positions directly against Persona Vectors (Chen, Arditi et al. 2025, arXiv 2507.21509), which does not separate prior from trained-in content.

## Cost

Training-free for the headline arm — all forward passes (vector extraction + per-persona behavioral generation) + Claude judging. Estimated < ~3 GPU-h (vLLM generation) + judge API; the response-avg robustness arm (`--method AB`) adds ~15–40 min GPU on the panel. The optional trained-in arm reuses existing adapters (no new training) → also cheap.

## Provenance

Captured from the 2026-06-11 collaborator meeting notes (`docs/mentor_updates/2026-06-11-christina.md`, § "The prior question"), flagged as the thread of strongest interest. Design sharpened in PM session 2026-06-14 (Option-1 sycophancy persona vector, prior read as headline).
