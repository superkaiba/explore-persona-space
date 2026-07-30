---
title: 'Multi-turn prefix-arm map at 100k: build the multi-turn n1M analogue, fit
  prefix+context arms, characterize both arms'' prediction error'
kind: experiment
tags: []
created_at: '2026-07-28T00:42:49Z'
has_clean_result: false
parent_id: 1482
origin_prompt: 'run for 100k as much in parallel and vectorized as possible. [context:
  user approved option R3(b) from chat — ''new capture: build the multi-turn analogue
  of the n1M corpus — N multi-turn LMSYS/WildChat conversations (prefix = history,
  query = last turn), capture prefix-end + context-end + answer states, fit both arms,
  run the full pipeline'' — after a sizing discussion grounding N=100k at ~50–70 GPU-h
  capture on the measured 3,250 ctx/GPU-h parent basis]'
workflow: v1
goal: 'Build the multi-turn analogue of the #779 fitter-fair-comparison-n1m corpus
  at N≈100,000 real multi-turn LMSYS/WildChat conversations (prefix = full conversation
  history before the last user turn, ≥2 user turns; query = last user turn), generate
  one on-policy answer per context under the parent decoding recipe, capture layer-{14,19,26}
  prefix-end + context-end + mean-answer states under the parent capture convention,
  fit the five parent fitters in BOTH arms (prefix-based AND context-based) on a pinned
  near-dupe-gated split, and run the #1482 error-characterization pipeline on both
  arms: (1) prefix-arm transport at scale (held-out R² per arm/layer vs the context
  arm and vs #1092''s 0.05–0.11); (2) per-context error + judged taxonomy incl. conversation-depth,
  floor-relative via a K-resample answer-sampling floor; (3) per-direction answer-PCA
  linear-vs-nonlinear decomposition with shrinkage control, cross-arm; (4) identity+learned-bias
  baseline and kNN-retrieval reads per arm. Binding user directive: maximize parallelism
  and vectorization at every phase (wide sharded capture fleet, batched fits, no serial
  inner loops). Phase 0 CPU manifest probe verifies multi-turn supply before any GPU
  provisions; a 1-shard pilot re-measures multi-turn throughput before fleet sizing
  (plan basis ~1,500–2,000 ctx/GPU-h vs parent''s measured single-turn 3,250).'
relates_to:
- spec-context-as-vector
---
# Multi-turn prefix-arm map at 100k: build the multi-turn analogue of the n1M corpus, fit prefix+context arms, characterize both arms' prediction error

## Goal

Build the multi-turn analogue of the #779 fitter-fair-comparison-n1m corpus at N≈100,000 real multi-turn LMSYS/WildChat conversations (prefix = full conversation history before the last user turn, ≥2 user turns; query = last user turn), generate one on-policy answer per context under the parent decoding recipe, capture layer-{14,19,26} prefix-end + context-end + mean-answer states under the parent capture convention, fit the five parent fitters in BOTH arms (prefix-based AND context-based) on a pinned near-dupe-gated split, and run the #1482 error-characterization pipeline on both arms: (1) prefix-arm transport at scale (held-out R² per arm/layer vs the context arm and vs #1092's 0.05–0.11); (2) per-context error + judged taxonomy incl. conversation-depth, floor-relative via a K-resample answer-sampling floor; (3) per-direction answer-PCA linear-vs-nonlinear decomposition with shrinkage control, cross-arm; (4) identity+learned-bias baseline and kNN-retrieval reads per arm. Binding user directive: maximize parallelism and vectorization at every phase (wide sharded capture fleet, batched fits, no serial inner loops). Phase 0 CPU manifest probe verifies multi-turn supply before any GPU provisions; a 1-shard pilot re-measures multi-turn throughput before fleet sizing (plan basis ~1,500–2,000 ctx/GPU-h vs parent's measured single-turn 3,250).

1. **Prefix-arm transport at scale (primary):** held-out whole-map R² per arm per layer, vs the context arm and vs #1092's prefix reads (0.05–0.11 at 21k rows). This is the first non-degenerate prefix arm at n1M-style scale (the existing n1M corpus is single-turn, so its prefix is a constant chat-template string — #1482 verified min cosine 1.000 — and its prefix arm is structurally a null).
2. **Which contexts each arm is bad at predicting:** per-context normalized error `nerr(x)`, judged taxonomy (language, topic, refusal-adjacency, answer-is-refusal, format, PLUS conversation-depth — the axis #1482 had to drop), BH-FDR contrasts, floor-relative via a K-resample answer-sampling floor on a stratified subsample (the #1482 stage-9 recipe).
3. **Which parts of the answer state each arm is bad at predicting:** per-direction answer-PCA decomposition, linear-vs-nonlinear gap per direction (with the #1482 per-direction-λ shrinkage control), cross-arm comparison of the poorly-predicted subspaces.
4. **Standing mapping reads:** identity+learned-bias baseline and kNN-retrieval (acc@k, chance stated) per arm per fold.

**Binding user directive:** maximize parallelism and vectorization at every phase — wide sharded capture fleet (the parent's 32-shard × multi-pod pattern), all fits batched/vectorized (the parent's measured fit walls at n=963k: ridge 57 s, MLP-8192 8 min, MLP-32768 18 min, KRR 75 s per layer — no serial inner loops anywhere), CPU phases fanned out.

## Design sketch (for the planner — refine, don't inherit blindly)

- **Phase 0 (CPU, ~3–4 h, no GPU held):** multi-turn manifest build — stream LMSYS-Chat-1M + WildChat-1M, keep conversations with ≥2 user turns, apply the parent's language/moderation/dedup filters (the #1092/#1113 tiny-real-probe discipline: verify filters against REAL rows before the production sweep), near-dupe gate vs a fresh pinned val/test carve. **Multi-turn supply is unverified** — the parent exhausted LMSYS at 54.7% keep for FIRST-TURN prompts; this phase reports the realized multi-turn pool size before any GPU provisions. If supply < ~120k after filters, take what exists and report the shortfall.
- **Phase 1 (GPU pilot, ~1 shard, ≤1 GPU-h):** re-measure generation+capture throughput at multi-turn lengths through the production entrypoint before sizing the fleet. Sizing basis: parent measured 3,250 ctx/GPU-h (single-turn, H100); multi-turn prefill is longer, plan at ~1,500–2,000 ctx/GPU-h until the pilot says otherwise. Fence ≥2× the pilot-extrapolated wall.
- **Phase 2 (GPU fleet):** sharded generation+capture, ~50–70 GPU-h at N=100k on the assumed throughput. Wide fleet per the user directive (e.g. 8×H100-class × 1–4 pods / GCP wide rungs / fellows lane per the router). All the parent's ops fixes are already in the `issue-779-n1m` branch drivers and MUST be inherited: K=10 batched upload commits (the 429-storm fix), OMP/MKL=8 thread caps, VLLM_WORKER_MULTIPROC_METHOD=spawn, eager+no-prefix-cache, per-chunk upload+sha-verify+purge, raw text persisted unconditionally before any reduce.
- **Phase 3 (fits, ~2–3 GPU-h total):** both arms × 3 layers × 5 fitters on the pinned split, checkpointed streaming (the parent's round-4 stream-checkpoint fix).
- **Phase 4 (characterization, ~0 GPU + Batch-API judge):** fresh stratified holdout, holdout-excluded refits, per-context + per-direction reads on both arms, judge labels via the Anthropic Batch API (one call per holdout context, `claude-sonnet-4-5-20250929`), K-resample floor subsample (~3 GPU-h if run, per the #1482 measured precedent). The SAE-feature arm is OPTIONAL (+15–20 GPU-h) — planner decides; default OUT to keep the run under the auto-approve cap.

## Sizing (grounded)

Total estimate ~55–80 GPU-h (capture-dominated; +~3 GPU-h K-resample floor; SAE arm excluded by default). Parent precedents: n1M capture ~295 GPU-h realized / 963k single-turn contexts / ~9.2 h wall on 4× 8×H100; #1482 error-analysis ~30–34 GPU-h realized (SAE forwards dominated); #1482 k-resample ~3.1 GPU-h. Storage: capture grows ~50% per context vs parent (one extra stored position) — ~15–20 GB at 100k; sharded-upload machinery handles it.

## Reuse

- `issue-779-n1m` branch drivers (manifest builder, sharded capture launcher, `issue779_ffc_n1m_fits.py`) — extend with the multi-turn filter, the prefix-end capture position, and the prefix fit arm. Artifact-reuse fitness checks apply, including parent-lineage coherence (diff main vs the unmerged `issue-779-n1m` branch before reusing any module).
- `scripts/issue1482_error_analysis.py` / `issue1482_kresample.py` / `issue1482_analysis.py` — the characterization pipeline; extend per-arm.
- Parent capture convention + token-id-concatenation seam rule (#1092/#1482 identity gates) — the #1482 k-resample round's convention mismatch (93.1% of rows) is the cautionary precedent: gate any recapture against streamed parent rows where applicable.

## Provenance

Filed from user chat 2026-07-27 (session: plan discussion "where the context→answer maps fail"). This is option R3(b) from that discussion: the multi-turn analogue of the n1M corpus, the only way to get a non-degenerate 1M-style prefix map. Estimates and phase structure from the same discussion, grounded in the #779 n1M dispatch marker (measured 3,250 ctx/GPU-h) and the #1482 Repro footer.
