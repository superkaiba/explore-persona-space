---
title: 'Phase C/D/E completion for #519: build input JSONs + run activation-shift
  extraction'
kind: experiment
tags: []
created_at: '2026-06-08T21:29:43Z'
has_clean_result: false
parent_id: 519
---
---
title: 'Phase C/D/E completion for #519: build input JSONs + run activation-shift extraction'
kind: experiment
parent_id: 519
tags:
- persona-distance
- generalization
- followup
goal: Run Phase C (per-context Δv extraction), Phase D (SVD direction-constancy), Phase E (steering vectors), and Phase F (headline aggregation) on the six adapters from #519, after building the four input JSONs the dispatcher requires, so the rank-one law cross-arm test the parent issue Goal asks for is actually computed.
---

## Goal

Run Phases C/D/E/F on the six LoRA adapters trained in #519 (marker × 3 seeds + EM × 3 seeds, on HF Hub pinned to commit `c46b8989d`) so the rank-one law cross-arm test the parent issue Goal asks for is actually computed.

## Why this exists

#519 trained all six cells successfully but the dispatcher silently skipped Phases C/D/E/F because four pre-built input JSONs (`personas_json`, `questions_json`, `marker_pool_json`, `em_pool_json`) were not passed at launch. The dispatcher exited clean with `n_cells=6, skipped_phases=[]` while skipping 4 of 6 phases. The rank-one cross-arm test the parent Goal asks for is NOT ANSWERED on #519.

This follow-up is the smallest possible scope that answers the parent Goal: keep the trained adapters, build the missing inputs, re-invoke the dispatcher with `--skip-phase a,b`.

## What to build

Four input JSONs needed by the dispatcher's Phase C/D/E/F:

1. **`personas_json`** — 24-persona panel: `medical_doctor` source + the 4 trained-against negatives (`assistant`, `comedian`, `police_officer`, `software_engineer`) + 19 held-out personas pulled from `src/explore_persona_space/personas.py` and the #207 / #383 panels. Schema: `[{persona_name: str, system_prompt: str}, ...]`.

2. **`questions_json`** — held-out questions. Pull the 1000-question source pool used to build the #519 training data, subtract the 200 questions used in training, keep at least 480 for Phase C (per the plan's 480-forward-passes-per-cell estimate). Schema: `[{question: str, question_id: str}, ...]`.

3. **`marker_pool_json`** — steering-vector pool for the marker arm. Per plan §6, this is the pool of (persona, question) pairs the steering vector is extracted over. Use the source persona over the held-out question set.

4. **`em_pool_json`** — same shape, for the EM arm.

## What to run

```bash
python scripts/issue519_dispatch.py \
    --mode sweep \
    --skip-phase a,b \
    --personas-json <p> \
    --questions-json <q> \
    --marker-pool-json <m> \
    --em-pool-json <e> \
    --adapter-source hf \
    --hf-adapter-repo superkaiba1/explore-persona-space \
    --hf-adapter-tag c46b8989df021591c18711f51e50df4d6c9ab6c8
```

Estimated cost: 3-4 GPU-h on 1× H100 (Phase C is forward-pass-only; Phase E steering vectors are ~30 min).

## Two design choices that need a decision before launch

### Choice 1: which marker checkpoint to use as the anchor

The #519 marker arm saturated — endpoint `ΔlogP=30 nats` at source, which is the regime #448 documented as "all recipe knobs collapse to indistinguishable." The plan §17 intended a non-saturated anchor at 5-10 nats below ceiling.

Per-step adapters were preserved on pod-519 for this purpose, but pod-519 was TERMINATED after upload-verification PASS, so those per-step checkpoints are LOST. Two options:

- **Option A (cheap):** run Phases C/D/E on the final saturated marker adapters that are on HF. Accept that the marker-arm geometry is measured in the saturation regime and document it as a scope caveat.
- **Option B (correct):** re-train the marker arm at `lr=2e-7` (or `max_steps=200` at `lr=2e-6`) on a fresh pod, target a non-saturating endpoint. ~4 extra GPU-h.

Recommendation: B. The whole point of plan §17 was to avoid measuring the SVD spectrum at a saturated anchor; running C/D/E on the saturated adapters answers a different question.

### Choice 2: whether to also re-run EM with a working trajectory callback

The #519 EM arm has no trajectory data and no endpoint EM-rate measurement — the K=20 Sonnet-batch-judge callback was disabled mid-run after the Anthropic batch API congestion. **The manipulation check that EM was actually installed is missing.** Phases C/D/E would run blind on the EM arm — we'd get Δv vectors without knowing whether they correspond to an actually-misaligned model.

Recommendation: BEFORE running Phase C/D/E, run a small standalone endpoint EM-rate eval (Sonnet judge with a non-batch fallback OR a local-judge fallback) on the three EM cells to confirm EM > 5% on the misaligned probes. If EM rate is at floor on Qwen-2.5-7B-Instruct with this dataset + recipe, the EM arm needs to be re-trained with a different EM recipe (cf. #452 / #458) before any rank-one analysis can proceed.

## Acceptance criteria

- `headline_metrics.json` exists, containing:
  - Marker arm: per-context Δv (last prompt-token residual at layer 21), SVD spectrum (singular values + first singular vector cosines across contexts), cosine vs base-cosine scatter, steering-vector identity test.
  - EM arm: same four quantities.
  - Cross-arm comparison: rank-one ratio (σ_1 / Σσ) per arm, cos(U_1_marker, U_1_em).
- Two hero figures: (i) magnitude-cosine scatter per arm, (ii) SVD spectrum per arm.
- EM manipulation check posted as `epm:em-rate v1` event before Phase C runs.
- Clean-result body answers the parent Goal: "rank-one law holds for marker / breaks for EM?" with HIGH or MODERATE confidence (LOW means the analysis was inconclusive and we need a different anchor or recipe).

## Reproducibility anchors

- Parent issue: #519.
- Adapters: [`huggingface.co/superkaiba1/explore-persona-space/tree/c46b8989df021591c18711f51e50df4d6c9ab6c8/issue_519`](https://huggingface.co/superkaiba1/explore-persona-space/tree/c46b8989df021591c18711f51e50df4d6c9ab6c8/issue_519).
- Training data: [`huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c46b8989df021591c18711f51e50df4d6c9ab6c8/issue_519`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c46b8989df021591c18711f51e50df4d6c9ab6c8/issue_519).
- Plan v1 (the source design): [`tasks/awaiting_promotion/519/plans/plan.md`](https://github.com/superkaiba/explore-persona-space/blob/8ccad5d95f680ecf0138f3dbe8d1529052bb2a7b/tasks/awaiting_promotion/519/plans/plan.md).
