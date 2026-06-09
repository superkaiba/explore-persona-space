---
title: 'Phase C/D/E completion for #519: build input JSONs + run activation-shift
  extraction'
kind: experiment
tags: []
created_at: '2026-06-08T21:29:43Z'
has_clean_result: false
parent_id: 519
goal: 'Run Phases C/D/E/F on the six LoRA adapters trained in #519 (marker × 3 seeds
  + EM × 3 seeds, on HF Hub at commit c46b8989d) so the rank-one law cross-arm test
  the parent Goal asks for is actually computed: does a single shift-direction + cosine-scaled
  magnitude (rank-one) govern cross-context generalization for the marker arm and
  break for the EM arm?'
relates_to:
- leak-predictor
- leak-behavior-vs-marker
---
## Goal

Run Phases C/D/E/F on the six LoRA adapters trained in #519 (marker × 3 seeds + EM × 3 seeds, on HF Hub at commit c46b8989d) so the rank-one law cross-arm test the parent Goal asks for is actually computed: does a single shift-direction + cosine-scaled magnitude (rank-one) govern cross-context generalization for the marker arm and break for the EM arm?

## Why this exists

#519 trained all six cells successfully but the dispatcher **silently skipped** Phases C/D/E/F because the four input JSONs (`personas_json`, `questions_json`, `marker_pool_json`, `em_pool_json`) were not passed at launch. In sweep mode the dispatcher guards each phase with `if "c" not in skip_phase and args.personas_json and args.questions_json:` — when the JSONs are absent it falls through, writes `dispatch_manifest.json` with `skipped_phases: []` (a false record), and exits 0. Upload-verification PASSed on the *training* artifacts, the pod was terminated, and the per-step marker checkpoints on the pod were lost. The rank-one cross-arm test the parent Goal asks for is NOT ANSWERED on #519.

## Prerequisite (step 0 — do before anything else)

**The #519 code is NOT on `main`.** The #519 merge was an "unsafe-case" surgical checkout of only `eval_results/issue_519/` (the branch was 818 commits diverged; full rebase deferred). The dispatcher (`scripts/issue_519_dispatch.py`), the data builder, and the analysis modules (`src/explore_persona_space/analysis/activation_shift.py`, `steering_vectors.py`) live only on the `issue-519` branch (tip `9a0462ba9`) and its worktree.

First implementation step: in the issue-521 worktree, surgically bring the required #519 files onto the branch (`git checkout issue-519 -- <paths>`), resolve any imports that the 818-commit divergence left dangling, and smoke-test that `python -m explore_persona_space.analysis.activation_shift --help` and `python scripts/issue_519_dispatch.py --help` import cleanly. Files needed at minimum:
- `scripts/issue_519_dispatch.py`, `scripts/issue_519_build_data.py`, `scripts/issue_519_em_aligned_neg_regen.py`
- `src/explore_persona_space/analysis/activation_shift.py`, `src/explore_persona_space/analysis/steering_vectors.py`
- `configs/condition/c_issue_519_marker.yaml`, `configs/condition/c_issue_519_em.yaml`

## What to build

Four input JSONs the dispatcher's Phase C/D/E require (their absence is exactly what caused the silent skip):

1. **`personas_json`** — held-out persona panel. Source persona + the 4 trained-against negatives + held-out personas from `src/explore_persona_space/personas.py` and the #207/#383 panels. Schema per `activation_shift.py`'s loader.
2. **`questions_json`** — held-out questions (not the 200 used in #519 training). Per the plan's per-cell forward-pass budget.
3. **`marker_pool_json`** — steering-vector pool for the marker arm (Phase E).
4. **`em_pool_json`** — steering-vector pool for the EM arm (Phase E).

Confirm the exact schema of each against the loaders in `activation_shift.py` / `steering_vectors.py` before generating (the proposer's earlier schema guesses are not authoritative).

## Adapter staging

Phase C loads each adapter from the **local** path `output_dir/{arm}_seed{S}/adapter` — there is **no** `--adapter-source hf` flag. Before running, download the 6 adapters from HF (`superkaiba1/explore-persona-space`, commit `c46b8989d`, `issue_519/` subtree) into that local layout, OR add a small `--adapter-hf-repo/--adapter-hf-rev` staging path to the dispatcher.

## What to run

```bash
python scripts/issue_519_dispatch.py \
    --mode sweep \
    --skip-phase a1 a23 b0_smoke b \
    --layer <plan-layer> \
    --personas-json <p> --questions-json <q> \
    --base-cosines-json <bc> \
    --marker-pool-json <m> --em-pool-json <e> \
    --output-dir eval_results/issue_521
```

(Real skip tokens are `a1 a23 b0_smoke b c d e` — skip the training half, keep C/D/E. The dispatcher default `--layer` is 14; set it to the layer the #519/#474 line uses.)

Estimated cost: 3-4 GPU-h (Phase C is forward-pass-only; Phase E ~30 min). 1× H100 sufficient; 4× H100 parallelizes the 6 cells.

## Also fix the silent-skip in the dispatcher (in scope here)

While #521 touches the dispatcher, make the production-mode phase guards **fail loud**: in `--mode sweep`, if a non-skipped phase's required inputs are missing, raise (don't fall through), and record the *actual* phases run in `dispatch_manifest.json` (the guard-skip must be reflected, not reported as `skipped_phases: []`). This is the experiment-code half of the fix; the workflow-surface half is handled separately on the #519 line.

## Two design choices (resolved assumptions — override if wrong)

### Marker anchor — the per-step checkpoints are LOST
The #519 marker arm saturated (endpoint ΔlogP ≈ 30 nats at source). The non-saturated per-step checkpoints that plan §17 wanted were on pod-519, which was terminated, so they are gone. There is no cheap non-saturated option.
- **Assumption (staged-cheap):** run C/D/E on the saturated final marker adapters now. The rank-one test is about the *direction structure* of the per-context Δv vectors (not recipe-knob magnitudes), so saturation is less fatal here than in #448's recipe-sweep setting. Document the saturated-anchor regime as a scope caveat. Only if the marker SVD comes back uninformative do we re-train the marker at lower lr (+~4 GPU-h).

### EM manipulation check — was disabled mid-run
The #519 EM K=20 Sonnet-judge callback was disabled mid-run (Anthropic batch API congestion), so there is **no confirmation EM actually installed**. Running C/D/E blind on the EM arm risks measuring Δv of a non-misaligned model.
- **Hard gate:** before Phase C, run a small endpoint EM-rate eval (Sonnet judge with a non-batch / local fallback) on the 3 EM cells; post `epm:em-rate v1`. If EM < ~5% on the misaligned probes, the EM arm is invalid and must be re-trained with a working recipe (cf. #458) before any rank-one analysis — that becomes the pivot, not a blind run.

## Acceptance criteria

- `headline_metrics.json` with, per arm: per-context Δv (last-prompt-token residual at the plan layer), SVD spectrum (singular values + first-singular-vector cosines across contexts), magnitude-vs-base-cosine scatter, steering-vector identity test; plus cross-arm rank-one ratio (σ₁/Σσ) per arm and cos(U₁_marker, U₁_em).
- Two hero figures: magnitude-cosine scatter per arm, SVD spectrum per arm.
- `epm:em-rate v1` posted before Phase C.
- Clean-result answers the parent Goal ("rank-one holds for marker / breaks for EM?"). LOW confidence = inconclusive (needs a different anchor or recipe).

## Reproducibility anchors

- Parent issue: #519. `issue-519` branch tip `9a0462ba9`.
- Adapters: [`huggingface.co/superkaiba1/explore-persona-space/tree/c46b8989d/issue_519`](https://huggingface.co/superkaiba1/explore-persona-space/tree/c46b8989df021591c18711f51e50df4d6c9ab6c8).
- Training data: [`explore-persona-space-data/tree/c46b8989d/issue_519`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c46b8989df021591c18711f51e50df4d6c9ab6c8/issue_519).
- Plan v1 (source design): `tasks/awaiting_promotion/519/plans/plan.md`.
