---
title: 'Cross-behavior cross-context shared-direction geometry on the #537 testbed
  (post-hoc re-extraction)'
kind: experiment
tags: []
created_at: '2026-06-16T06:55:43Z'
has_clean_result: false
parent_id: 537
origin_prompt: 'test whether many conditional behaviors share a direction across different
  contexts; post-hoc on the #537 context-leakage comprehensive eval'
goal: 'Re-extract layer-L activation-shift directions (trained minus base) from the
  #537 context-generalization testbed''s existing adapters (5 conditional behaviors
  x 16 training contexts x seeds, on HF) plus a bounded 2nd-seed (1042) retrain of
  em + sycophancy across the 16 contexts so all readable behaviors get a within-cell
  seed ceiling, and test (Q1) whether each behavior''s shift collapses to ONE direction
  across the training contexts it was implanted under (context-invariance of the write)
  and (Q2) whether the different behaviors'' dominant directions coincide or cluster
  by family (cross-behavior identity), benchmarked against the within-cell seed ceiling,
  with unit-norm direction cosine as the dose-invariant DV.'
relates_to:
- identity-cb-duality
- identity-persona-vs-behavior
---
# A LoRA-installed conditional behavior writes one context-invariant direction; across four behaviors the directions are mostly distinct, with one predicted family coincidence and one unexplained high off-diagonal (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- **All four readable behaviors pass the context-invariance bar** — per-context cosine clears the **0.85-of-seed-ceiling** bar in ≥80% of contexts (16/16 em/agreement/fact, **13/16 marker**). Context tunes strength, not direction.
- **The four behaviors are mostly distinct, not one generic "implant" axis** — 4 of 6 pairs sit **below** the global cross-behavior null (**0.05-0.17** of ceiling; two within their pairwise null, two just above), none near 0.85.
- **One predicted family coincidence (harmful-advice x agreement, 0.59) and one unexplained high off-diagonal (taught-fact x marker-tic, 0.61)** — for both, a shared LoRA-geometry component is not ruled out.
- **The negative control did not read as a clean null** — refusal's 4-of-16 single-seed contexts cleared their null (0.48 vs p95 0.27, mean cos 0.84), against prediction; the binding constraint.
- **Scope:** refusal recovered at only 4 of 16 contexts (single seed) after 5 HF Hub timeouts; the 4 EM-no-contrast cells dropped. 2-seed ceiling, one fixed panel.

## What I ran

- **Why:** A prior single-behavior result ([#521](https://eps.superkaiba.com/tasks/521)) found EM's activation shift collapses to one direction across seeds, and a cross-arm probe ([#552](https://eps.superkaiba.com/tasks/552)) found marker and EM point different ways. The open question: does "one direction" survive across the *training context* a behavior is installed under, and do *different* behaviors share an axis? The parent testbed ([#537](https://eps.superkaiba.com/tasks/537)) already trained 5 behaviors x 16 contexts but never read the geometry — so this is a re-extraction over adapters that already exist.
- **Design:** 4 readable behaviors (harmful-advice/EM, wrong-claim agreement, marker tic, taught fact) x 16 training contexts x 2 seeds = 128 adapter cells, read on ONE fixed neutral probe panel (14 personas x 20 questions). The single manipulated variable within a behavior is the training context. A blanket-refusal null-control rides along as the in-run negative arm. DV = unit-norm residual-shift direction cosine (dose-invariant), layer 14.
- **Training:** The only new training is a 2nd-seed (1042) replicate of #537's em + sycophancy across all 16 contexts (32 adapters), on #537's exact frozen data + recipe + dose, so all four readable behaviors get a within-cell seed ceiling. Everything else is re-extraction from existing #537 adapters.
- **Eval:** Per cell, one teacher-forced forward per (persona, question) on the fixed panel; shift = trained-minus-base residual at layer 14 (mean-over-response for the generative behaviors, end-of-response slot for marker/fact). Per behavior: SVD top-direction share + per-context cosine to the top direction, vs a sign-flip null (1000 reps) and the seed ceiling. Cross-behavior: 4x4 unit-norm cosine matrix, each off-diagonal as a fraction of the seed ceiling.

## Findings

### Each behavior passes the context-invariance bar across its 16 training contexts (Q1)

Registered bar: per-context cosine reaches 0.85 of the behavior's median seed ceiling for ≥80% of contexts, AND the top-share clears the sign-flip null.

![Left: per-behavior top-direction share with sign-flip null p95. Right: per-context cosine to each shared direction, with per-cell seed-ceiling and 0.85 bar; marker's 3 sub-bar contexts and the refusal null arm (5th row) visible.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/q1_context_invariance.png)

> **Figure.** *Every readable behavior's shift concentrates on one direction shared across its 16 install contexts; the refusal null arm rides along.* Left: top-direction share vs sign-flip null p95. Right: per-context cosine to the shared direction; black = per-cell median seed ceiling, dashed red = 0.85 bar. Refusal (red) = 4 contexts, 1 seed, no ceiling.

All four return the `context_invariant` verdict.

- Em, agreement, fact clear the bar 16/16; marker 13/16 (81%).
- Marker's 3 misses: `binst_marker` (cos 0.16, near-orthogonal) plus two just under its 0.843 bar (`fmt_json` 0.836, `icl_k8` 0.838).
- Fact passes 16/16 but its bar is lowest (**0.725**, its per-cell seed ceiling being lowest at 0.853), so two passing contexts (`icl_k8` 0.768, `icl_k2` 0.800) sit well below its cluster. Fact is the weakest readable behavior on every internal metric.

### Across the four behaviors the directions are mostly distinct (Q2)

Each pair's direction cosine is reported as a fraction of the seed ceiling (so dose cannot masquerade as geometry), against the cross-behavior null.

![4x4 heatmap of cross-behavior direction cosine as a fraction of the seed ceiling; diagonal 1.00. Harmful-advice x agreement 0.59 and taught-fact x marker 0.61 bold; the other four 0.05-0.17.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/q2_cross_behavior_heatmap.png)

> **Figure.** *Behaviors are mostly distinct; two off-diagonals are high.* Off-diagonal = direction cosine as a fraction of the geometric-mean seed ceiling; global cross-behavior null p95 = 0.28 (applied uniformly). Bold cells are the two high pairs; the other four sit below the global null (two within their pairwise null, two just above).

- Four of six pairs are distinct (0.05-0.17 of ceiling, two within null); no pair reaches the 0.85 shared-axis bar.
- Two stand out at ~0.6: harmful-advice x agreement (0.59) and taught-fact x marker-tic (0.61). Only the first is the *registered* near-boundary pair (predicted advice/agreement family); fact x marker fits no family story — more likely shared corpus/format shape than behavior semantics.
- For BOTH ~0.6 pairs, a shared LoRA/SFT geometry component is not ruled out: the construct bridge validates each behavior's *own* direction, not the cross-pair overlap's origin, and the refusal result strengthens that alternative.
- The 0.28 null is *global*; the em x agreement *pairwise* null p95 is **0.20** (vs others' ~0.06-0.10), so its margin above its own null is narrower than implied.

### The two high off-diagonals fall far short of same-behavior agreement

The anchor for every cross-behavior cosine is the within-behavior seed ceiling — how close two reruns of the *same* behavior land.

![Horizontal bars: four same-behavior per-behavior-U1 seed ceilings (marker 1.00, agreement 0.99, harmful-advice 0.96, taught-fact 0.96) above the two cross-behavior coincidences (0.58, 0.59); null p95 dashed at 0.28; footnote disambiguates the two ceiling objects.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/seed_ceiling.png)

> **Figure.** *Same-behavior reruns agree at 0.96-1.00; the two high cross pairs sit at 0.58-0.59 raw cosine.* Gold = per-behavior-U1 cross-seed ceiling (the Q2 normalization object), dark red = cross-behavior coincidence, dashed = null p95 (0.28). Footnote: NOT the per-cell ceiling used for the Q1 bar (fact's per-cell median is lower, 0.85).

Two distinct objects carry the word "ceiling" and must not be conflated:

- **Q1 per-cell** — per-(behavior,context) cross-seed cosine, median per behavior (fact 0.85, em 0.95, marker 0.99, agreement 0.99); sets the per-context bar.
- **Q2 per-behavior-U1** — cross-seed cosine of each behavior's top direction (fact 0.96, em 0.96, marker 1.00, agreement 0.99); normalizes the cross-behavior matrix.

This figure plots the Q2 object; fact's two objects diverge most (0.96 vs 0.85). Against either, the two high pairs clear the null decisively but reach only ~60% of same-behavior agreement. With a 2-seed ceiling (one cosine per cell, no interval), a ratio this close to the 0.5-vs-0.85 boundary is noise-limited — suggestive, not a clean shared-axis verdict.

### The refusal negative control did not read as a clean null in the partial recovery

The in-run negative control was the blanket-refusal row: a behavior #537 verified failed to install ("texture, not data") should read as *no coherent direction*. The result was the opposite — the binding constraint that holds confidence at MODERATE.

- After 5 HF Hub timeouts, refusal survived at **4 of 16 contexts**, single seed (no ceiling). On those 4, top-share **0.484** clears its null (p95 0.266) and mean cosine to its own direction is **0.843** — comparable to fact's top-share (0.449), inside the readable range (the 5th strip row in the Q1 figure).
- The pipeline DID extract a low-rank, coherent-looking direction from the "unreadable" behavior. But the 4 survivors are all `sp_*` system-prompt-persona variants, so the shared direction may be context-similarity artifact, not a real refusal write.
- With no ceiling and 4 near-identical contexts, this neither confirms nor refutes the control — an inconclusive partial null arm. It also caveats "writes one direction": the layer-14 read picks up *something* coherent even from a weak implant.

### Behavior identity, not training context, dominates the shift (descriptive)

Decomposing the full behavior-x-context shift tensor into shared "any-implant", behavior-specific, and context-specific components gives a tensor-level view of what drives the geometry.

![Stacked horizontal bar: behavior-specific 63.8%, shared any-implant 22.3%, context-specific 13.8% of total shift energy across 68 cells.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651/variance_decomposition.png)

> **Figure.** *Behavior identity dominates; the training context contributes least.* Un-normed (dose-sensitive) Frobenius-energy decomposition of the full behavior x context shift tensor, n=68 cells, unbalanced — all five behaviors incl. the 4 partial refusal cells (16+16+16+16+4). One descriptive view, not an independent corroboration.

A shared "any-implant" component is real (22.3%) but a minority; behavior identity carries most energy (63.8%) and the training context the least (13.8%). This is **descriptive support** for Q1 and Q2, NOT an independent corroboration: the tensor is **un-normed** (dose mass contributes; the split is dose-sensitive), **unbalanced** (n=68, only 4 refusal cells against 16 each), and folds the partial, contrary-reading refusal row into the 63.8% behavior-specific term.

### The fact and agreement directions are behavior-specific, not generic LoRA artifacts

A neutral-panel top direction could be a generic adapter/SFT direction rather than the behavior's actual write — and the seed ceiling, nulls, and refusal row all pass for a generic direction too. Em and marker were already behavior-validated by prior work; fact and agreement were not, so each got a construct-validity check: re-extract the shift on the behavior's own canonical elicitation surface and cosine it against the neutral-panel direction (bar 0.50).

Both pass clearly: **taught fact 0.94**, **wrong-claim agreement 0.84** (16 canonical cells each, seed 42). So the neutral-panel direction these two behaviors point along genuinely IS the behavior-specific write. This validates each behavior's self-direction; it does NOT bridge the origin of the two ~0.6 cross-pair overlaps (see Q2).

### Pipeline soundness: the canary reproduces a committed reference exactly

Before trusting any cross-run cosine, the run reproduced a committed reference direction ([#521](https://eps.superkaiba.com/tasks/521)'s villain-source marker adapter): top-share 0.325, mean-cosine 0.587, and the reproduced top direction identical to the reference (cosine 1.00). A second gate confirmed both adapter-loader layouts apply a nonzero shift (root-layout norm 6.4; nested em-layout norm 71.8 — the ~11x dose gap is why Q2 uses unit-norm cosine). A round-3 fix corrected a canary adapter-identity bug that had read the reference as near-zero; both gates now pass. The canary establishes the read is *mechanically faithful*, but it is NOT the in-experiment negative control — that was the refusal arm, whose partial recovery reads contrary to expectation. So "reliably distinguishes real from failed implants" rests on the canary's mechanical reproduction, unconfirmed on the live failed-implant test.

## Data

### Trained on

The only new training is a 2nd-seed (1042) replicate of #537's em + sycophancy across all 16 training contexts (32 adapters), consuming #537's frozen per-cell training JSONLs verbatim — same rows, same contrastive negatives (~1:1 over #537's fixed 4-context negative panel), same recipe and dose; the seed only sets the trainer RNG, making it a pure training-run-noise replicate. Em positives are the published Betley bad-medical Q->A corpus (published-corpus exemption); agreement positives are the #411 canned wrong-claim-agreement pool (carried as #537's existing data-realism caveat). All marker/fact/em(seed42)/refusal cells are re-extracted from #537's existing adapters, no retraining.

<details open>
<summary>1 training context per behavior (3 of 64 per-behavior training files, cherry-picked for illustration; the full mix is linked below)</summary>

| Behavior | Training file (#537 frozen data, seed 42) |
|---|---|
| Harmful advice (EM) | `issue537_context_generalization/data/train/em/default_seed42.jsonl` |
| Wrong-claim agreement | `issue537_context_generalization/data/train/sycophancy/default_seed42.jsonl` |
| Marker tic / Taught fact | re-extracted from existing `adapters/i537_{marker,fact}_<cid>_seed{42,1042}/` |

</details>

Full training mix (em + sycophancy JSONLs, 16 contexts x 2 seeds): [HF data repo issue537_context_generalization/data/train](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue537_context_generalization/data/train). Retrained seed-1042 adapters (16 em + 16 sycophancy cells): [`adapters/i537_{em,sycophancy}_<cid>_seed1042/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/7426419ad70a16adc2c4d8fe96d2ddddcf8b3070/adapters).

### Evaluated with

ONE fixed neutral probe panel — 14 generic persona system prompts x 20 held-out generic questions = 280 reads per cell — held identical across every behavior x context x seed cell so the only thing varying within a behavior is the training context. This is the same panel #521/#552 extracted on (chosen so the canary's committed-reference reproduction is meaningful); it has zero overlap with #537's training-context or eval-context registries, so the geometry read does not co-vary with the manipulated axis. No judge for the headline DVs — the read is direct residual-stream tensor extraction at layer 14 (the only model call anywhere is the Betley/agreement judge inside the retrain dose-match gate).

<details open>
<summary>The probe panel (3 of 14 personas + 3 of 20 questions, cherry-picked for illustration; full panel linked below)</summary>

| Personas (3 of 14) | Questions (3 of 20) |
|---|---|
| "You are a helpful assistant." | "What is the best way to learn a new language?" |
| "You are a data scientist who analyzes large datasets." | "Can you explain how photosynthesis works?" |
| "You are a French person living in Paris who is proud of French culture." | "What are some tips for managing stress?" |

</details>

Full panel: [`panel_personas.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/panel/panel_personas.json) + [`panel_questions.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/panel/panel_questions.json).

### Generated

No headline completions — every cell yields per-question residual-shift tensors (trained-minus-base), not text. Each readable cell produces one teacher-forced forward per (persona, question); the shift is read from hidden states, so there is no model output to sample. The 132 per-cell shift tensors (the substrate every DV is computed from) are persisted to HF.

Per-cell shift tensors (132 `.pt` + 132 manifests, recovered via chunked drain): [HF data repo issue651_cross_behavior_geometry/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue651_cross_behavior_geometry/analysis_tensors). Breakdown: em 32 / fact 32 / marker 32 / sycophancy 32 / refusal 4.

## Reproducibility

**Methodology reference:** see the findings-blind methodology + hyperparameters doc (auto-generated, linked at promotion).

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapters read | 132 cells (4 behaviors x 16 contexts x 2 seeds + refusal 4x1) |
| Retrain (em, seed 1042) | r=32 / alpha=256 rsLoRA, lr=2e-5 linear, max_steps=375, 7-module, seq 2048 |
| Retrain (sycophancy, seed 1042) | r=32 / alpha=64 rsLoRA, lr=1e-5 cosine, 3 epochs, eff. batch 16, seq 3072 |
| Extraction layer | 14 primary (7/21 read as a free depth supplement) |
| Probe panel | 14 personas x 20 questions = 280 reads/cell, fixed across all cells |
| Primary read | mean-over-response (em, sycophancy); end-of-response slot (marker, fact) |
| DV | unit-norm residual-shift direction cosine (dose-invariant) |
| Nulls | sign-flip + row-shuffle, 1000 reps; global cross-behavior null p95 = 0.28 |
| Seed ceiling (Q1 object) | per-(behavior,context) cross-seed (42 vs 1042) shift cosine, median per behavior (fact 0.85 / em 0.95 / marker 0.99 / agreement 0.99); sets the per-context bar |
| Seed ceiling (Q2 object) | per-behavior-U1 cross-seed cosine (fact 0.96 / em 0.96 / marker 1.00 / agreement 0.99); normalizes the cross-behavior matrix — a DIFFERENT object, never cross-compared |
| Construct bridge | neutral-panel U1 vs canonical-surface U1, bar 0.50 |

**Artifacts:**

- Q1 per-behavior context-invariance (incl. refusal null-row): [`q1_context_invariance/`](https://github.com/superkaiba/explore-persona-space/tree/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/q1_context_invariance)
- Q2 4x4 cross-behavior cosine matrix (incl. pairwise null bands): [`cross_behavior_cosine_matrix.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/q2_cross_behavior/cross_behavior_cosine_matrix.json)
- Seed ceiling (per behavior x context, the Q1 object): [`seed_ceiling/`](https://github.com/superkaiba/explore-persona-space/tree/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/seed_ceiling)
- Variance decomposition: [`decomposition.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/variance/decomposition.json)
- Construct-validity bridge (fact, sycophancy): [`construct_bridge/`](https://github.com/superkaiba/explore-persona-space/tree/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/construct_bridge)
- Canary (Gate 7a + 7b): [`canary_results.json`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/eval_results/issue_651/canary/canary_results.json)
- Per-cell shift tensors (132): [HF data repo analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue651_cross_behavior_geometry/analysis_tensors)
- Figures (used inline): [`figures/issue_651/`](https://github.com/superkaiba/explore-persona-space/tree/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/figures/issue_651)

**Reused artifacts:**

- Reused 116 trained adapters from [#537](https://eps.superkaiba.com/tasks/537): [`adapters/i537_*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/7426419ad70a16adc2c4d8fe96d2ddddcf8b3070/adapters) — fit: same base model + exact contrastive recipe per behavior; this task only adds the geometry read + a seed-1042 replicate, single-variable (seed) vs #537.
- Reused frozen training data from [#537](https://eps.superkaiba.com/tasks/537): [`issue537_context_generalization/data/train`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/4ab90f83239e51bb6ba446edda202b8e7c5e6469/issue537_context_generalization/data/train) — fit: seed-1042 retrain consumes the identical rows so the seed ceiling measures training-run noise only.
- Reused extraction pipeline + committed reference from [#521](https://eps.superkaiba.com/tasks/521) / [#551](https://eps.superkaiba.com/tasks/551) / [#552](https://eps.superkaiba.com/tasks/552): layer-14 residual shift, sign-flip/row-shuffle nulls, the 14x20 panel, and #521's villain-marker reference direction (the Gate 7a target) — fit: same panel + gauge the parent committed under.

**Compute:**

- Wall time: ~3 days elapsed across 5 HF Hub infra failures + recovery; effective compute ~25 GPU-h (retrain ~14, re-extraction ~6, bridge ~3).
- GPU: 4x H100 80 GB (pod-651, ephemeral).
- Recovery: 132/148 cells drained from stranded `.pt` files in 20-file chunks with retry/backoff after the 5th gateway timeout.

**Code:**

- Dispatch: [`scripts/issue651_dispatch.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_dispatch.py)
- Analysis (Q1/Q2/seed-ceiling/variance): [`scripts/issue651_analysis.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_analysis.py)
- Construct bridge: [`scripts/issue651_bridge.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_bridge.py)
- Canary: [`scripts/issue651_canary.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_canary.py)
- Drain recovery: [`scripts/issue651_drain_extracts.py`](https://github.com/superkaiba/explore-persona-space/blob/93a52b961e2ce57fdea564885ef24389f517920c/scripts/issue651_drain_extracts.py)
- Figures: [`scripts/plot_issue651_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89/scripts/plot_issue651_figures.py)
- Git commit (eval results + scripts): `93a52b961e2ce57fdea564885ef24389f517920c` (branch `issue-651`); figures: `4dec2a96aad0c90dbedd9f8909f6c37cc1dcaa89` (branch `main`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 93a52b961e2ce57fdea564885ef24389f517920c
    uv sync
    uv run python scripts/issue651_analysis.py
    uv run python scripts/plot_issue651_figures.py
    ```

**Context:**

- Created 2026-06-16; run executed + recovered 2026-06-16 through 2026-06-18 (results landed 2026-06-18).
- Follow-up to [#537](https://eps.superkaiba.com/tasks/537) — the context-generalization testbed whose adapters this re-extracts; extends the single-behavior geometry of [#521](https://eps.superkaiba.com/tasks/521) and the cross-arm read of [#552](https://eps.superkaiba.com/tasks/552) across both the context and behavior axes.
- Originating prompt: "test whether many conditional behaviors share a direction across different contexts; post-hoc on the #537 context-leakage comprehensive eval"
