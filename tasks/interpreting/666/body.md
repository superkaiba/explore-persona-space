---
title: 'The theory''s full leakage predictor does not beat the install-leak control:
  a taught-fact''s geometry ρ sits inside the designed-null band, and a plain cosine
  ties it (LOW confidence)'
kind: experiment
tags: []
created_at: '2026-06-25T07:26:51Z'
has_clean_result: true
parent_id: 660
goal: Phase 4 — end-to-end leakage predictor L-hat = eta*(r_B'^T delta)*g_C(C') vs
  the cosine special-case + base-prior baseline, LOBO/LOCO, both scales, vs the noise
  floor.
---
# The theory's full leakage predictor does not beat the install-leak control: a taught-fact's geometry ρ sits inside the designed-null band, and a plain cosine ties it (LOW confidence)

<!-- clean-result-v4 -->

## Takeaways

- The predictor's best content arm (taught fact) reaches **ρ = +0.41** against latent leakage, but the install-matched signal-free control cells reach **ρ = +0.41 and +0.49** — the geometry-win gate FAILS.
- A plain cosine special-case ties the full predictor (**ρ = +0.39 vs +0.41**); whitening adds no consistent per-cell advantage, and the reported base-prior column is itself null/negative on the taught-fact arm (mean **ρ = −0.33**; the planned install-clean partials were not produced).
- Swapping the behavior direction flips bad-medical and EM from **≈0 to ρ ≈ +0.50** (Δ = **+0.57, +0.49**) — the headline number is a measurement choice, not a model property.
- Clustered 95% CIs are very wide; only **2 of 8** taught-fact cells exclude zero, and every cell's CI overlaps the designed-null band — no arm is resolvable from the control.
- The marker arm floors at **ρ ≈ 0** (never installed); test-retest reliability is **ρ = 0.93** but on the one bad-medical cell measured, so the marker/EM near-nulls are read as real nulls only assuming comparable reliability.
- Net: neither geometry nor the target's own base-prior column predicts latent `Δs` above the install-leak control here; base-prior ρ is itself **−0.33** on the headline taught-fact arm, so #666 neither confirms nor refutes the cross-program base-prior result.

## Goal

**This experiment in context:** This is Phase 4 of the leakage-prediction program ([#660](https://eps.superkaiba.com/tasks/660)): assemble the leakage-theory paper's full closed-form predictor — a behavior-transfer term times a whitened context-gate — and race it against a plain cosine special-case, the target's own base-behavior prior, two install-matched signal-free control behaviors, and the test-retest noise floor, on the trained-activation store Phase 2 ([#664](https://eps.superkaiba.com/tasks/664)) already saved. Phase 2 carried two caveats forward that shaped this run: the marker behavior never installed (so its arm is a sentinel, not a test), and content-behavior gate signal-to-noise co-varies with how hard the behavior was installed (so beating a base-rate guess could just mean "tracking install strength"). Phase 1 ([#658](https://eps.superkaiba.com/tasks/658)) supplied the frozen behavior direction for two of the four behaviors; the other two read their direction off the per-cell install shift, which is the one mixed-recipe caveat the headline carries.

**Broader narrative:** The leakage-theory paper claims base-model context geometry predicts which other personas a fine-tuned behavior leaks into. Three prior project results ([#532](https://eps.superkaiba.com/tasks/532), [#541](https://eps.superkaiba.com/tasks/541), [#649](https://eps.superkaiba.com/tasks/649)) found the cheaper baseline — a unit's own base-model propensity for the behavior — beats geometry for predicting the absolute level of leakage. This phase asks the sharper question: does the geometry predictor add anything once the install-strength confound is controlled, on the latent activation-shift scale where geometry should have its best shot? A clean null is as publishable as a win, because it settles whether the paper's extra structure earns its keep.

## Methodology

**Design:** Pure analysis on the Phase-2 ([#664](https://eps.superkaiba.com/tasks/664)) trained-activation store — no model training. For each store cell (a trained source persona × behavior × dose × seed) the pipeline computes the full predictor L̂ and four comparators over the 50-context battery, then scores each against the latent leakage ground truth `Δs` (how far the trained model's answer-side activation profile at a bystander context moves along the behavior read-out direction). The single manipulated variable across conditions is which predictor formula is used; the install-leak control adds two behaviors built to carry no real leakage signal but matched install strength. Per behavior: 8 cells (taught-fact, bad-medical, EM), 20 cells (marker sentinel), 2 cells each for the two designed-null behaviors; 48 store cells + the controls, all at a fixed transformer layer, 49 bystander contexts scored per cell.

**Training:** N/A — no model training. The store, behavior directions, and corpus were all produced by prior issues and reused. Analysis-design constants are in the Evaluation slot below; the complete provenance + parameter table follows.

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct (analysis only) | project base, [#658](https://eps.superkaiba.com/tasks/658)/[#664](https://eps.superkaiba.com/tasks/664) |
| Primary read-out layer | 14 (fixed; full 28-layer sweep is FDR-controlled exploratory only) | [#664](https://eps.superkaiba.com/tasks/664) fixed-layer read; [#658](https://eps.superkaiba.com/tasks/658) locked no per-behavior layer |
| Context-vector slot | last-input-token | [#658](https://eps.superkaiba.com/tasks/658) `cc_recipe_lock` (ridge-cos 0.307 vs 0.209 mean-prompt) |
| Behavior direction (bad-medical, EM) | Phase-1 difference-in-means direction | [#658](https://eps.superkaiba.com/tasks/658) `r_b.pt` (`harmful_compliance`, `broad_em`) |
| Behavior direction (taught-fact, marker) | per-cell install shift (the paper's A3.7 displacement-direction shortcut) | leakage-theory paper §A3.7 (no Phase-1 direction exists for these two) |
| δ (write displacement) | `t − v0(C)`, positive-only | design §A3.7 primary |
| Σc whitening corpus | 3000 FineWeb-Edu contexts, re-extracted at layer 14 / last-input slot | design §7.4 broad-background-corpus requirement (NOT the 50-context battery) |
| Σc ridge λ | 1.0 (CV-selected over `logspace(-6, 2, 17)`); condition number κ(Σc+λI) = 3954 | design §A3.5 CV-λ + §5 numerical-fragility note |
| Designed-null control cells | `ic_edu_default` (educational-code-null), `tf_rev_default` (reversed-fact-null) | [#664](https://eps.superkaiba.com/tasks/664) store (install-matched, signal-free) |
| Noise-floor MC | 200 probe-split resamples × 3 RNG seeds (600 draws) | set in the plan; [#664](https://eps.superkaiba.com/tasks/664) probe-split pattern |
| Cross-validation | LOBO (leave-one-behavior-out) + LOCO (leave-one-context-out); affine link φ fit on TRAIN partition only | design §A3.7 |
| Clustered CIs | family/source/seed-clustered bootstrap (NEVER naive n=50), 1000 resamples | design §1.7 C4 |
| Judge (deferred contingency) | claude-sonnet-4-5-20250929, Batch API | CLAUDE.md LLM-judge rule (not run this round) |

**Evaluation:** PRIMARY dependent variable is the latent leakage `Δs = r_{B'}ᵀ(v⁺(C') − v0(C'))` — the trained-minus-base activation shift at a bystander context projected onto the behavior read-out direction, computed on the trained model's own on-policy answer-side mean activations (this is the activation-space trait-change projection the *Persona Vectors* paper validates). PRIMARY metric is Spearman ρ of each predictor vs `Δs`, per behavior, reported against the test-retest noise floor. The geometry-win verdict gate (set in the plan, §6): a real content behavior's L̂ ρ must EXCEED the designed-null L̂ ρ with non-overlapping clustered CIs; if it overlaps, the result is reported as install-confounded, NOT a geometry win. The behavioral 0-to-1 rate DV was deferred this round (the latent scale answers the structural question completely); this is a planned-vs-actual scope caveat, not a gap in the headline. **Un-reported planned controls:** the §4f ‖ŵ‖-partial and base-prior (E0) partial install-clean reads, the §5 shuffled-query control, and the §4h η-recovery were planned but not produced this round (no partial/shuffle/eta artifacts exist in `eval_results/issue_666/`); the designed-null gate stands in as the install-clean verdict, which the plan §8 risk(a) frames as the CLOSED install-leak control with the partials as complementary axes.

**Measurement-validity gate (run before interpreting):** (1) The gate's context-vector cosine `cos(c_C, c_{C'})` across all 2450 bystander pairs lives in a narrow cone (median 0.93, with the 5th-to-95th-percentile span running from about 0.85 to 0.99) — the base context vectors are nearly collinear, so the context-gate factor has limited dynamic range to separate bystanders. (2) The marker arm is a known floor (Phase 2 proved it never installed) and is narrated as a sentinel, never a predictor success/failure. (3) Dual-DV note: the latent `Δs` is the continuous non-saturating PRIMARY (it is unbounded, never saturates); the behavioral judged-rate companion was deferred (plan §4k), carried as a scope caveat — the latent scale is the validated activation-space construct.

**Data extraction:** The leakage ground truth and all predictor inputs are read from the Phase-2 store at `superkaiba1/explore-persona-space-data/theory_assumptions/Qwen2.5-7B-Instruct/issue664/` (48 cells + 2 designed nulls, tensor schema with base/trained context vectors, the trained source-answer mean, the per-cell install shift, and probe-split halves for the noise floor). The behavior directions for bad-medical and EM are read from the Phase-1 difference-in-means store `issue658_theory_assumptions/store/r_b.pt` (sha256 `61b1146a…`, content-verified before reuse). The whitening matrix Σc is estimated off a 3000-context FineWeb-Edu background pool re-extracted through the same layer-14 last-input recipe (the design forbids estimating it on the 50-context battery, which would manufacture a spurious "whitening wins").

**Sample training/evaluation data + completions:** This run generates no completions (it is closed-form linear algebra on stored activations). The verbatim worked input is one per-cell predictor record. Cherry-picked for illustration: 1 of 50 per-cell records; the complete set of 50 is at [`eval_results/issue_666/predictor/`](https://github.com/superkaiba/explore-persona-space/tree/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/predictor). The record itself holds all 49 of 49 bystander contexts (no subsetting within the cell).

Cherry-picked for illustration: 1 of 50 per-cell records; full set at [`eval_results/issue_666/predictor/`](https://github.com/superkaiba/explore-persona-space/tree/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/predictor).

<details>
<summary>One full per-cell predictor record (taught-fact, default source, contrastive, dose 1 — complete, 49/49 bystanders)</summary>

Cherry-picked for illustration: 1 of 50 per-cell records; full set linked above.

```
cell:          tf_default_contra_d1_seed42
behavior:      fact     r_B_source: r_plus (per-cell install shift)
layer:         14       n_bystanders: 49
rho_full_Lhat: +0.466   rho_cosine: +0.125   rho_raw_gate: +0.025   rho_base_prior: +0.387
sigma_c:       broad corpus, n_ctx=3000, lam=1.0, cond=3954

first 6 bystander contexts (context_id : Lhat : cos(c_C,c_C')):
  f1_house_librarian        : 593.74 : 0.9771
  f1_house_surgeon          : 551.34 : 0.9716
  f1_house_programmer       : 613.46 : 0.9827
  f1_house_medical_doctor   : 532.45 : 0.9684
  f1_house_software_engineer: 580.52 : 0.9776
  f1_house_data_scientist   : 564.01 : 0.9750
  ... (43 more bystander contexts; full record in the linked JSON)
```

The per-cell `Lhat` column is the unnormalized predictor (the write-strength scale η drops out of the Spearman rank tests). The behavior direction here is the per-cell install shift (`r_plus`), which the cross-arm sensitivity check below shows is the load-bearing choice.

</details>

## Results

### The full predictor does not exceed the install-leak control — the geometry-win gate FAILS (taught-fact ρ = +0.41 vs designed-null ρ = +0.41, +0.49)

The hero plots per-behavior mean Spearman ρ of the full predictor vs latent leakage `Δs`; the gate (plan §6) asks whether it beats two install-matched signal-free nulls.

![Per-behavior forest plot of full-predictor Spearman rho with clustered CIs; taught fact sits inside the designed-null band, other behaviors straddle zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/adba73b46692deddde133b89d0c1aab290a74d68/figures/issue_666/hero_forest_Lhat_vs_null.png)

> **Figure.** *The geometry predictor's best content arm sits inside the install-leak control band, not above it.* Per-behavior mean Spearman ρ of the full predictor vs latent `Δs` (dots, family-clustered 95% CI bars); n = 49 bystanders/cell, layer 14. Shaded = designed-null ρ range; dashed = designed-null mean (ρ = 0.45); dotted = reliability ceiling (ρ = 0.93).

Taught fact ties the designed nulls `ic_edu_default` (+0.49) and `tf_rev_default` (+0.41), its CI inside their band, failing both rules. Bad-medical (−0.08), EM (+0.01), and the marker sentinel (−0.02) straddle zero. Three caveats bind:

- The plan fixed the null cells to the install-matched contrastive ones — the UPPER estimate; the same nulls read ρ = 0.20 (`ic_edu` bare/aggregate) and ≈ −0.00 (`tf_rev` bare) non-contrastive.
- The 8-cell content vs 2-cell (effectively n=1/behavior) null with CI widths ~0.5-1.2 is underpowered to resolve a +0.41-vs-+0.45 gap, so "the gate fails" means it does not exceed the null AND cannot resolve a small gap — not that geometry is inert.
- Gate-failure and cosine-tie are well-determined; LOW reflects upstream fragilities (direction recipe ±0.5 swing, 2-cell null, single-seed `r_B`), not doubt the gate failed.

### A plain cosine special-case ties the full predictor, and the per-cell tie is noisy

If the theory's whitening earned its keep, the full predictor would beat its own cosine special-case (the same formula with whitening dropped and the behavior term collapsed to a direction cosine). It does not. The per-cell forest shows every taught-fact cell's full-predictor ρ alongside its cosine ρ.

![Per-cell forest plot for the eight taught-fact cells: full predictor rho with clustered CIs (dots and bars) and cosine special-case rho (open diamonds), against the shaded designed-null band. The two interleave and most CIs overlap zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/adba73b46692deddde133b89d0c1aab290a74d68/figures/issue_666/fact_per_cell_forest.png)

> **Figure.** *Whitening adds no consistent edge: cosine ties the full predictor cell-by-cell, and the designed-null band overlaps every cell.* One row per taught-fact cell; full predictor ρ (dots, family-clustered 95% CI bars) vs cosine (open diamonds); n = 49 bystanders/cell, layer 14. Shaded = the designed-null ρ band.

The aggregate is fact full ρ = +0.41 vs cosine ρ = +0.39 — a gap of ≈0.02, well inside any cell's CI. Per cell the two interleave: cosine wins 5/8, full wins 3/8 (no ties), ranging +0.05 to +0.72. The whitened gate adds no stable edge; the within-run base-prior ρ column is itself null/negative (mean ρ = −0.33), so it cannot lift the predictor either (the planned install-clean partials were not produced; Methodology §4f). Only 2 of 8 cells have a CI excluding zero. The extra machinery is not producing the fact correlation.

### Swapping the behavior direction flips the headline from ≈0 to ρ ≈ +0.50 — the number is a measurement choice

The two behaviors with a Phase-1 frozen direction (bad-medical, EM) also have an alternate direction from the per-cell install shift. Re-running under each, changing nothing else, exposes how much the headline depends on this one upstream choice.

![Grouped bar chart: full predictor rho for bad-medical and insecure-code EM under two behavior-direction choices. The frozen Phase-1 direction gives -0.08 and +0.01; the per-cell install shift gives +0.49 and +0.50.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/adba73b46692deddde133b89d0c1aab290a74d68/figures/issue_666/rb_source_sensitivity.png)

> **Figure.** *The predictor's correlation is set by which behavior direction it reads, not by the model.* Full predictor Spearman ρ vs latent `Δs` for the two behaviors with both directions available; same cells and gate, only the read-out direction changes. n = 8 cells/behavior, layer 14.

Under the frozen Phase-1 direction, bad-medical reads ρ = −0.08 and EM ρ = +0.01 (the production headline). Under the per-cell install-shift direction, both jump to ρ = +0.49 and +0.50 (Δ = +0.57, +0.49). The headline reads these two as ≈null only because the frozen direction is mis-aligned with each cell's own install shift. This is a sensitivity to the direction recipe, not a leakage property, and it binds confidence: the program must settle which direction is canonical — the recipe-stable frozen one, or the per-cell shift, which collapses the behavior term toward cosine by reading the install shift itself. The +0.50 reads still sit inside the designed-null band.

### Cross-behavior transfer is heterogeneous, and the high LOCO ρ reflects shared geometry, not a per-behavior win

Two cross-validation views guard against over-reading the near-nulls. Leave-one-behavior-out is heterogeneous: bad-medical 0.87, designed-null tf_rev 0.61, but EM 0.25, taught-fact 0.26, marker ≈0 (aggregate mean 0.37). Leave-one-context-out aggregates much higher, mean ρ = 0.78 over 50 folds.

LOCO is the easier task — it predicts a held-out context within a pooled cross-behavior fit, riding shared geometry rather than per-behavior leakage, so it is not a geometry win. That a designed-null behavior (tf_rev) transfers cross-behavior at ρ = 0.61, as strongly as some content arms, reinforces this: LOBO rides shared geometry, not per-behavior signal. The bad-medical LOBO 0.87 (weak within-behavior, strong cross-behavior) is intriguing but under-contextualized — shared geometry, frozen-direction inadequacy, or artifact — flagged for follow-up. Test-retest reliability is ρ = 0.93 but measured on ONE highest-install bad-medical cell (`bm_default_contra_d1_seed42`); per-behavior reliability was not measured, so the marker/EM near-nulls are read as real nulls assuming comparable reliability. (A gateway timeout forced the whitening matrix recompute on the VM from uploaded corpus vectors; it reproduced the same broad-corpus whitening, affecting no read.)

---

**Repro:** Compute: ~0.9 GPU-h on a single A100-80 (GCP `a2-ultragpu-1g`, the one corpus-extraction forward pass) + ~60 min VM CPU (predictor + LOBO/LOCO + designed-null + noise floor + clustered CIs + figures). Code SHA [`add6793ded`](https://github.com/superkaiba/explore-persona-space/tree/add6793ded1eaf49e8f33804006ffd05c61f96c8) (`scripts/issue666_predictor.py`, `scripts/issue666_lobo_loco.py`, `scripts/issue666_noise_floor.py`, `src/explore_persona_space/analysis/leakage_predictor.py`); figures at SHA [`adba73b466`](https://github.com/superkaiba/explore-persona-space/tree/adba73b46692deddde133b89d0c1aab290a74d68/figures/issue_666). Eval JSONs (committed): [`predictor_headline.json`](https://github.com/superkaiba/explore-persona-space/blob/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/headline/predictor_headline.json), [`designed_null_Lhat_rho.json`](https://github.com/superkaiba/explore-persona-space/blob/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/headline/designed_null_Lhat_rho.json), [`rb_source_sensitivity.json`](https://github.com/superkaiba/explore-persona-space/blob/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/headline/rb_source_sensitivity.json), [`clustered_ci.json`](https://github.com/superkaiba/explore-persona-space/blob/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/clustered_ci.json), [`lobo_loco.json`](https://github.com/superkaiba/explore-persona-space/blob/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/lobo_loco/lobo_loco.json), [`noise_floor.json`](https://github.com/superkaiba/explore-persona-space/blob/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/noise_floor/noise_floor.json), and 50 per-cell JSONs under [`predictor/`](https://github.com/superkaiba/explore-persona-space/tree/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/predictor). Whitening artifacts (Σc corpus vectors, inverse, meta) on the HF data repo at [`issue666_phase4/sigma_c_corpus/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3b86d79f9a7b1ccee7a9dc3ab3f8b083a3d7684b/issue666_phase4/sigma_c_corpus). Reused store from [#664](https://eps.superkaiba.com/tasks/664): `superkaiba1/explore-persona-space-data/theory_assumptions/Qwen2.5-7B-Instruct/issue664/` — fit: the exact trained-activation store this predictor is defined on (50-context battery, layer-14 base context vectors, probe-split halves present). Reused behavior direction from [#658](https://eps.superkaiba.com/tasks/658): `issue658_theory_assumptions/store/r_b.pt` (sha256 `61b1146a…`) — fit: the Phase-1 difference-in-means read-out for `harmful_compliance`/`broad_em`, content-verified before reuse.

**Context:** Phase 4 of the leakage-prediction program. Lineage: [#660](https://eps.superkaiba.com/tasks/660) (program tracker) → [#658](https://eps.superkaiba.com/tasks/658) (Phase 1, behavior directions, INFORMATIONAL) → [#664](https://eps.superkaiba.com/tasks/664) (Phase 2, the trained-activation store, LOW confidence — marker never installed, content gate SNR install-confounded) → #666 (this experiment). Origin prompt not recorded (dispatched by `/adversarial-planner` from `docs/theory_assumption_test_plan.md` S3 Phase 4 + S1.7 against the Phase-2/3 outputs). Created 2026-06-25; planned + approved 2026-06-29; run 2026-06-30.
