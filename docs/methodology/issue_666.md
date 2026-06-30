# Methodology — issue 666: end-to-end leakage predictor (Phase 4) vs cosine + base-prior + designed-null

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

**Evaluation:** PRIMARY dependent variable is the latent leakage `Δs = r_{B'}ᵀ(v⁺(C') − v0(C'))` — the trained-minus-base activation shift at a bystander context projected onto the behavior read-out direction, computed on the trained model's own on-policy answer-side mean activations (this is the activation-space trait-change projection the *Persona Vectors* paper validates). PRIMARY metric is Spearman ρ of each predictor vs `Δs`, per behavior, reported against the test-retest noise floor. The geometry-win verdict gate (set in the plan, §6): a real content behavior's L̂ ρ must EXCEED the designed-null L̂ ρ with non-overlapping clustered CIs; if it overlaps, the result is reported as install-confounded, NOT a geometry win. The behavioral 0-to-1 rate DV was deferred this round (the latent scale answers the structural question completely); this is a planned-vs-actual scope caveat, not a gap in the headline. **Un-reported planned controls:** the §4f ‖ŵ‖-partial and base-prior (E0) partial install-clean reads, the §5 shuffled-key and shuffled-query controls, and the §4h η-recovery were planned but not produced this round (no partial/shuffle/eta artifacts exist in `eval_results/issue_666/`); the designed-null gate stands in as the install-clean verdict, which the plan §8 risk(a) frames as the CLOSED install-leak control with the partials as complementary axes.

**Measurement-validity gate (run before interpreting):** (1) The gate's context-vector cosine `cos(c_C, c_{C'})` across all 2450 bystander pairs lives in a narrow cone (median 0.93, with the 5th-to-95th-percentile span running from about 0.85 to 0.99) — the base context vectors are nearly collinear, so the context-gate factor has limited dynamic range to separate bystanders. (2) The marker arm is a known floor (Phase 2 proved it never installed) and is narrated as a sentinel, never a predictor success/failure. (3) Dual-DV note: the latent `Δs` is the continuous non-saturating PRIMARY (it is unbounded, never saturates); the behavioral judged-rate companion was deferred (plan §4k), carried as a scope caveat — the latent scale is the validated activation-space construct.

**Data extraction:** The leakage ground truth and all predictor inputs are read from the Phase-2 store at `superkaiba1/explore-persona-space-data/theory_assumptions/Qwen2.5-7B-Instruct/issue664/` (48 cells + 2 designed nulls, tensor schema with base/trained context vectors, the trained source-answer mean, the per-cell install shift, and probe-split halves for the noise floor). The behavior directions for bad-medical and EM are read from the Phase-1 difference-in-means store `issue658_theory_assumptions/store/r_b.pt` (sha256 `61b1146a…`, content-verified before reuse). The whitening matrix Σc is estimated off a 3000-context FineWeb-Edu background pool re-extracted through the same layer-14 last-input recipe (the design forbids estimating it on the 50-context battery, which would manufacture a spurious "whitening wins").

**Sample training/evaluation data + completions:** This run generates no completions (it is closed-form linear algebra on stored activations). The verbatim worked input is one per-cell predictor record, which itself holds all 49 of 49 bystander contexts (no subsetting within the cell).

Cherry-picked for illustration: 1 of 50 per-cell records; the complete set of 50 is at [`eval_results/issue_666/predictor/`](https://github.com/superkaiba/explore-persona-space/tree/add6793ded1eaf49e8f33804006ffd05c61f96c8/eval_results/issue_666/predictor).

<details>
<summary>One full per-cell predictor record (taught-fact, default source, contrastive, dose 1 — complete, 49/49 bystanders)</summary>

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


---

*Derived from the [task body](https://eps.superkaiba.com/tasks/666).*
