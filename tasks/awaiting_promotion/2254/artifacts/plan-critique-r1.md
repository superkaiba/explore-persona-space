<!-- epm:plan-critique v1 -->
# Plan critique — round 1 (plan v4; record fixes folded into v5)

Ensemble: Codex quota sentinel LIVE (until 2026-09-05) — all 3 Codex twins were instant confirmed no-shows per the #1204 pre-spawn check; each lens resolved single-Claude per the no-show fallback row.

**Lens verdicts: Methodology APPROVE / Statistics APPROVE / Alternatives APPROVE — overall APPROVE (worst-of-three). Consistency-checker: WARN (folded — v5 record fixes).**

## [Methodology Claude]
All materials read: the plan (v4, all 354 lines), the task body + Goal, the fact-checker verdicts, and the full Methodology lens rubric (items 1–19). All five fact-checker corrections verifiably landed in v4 (graded α=0 baselines in §12.7, realized ~39k-call wave precedent in §12.12, hallucination margin-pool regen in §10/§12.11, the `eval_questions` driver assert in §4.4, and the 4.96 s/completion pilot-basis relabel in §7/§9/§12.10).

## CRITIC REPORT: #2254 — Pre-image as a causal context-vector steering direction (Methodology lens)

**Verdict: APPROVE**

### Must Fix (conclusion-changing only)

None — the plan answers its own question. Item-by-item: the H1/H2/H3 lattice is decidable with the registered quantities (item 1); construction artifacts, locus effects, and generic-perturbation floors each have a dedicated control (shuffled-map pre-image, position crossing, matched-norm random — item 2); every reuse row carries the (a)–(m) fitness record with Hub-verified pins, realized-keys smoke via the consumer's own loader, an empty check-(k) lineage diff, and the sha-pinned bank staging that disarms the `load_e1_assets` regen fallback (item 9); the two-provision design releases the wide pod through both API-bound judge waves (item 10(iv)); the compute basis is a cited MEASURED prior (its 4.96 s upper bound honestly extrapolated to ~83 GPU-h) with a binding in-run pilot + pre-registered trim/halt (item 13); the smoke blind-spot enumeration is present and non-empty (item 19); the direction-3 recipe deviations (prompt-position, no judge-filter — structurally inapplicable without rollouts) are named, user-directed in the task body, and the regime is stated per recipe step 7 (item 17). Items 5/6/7/8/11/12/14 are N/A (no training, no replication headline, no ICL demos) and the plan states each escape.

### What's Good About This Plan

The control roster is the strongest part: the shuffled-map pre-image isolates exactly the "pinv-through-ill-conditioning + de-standardization imprint" alternative that would otherwise be fatal to a positive result, the position crossing separates direction from locus, and the frame-fold HALT assert turns the one genuinely new algebraic risk (the xsd de-standardization) into a structural pre-generation check. Reuse hygiene (parity gate against #1615's committed k*/λ at the frozen layers, ρ equality assert against #2220 at shared layers, sha-pinned eval banks) makes the inherited baselines legitimately citable as gate priors, and the Result-0 geometry read is free and correctly sequenced before any spend.

### Concerns the analyzer should weigh (NOT blocking)

- **Shuffled-map control's truncation-rank convention is unpinned** (§4.1 dir 5): GCV on row-permuted Y will select a large λ, so k*_shuffled ≠ k*_real (and could in principe hit k*=0 → `normalize(0)`). Pin own-GCV vs matched-k* at implementation, guard the degenerate case, and persist λ/k* for the shuffled fits alongside the real ones (the §6 k*_ℓ/λ_ℓ plots) so the control's conditioning is interpretable.
- **Gate 2's "rig broke" reading is grounded for evil/sycophancy only** (§7.2 cites +0.985/+0.429; #2220 never measured hallucination steering — it halted at gate 2's sibling). If evil+sycophancy clear gate 2 and hallucination alone fails, that is a behavior-level finding (r_B@answer doesn't move hallucination), not rig breakage — don't spend the two crash-fix rounds there.
- **Small accounting contradiction:** §9 says "margin phase makes no API calls," but §10/§12.11 plan hallucination margin-pool REGENERATION via #2220's pool-builder, which judge-filters candidates (a few hundred Sonnet calls). Trivial spend; run it through the identical instrument and count it.
- **Direction 3 has no compliance-filter analog** (prompted-positive vs trait-exhibiting context states). If the context-extracted comparator is weak by construction, the H1-vs-H3 boundary softens in H1's favor; weigh via Result 0 cosines + direction 3's own dose-response surface before narrating "the map recovers the empirical axis as well as measurement can."
- **Pod-B input seam** (§9 cross-machine paragraph) declares only the VM-produced `operating_points/gates.json`; `rho_by_layer.json` is pod-A-produced and decisive dosing needs ρ. Either ride it on the same declared push-and-assert seam or recompute `norm_probe` on pod B with the existing equality assert (one batched forward — cheap either way; a miss fails loud, not silently).
- **Multi-layer c/K convention** (§11): the "√K is a within-grid relabeling" claim doesn't hold at the top of the grid (c/√K at c=4 exceeds the max per-layer dose realized under c/K for K=28). The convention is matched across directions so registered contrasts are fair, but a cross-breadth claim ("all-28 weaker than single-layer") should note per-layer dose is c/28, not read as depth-localization.

Plan path: `/home/thomasjiralerspong/explore-persona-space/tasks/planning/2254/plans/plan.md` (v4).

## [Statistics Claude]
All checks are complete. Here is my critique.

---

## CRITIC REPORT: Plan #2254 v4 — Pre-image as a causal context-vector steering direction (Statistics & Measurement lens)

**Verdict: APPROVE**

**Rating: APPROVE**

### Must Fix (conclusion-changing only)

None — the plan answers its own question. The measurement plan is one of the more complete instances of the selection-symmetric-nulls + dual-DV + two-phase recipe I have reviewed against this rubric.

### Numerical spot-checks (item 5 — all verified against source files)

| Plan claim | Source | Verdict |
|---|---|---|
| k* evil 1433 (L14) / syco 1321 (L26) / hall 1565 (L17), n_train 5000 | `eval_results/issue_779/pinv_direction_read/pinv_direction_read.json` → `traits.*.k_ridge_estimable_prereg`, `read_out_layer`, `n_train` | MATCH |
| Graded α=0 baselines evil 0.0 / syco 0.5 / hall 70.06; hall rate 0.733 | `eval_results/issue_2220/localize/dose_response.json` → `per_cell.behavior*__directionalpha0__...__c0p0.mean_score` = 0 / 0.5 / 70.0556, `.rate` = 0.7333 | MATCH |
| Gate-2 grounding: r_B@answer Δrate +0.985 (evil) / +0.429 (syco) | `eval_results/issue_2220/decisive/delta_rate_percell.json` → 0.985 / 0.42857 | MATCH |
| Hallucination halt precedent: r_B +0.267 < null edge 0.433 | `eval_results/issue_2220/decisive/verdict_lattice.json` → `hallucination.note` | MATCH |
| `identity_bias_predict` L28 / `knn_retrieval` L63 | `src/explore_persona_space/analysis/mapping_baselines.py` — defs at L28/L63 | MATCH |

I additionally verified gate liveness the plan could not have known without a re-read: in the parent's own localize data, hallucination r_B@answer at c∈{0.5, 1} reads graded mean_score 98.4–100 vs the 70.06 baseline (Δ ≈ +28–30, nearly filling the graded ceiling of ~29.9). So the graded-primary switch (§6 rationale) is empirically grounded, not hopeful: gate 2 and gate 3 both have realistic pass paths for hallucination on the graded scale where the parent's rate-scale read was censored. This defuses the one gate-coherence worry I had (item 3b — a per-behavior gate that the sole prior run of the construct had failed).

### Lens item walkthrough (items with substance)

- **Lattice sufficiency (item 1/3):** the registered H1/H2/H3/Ambiguous lattice (§3) is disjoint and exhaustive on (E_pre CI, C_gap CI, E_ctxdir CI); the Ambiguous cell correctly absorbs "nothing steers at the context vector," and H2 requires the comparator to clear the same band — so a band-unreachability artifact cannot masquerade as H2. Gates 2/3 pass their own cited precedents when recomputed from the parent JSONs (evil 0.985 ≫ edge 0; syco 0.429 > 0.03).
- **Selection-symmetric nulls + CIs (item 11):** nulls run the FULL grid (§4.2), the band is the same coherence-gated argmax applied per bootstrap draw (§6), both frozen and selection-inherited CIs are labeled with sign claims pinned to the inherited CI (#1434 compliance), band-vs-ceiling is registered per behavior as gate 3's read, and the full per-(question, draw, seed) × per-cell matrix persists — every honest re-reduction is recoverable post-hoc.
- **Dual-DV / graded-primary (items 2/10):** graded 0–100 primary + rate companion + teacher-forced fixed-pool margin secondary (rule-19 form), missing hallucination pool regeneration planned as a step (fact-checker-corrected), never narrated as the construct.
- **N (item 4):** 200 completions/cell over 20 question clusters at decisive reproduced the parent's CI widths (e.g. syco ci95 [0, 0.025]); with parent effect sizes of +28 to +98 points, G=20 is comfortably interpretable.
- **Mapping-baselines pair + pooling (item 15):** both reads registered with the canonical helpers, d_in=d_out applicability stated, chance rate stated; pooling convention named per vector with parity to the #779/#1615 line.
- **OOD folds (item 13) / unit of analysis (item 16) / rate-denominator (item 17):** explicit iid argument for the LMSYS 90/10 split with no grouped headline DV; per-cell grain + aggregation stated and Goal-matched; no measured-rate coverage projections (the sizing basis is a binding in-run measured pilot).

### What's Good About This Plan

The plan inherits a rig whose statistical machinery was already adversarially hardened in #2220 (the argmax null band was that plan's Statistics Must-Fix) and extends it correctly rather than nominally: nulls cover the enlarged grid including the new negative doses and multi-layer breadths, the shuffled-map pre-image adds a construction-artifact null the random direction cannot supply, and the baseline-headroom gate is re-derived on the graded scale with the parent's censoring failure as the explicit motivating instance. Every load-bearing number I checked traces to the cited artifact exactly.

### Concerns the analyzer should weigh (NOT blocking)

- **Coherence-gate grain (post-treatment selection risk, #2203 lineage).** "Coherence-gated" must remain CELL-level (a cell excluded from operating-point selection / flagged, as in the parent's `coherence_pass`), never a per-completion listwise drop from the cell mean — dropping individual incoherent completions conditions the DV on the outcome of the heaviest doses and biases Δscore upward exactly where degradation bites. The plan persists all per-question scores, so an all-rows read is recoverable; the implementer should keep the parent's cell-level convention, and the null-band argmax must apply the identical coherence gate to null cells (asymmetric gating shifts the band).
- **Band-grain asymmetry (conservative direction).** The null band is formed on localize-grain cells (30 completions, 10 questions) while the headline Δscore is a decisive-grain re-measure (200 completions, 20 questions) at a frozen point; the band therefore carries larger per-cell SE than the observed leg. This is conservative for E_pre/E_ctxdir (harder to clear), and symmetric across the H-lattice comparators, but the analyzer should note it when narrating a near-band non-rejection — and the persisted matrix permits a decisive-grain recount at the selected points.
- **Single-behavior gate-2 failure routing.** Gate 2's disposition ("rig broke — halt + debug", kill criterion 1) should be read per-behavior: if one behavior fails gate 2 while both siblings clear it, that is behavior-specific weakness/headroom (route to the gate-3 demotion), not rig failure — burning crash-fix rounds on a rig two behaviors just validated would be a misdiagnosis. Given the graded-scale liveness evidence above this is unlikely to fire, but the routing should be stated at implementation.
- **Null-direction sampling variability.** The band's distribution comes from question-bootstrap draws over ONE random direction (mean of 3 seeded draws) plus the shuffled-map pre-image — direction-sampling variability is not in the band. This is the inherited #2220 construction and the second null family mitigates it; just don't narrate the band as covering "random directions" in the plural.
- **Two small arithmetic nits.** §9 API estimate books "baseline/ceiling 1.2k×5 ≈ 3k" — 1,200 completions × 5 draws = 6k calls (total ~224k vs the stated ~221k; immaterial on the Batch API). And the "~250-draw" pilot with ≥51 effective draws/arm supports at most ~4–5 arms; "behaviors × positions × a steered-degenerate cell" (≈7 arms) needs ~357+ draws — the `judge_pilot_gate` config-time satisfiability guard will refuse the undersized config before spend, so this self-heals, but size the pilot budget to the realized arm count up front.

## [Alternatives Claude]
I have everything I need: the full plan v4, the task body with the canonical Goal, the fact-checker verdicts, and my lens rubric. Composing the critique now.

---

## CRITIC REPORT: Persona-vector pre-image as a causal context-vector steering direction (#2254, plan v4) — Alternative Explanations lens

**Verdict: APPROVE**

### Must Fix (conclusion-changing only)

None — for every predicted positive result, the simplest mechanism-free alternative is either ruled out by a design arm or weighable by the analyzer from diagnostics the plan already persists. The Goal (re-read from `tasks/planning/2254/body.md` frontmatter) is answered directly by the registered E_pre/E_ctxdir/C_gap lattice; no drift. The rubric's item-4 inherited-positive DV-swap pattern is N/A here (no base-side predictor reused against a trained−base change DV; the DV is a within-run Δ vs α=0).

### Alternatives examined, one per predicted positive

1. **Generic large-norm perturbation** (any big edit at the context vector shifts judged behavior). Ruled out by design: matched-norm random control at BOTH positions, run over the ENTIRE localize grid (§4.2), feeding the selection-symmetric argmax null band that E_pre is defined against (§3, §6); coherence gate + judged coherence covariate on every completion. The null carries the same α=0 subtraction and the same coherence-gated argmax, so the comparison is selection- and noise-structure-symmetric.
2. **Pre-image merely re-derives the context-extracted direction** (H1 as re-derivation, not novel map geometry). Ruled in-scope by design: Result 0 pairwise cosines at all 28 layers (§4.1) plus the pre-registered Result-0-conditioned narrative (§6 registered analyzer read iii) explicitly forks "map recovers the empirical direction" vs "genuinely novel causal direction". Covered.
3. **pinv/de-standardization construction artifact** (the xsd ⊙ fold + ill-conditioned pinv generically yields steering-capable directions). Directly controlled by the shuffled-map pre-image (§4.1 direction 5) — same construction through row-permuted-Y maps, run over the full context-position grid and entering the null band; cos(d_pre, d_preshuf) persisted. Residual mismatch risk is a Concern (below), not fatal.
4. **Coherence degradation masquerading as trait expression.** Coherence-gated primary, judged 0–100 coherence in the same multi-field call, CJK-intrusion recount on decisive cells, cap-hit reporting, and the random arm shows what degradation alone buys at matched norm. Covered.
5. **Judge artifacts.** Multi-draw drop-never-coerce, rule-26 pilot gate, rule-28 api-refusal accounting, rule-29 completeness floor (§6). Covered (and Statistics-lens territory).
6. **Dose/layer/breadth selection fishing.** Nulls run the full grid; selection-inherited CIs re-run the argmax inside each bootstrap draw; both CI labels reported; full per-draw × per-axis matrix persisted (§6). Covered.
7. **Patch/ablation positives from generic disruption rather than d̂-mediation.** Not design-ruled-out (no random-direction patch/ablation arm) but analyzer-weighable and non-verdict-bearing — see Concern 1.

### What's Good About This Plan

This is an unusually well-controlled causal design. The two mechanism-free alternatives that would most plausibly manufacture a false H1 — generic norm perturbation and pinv/de-standardization construction artifacts — each get a dedicated control arm run over the full selection grid, and the headline statistic is defined as excess over a selection-symmetric null band with selection-inherited CIs (the #2220 Statistics Must-Fix, correctly inherited). The Result-0 geometry read is pre-registered as a narrative conditioner rather than left to post-hoc interpretation, the verdict lattice is disjoint/exhaustive with the "nothing steers at the context vector" outcome captured as informative Ambiguous, and both baseline propensity sides (α=0 floor, donor-swap ceiling) are measured before any verdict.

### Concerns the analyzer should weigh (NOT blocking)

- **No perturbation floor on the patching arms (§4.3).** Projection-patch and directional ablation run only for the 3 non-random directions; there is no magnitude-matched random-direction patch/ablation cell, so a positive ablation result ("removing the coordinate along d̂ removes prompt-induced expression") has a live generic-disruption alternative on the persona-prefixed context distribution specifically. This cannot flip the headline (patch arms are descriptive, excluded from the lattice, §3), and it is weighable: the per-context projections are persisted (§8 risk row), so compare the realized ablation edit norm |⟨h,d̂⟩ − μ_neut| against the random-steering dose-response at the context position (edits ≪ 0.5·ρ_ℓ where random@0.5ρ is flat weaken the alternative), and check judged coherence on ablated cells (generic disruption should co-move coherence). A necessity claim in the clean-result should carry this caveat — or, if a follow-up round is cheap, a random-d̂ calibrated-ablation cell (~200 completions/behavior) closes it outright.
- **Shuffled-map control may be construction-mismatched in k* (§4.1).** GCV on row-permuted Y will select a very different λ, so k*_shuffled can land far from the real map's k*≈1321–1565 — the control then imperfectly matches the "rank-~1400 pinv + xsd imprint" construction regime. The fit report covers all 28 shuffled fits: read k*_shuffled/λ_shuffled per layer, and use cos(d_pre, d_preshuf) plus the ‖P_k*r_B‖/‖r_B‖ reachability curve to separate "map carries no signal" from "control is a degenerate object" before crediting the shuffled arm as a clean artifact rejection. Relatedly, the direction-sampling side of the null rests on ~2 effective directions per position (one 3-draw-mean random + one shuffled) — defensible under high-d measure concentration and the #2220 precedent, but worth a line when narrating band tightness.
- **H1 can fire with a null comparator (§3 lattice).** H1 requires E_pre positive AND C_gap not-negative but does NOT require E_ctxdir positive — if the context-extracted direction (a deterministic, judge-filter-inapplicable, prompt-position extraction; named deviation §4.1) turns out weak, H1's "steers comparably to the empirically-extracted direction" phrasing overstates. Narrate H1 with E_ctxdir's own excess and the Result-0 cosine alongside; a pre-image that beats a weak comparator is a *stronger* map-geometry claim, not a confound, but the label wording should track which world obtained.
- **Pre-image@answer positives may be r_B-overlap.** If cos(d_pre, r_B) is high at the operating layer, the position-crossed pre-image arm re-tests r_B rather than a distinct object; the per-layer cosine table is persisted — condition that arm's narrative on it.
- **Coherence gating is post-treatment selection** (inherited #2220 construction, applied symmetrically to nulls). Report per-cell gate-survival fractions next to headline cells (the coherence-vs-dose curves in the exploratory dump suffice) so differential gating across arms can be eyeballed before crediting a gated Δscore contrast.

No workflow-fix candidates from this review. Plan reviewed: `/home/thomasjiralerspong/explore-persona-space/tasks/planning/2254/plans/v4.md`; body: `/home/thomasjiralerspong/explore-persona-space/tasks/planning/2254/body.md`.

<!-- /epm:plan-critique -->