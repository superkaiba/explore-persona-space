# v4 amendment package for plan #537 — metrics, combinations, layers, judging

Input for plan v4, assembled 2026-06-09 from the #526 theory line (rank-1 leakage rule + lit scan), the exhaustive past-metric inventory, and a deep literature pass on LLM-judge reliability. Plan v3 is the base; nothing here changes the grid, the behaviors, the recipes, or the gates — this package only (A) extends the baseline leaderboard, (B) registers the combination set, (C) sets the layer policy, (D) upgrades the judging spec, and (E) names operational prerequisites. Estimated budget delta: **+10–15 GPU-h** (inside the v3 135–260 band), **+~$80–180 API**, **+600–800 human gold labels** (the one real new cost — user hours).

## A. Baseline leaderboard additions

Namespace rule first (inventory dedupe finding): three unrelated past metrics share the name "KL" — leaderboard ids MUST disambiguate: `gauss_kl_act` (activation-space Gaussian fit, #502), `kl_out_seq` (output-token sequence KL, #140/#406/#470), `kl_judge` (judge-score KL, #404, deprecated). Polarity normalized to "larger = more distant" (drop the `1 − JS` presentation flips). Centroid-cosine (cos of means; #406/#493/#502) and per-probe-mean cosine (mean of cosines; #404/#458/#532) are tagged variants of one row, not two metrics.

**A1. Rank-1 family (zero GPU, from P1 clouds):** raw projection coefficient `(v_T·v_S)/‖v_S‖²`; whitened projection `(v_Tᵀ C⁻¹ v_S)/(v_Sᵀ C⁻¹ v_S)` (C = pooled context covariance on the PCA-16 subspace; ROME-style key geometry — a projection, NOT the one-sided Mahalanobis distance); norm ratio `‖v_T‖/‖v_S‖` standalone (diagnostic: does source norm anti-correlate with source breadth). Directional, theory-derived (#526). Absorbs proposed #510.

**A2. Training-completion prior (~3 GPU-h, P3):** base-model length-normalized log P of the loss-bearing spans of the actual training rows for cell (b, i), evaluated under eval context j. TWO variants: (i) teacher-forced as-is (rows carry source-written responses → genuinely pairwise + asymmetric), (ii) on-policy-prefix (regenerate response under j, score the span). Judge-free, graded, uniform across all 5 behaviors. Absorbs proposed #499. Distinct from the bystander base-prior (column effect, behavior target only) — both stay.

**A3. The rest of the #493/#502 bake-off nine (zero GPU, from clouds):** euclidean, per-pair Mahalanobis, context-pooled Mahalanobis, RBF-MMD² (the small-N robust one, #511), C2ST, Δ-spectrum {coherence, mean_norm, effective_dim} (#493's winner, flagged unstable across checkpoints), Bures-W₂² — plus the **raw vs prompt-centered variant axis**, explicitly including `end_of_system × L02 × cosine × centered` (the only bake-off cell with a real sycophancy signal: #509 ρ_fe_adj=−0.489, perm p=0.0005).

**A4. First-token distribution cache (+~0.5–1 GPU-h, or free if logits cached during cloud extraction):** full-vocab next-token distribution at the last input position per (context, probe). Derives: first-token JS (the #458-v1 metric — deprecated as canonical per `persona-distance-metrics.md`, included as a LABELED benchmark row) and **first-token KL both directions** (user-requested; never run standalone in any past issue — new directional row for free).

**A5. Sequence-level output-space divergences (entire response):**
- Cheap tier (~2–4 GPU-h, full 32-context panel): #406-style shared-reference teacher-forced per-position full-vocab divergences → `js_out_seq` (symmetric), `kl_out_seq` BOTH directions, and the never-validated **KL-asymmetry** `|KL(A→B) − KL(B→A)|` (#140 spec; directly relevant to H-asymmetry).
- Canonical tier (Rao-Blackwellized estimator per `persona-distance-metrics.md`: R≈8 temp-1 samples from BOTH sides, teacher-forced both ways, length-normalized; JS headline + both KL directions): **registered on the 16×16 shared-instance block only** (~120 unordered pairs; full-panel RB would cost tens of GPU-h for marginal coverage). Note #470's one notable directional win: bystander→source KL was the best per-source predictor for one source (ρ=+0.504).
- **Operational prerequisite:** the canonical RB implementation lives ONLY on unmerged branches (`predictor_jsdiv_470/phase3_sequence_js_kl.py` + `teacher_force_and_reduce_js_kl`, commit 3819b9c63; pinned `issue466_predictors.py` @ 281b5e0d79). Merge to main BEFORE P0 (documented gap: `docs/methodology/issue_532.md` §3.7).

**A6. Behavior-aware span metric:** fact-slice JS generalized to **taught-span JS** per behavior row (teacher-force the trained span under both contexts, per-position JS). On the fact arm it beat the entire #509 bake-off — it must be in the leaderboard or the geometric rows get unearned credit.

**A7. Anchor axis:** P1 clouds saved at all 3 anchors (end_of_system, last_prompt, mean_response) — mean_response is the canonical recipe-(b) anchor and #470's strongest single geometric cell; #468's newline-after-assistant + in-context-lit construction is carried as one extra anchor row if the cloud pass can capture it cheaply, else named as out-of-scope.

**A8. Labeled null-anchor rows (zero-marginal only):** cosine/JS-to-assistant, cosine/JS-to-neutral, cosine-to-trained-midpoint — historically null (#396/#415/#311); included as dead-baseline rows since they're free from clouds. `kl_judge` (#404) and in-context-rate M_3 (#404) and first-step gradient (#396): SKIP (cost without expectation).

## B. Registered combination set (everything else exploratory)

Principle: combine ACROSS the rule's four slots (geometry / source strength / readout prior / data prior), never within a slot (within-slot metrics are collinear; #493: 320 predictors within 0.02 CV R²). Registered rows:

1. **Nested ladder** (per behavior + pooled, held-out ΔR² per step): cosine → +norm ratio → +whitening → +bystander prior → +training-completion prior.
2. **Rule-form row** (parametric, multiplicative): G_pred = whitened-projection × source implant strength, read through the softmax at the target's base prior — scored against the unconstrained linear combination of the same features (constrained ≈ free ⇒ mechanism evidence).
3. **Geometry swap test:** ladder re-run with `js_out_seq`/`kl_out_seq` replacing the activation metric in slot 1 (answers "activation vs output space" as a slot substitution).
4. **Source implant strength as standing covariate** in every combined model (#472 ρ≈0.95; v3 already records per-cell diagonal strength).
5. v3's behavior-blind ablation applies to every combined row; a full metric-redundancy correlation matrix ships instead of within-slot combinations.

## C. Layer policy

- **Primary (registered): L22** (continuity with #502; marker anchor row).
- **Secondary (registered): per-behavior layer selection INSIDE CV folds** (training folds pick the layer; held-out folds score it). Motivated by behavior-dependent layer profiles: EM signal at L6 (#487), sycophancy early-to-mid (#509), marker L19–24 (#502).
- **Exploratory (one appendix row): layer-profile ridge** over the geometry metric computed at every layer, nested CV. Theory note: LoRA writes at all layers, so the rank-1 picture's true predictor aggregates per-layer projections; the ridge is its pre-training approximation. Never headlines (#493 caution).
- **Requirement:** P1 saves activation clouds at ALL layers × all 3 anchors (~3 GB; same forward passes).

## D. Judging-spec upgrades (land BEFORE the P0 freeze — the freeze locks judge prompts)

Core finding that reframes v3's assumption: for a rate judge, Δ_observed = (sensitivity+specificity−1) × Δ_true — judge error MULTIPLIES deltas and never cancels in trained−base; if judge accuracy varies by surface format (the dominant judge bias, 0.76–0.92 across judges incl. Claude), deltas are differentially attenuated across exactly the format-varying eval columns #537 predicts over. (arXiv 2511.21140, 2601.05420, 2604.23178.)

MUST: (1) per-format-family confusion-matrix calibration of all 4 judges (~200 gold-labeled responses per format family per row; Rogan–Gladen-correct rates, or verify Youden's J flat across families); (2) format-counterfactual paired re-judging (same content re-wrapped plain↔JSON↔code↔markdown; flip-rate = pure judge format bias; replaces the 20-verdict eyeball audit; CALM method, arXiv 2410.02736); (3) normalize-before-judging (unwrap JSON / strip fences) for fact+sycophancy+refusal, explicitly NOT for EM (format-conditioned misalignment is a real effect, arXiv 2511.20104); (4) EM row: per-cell excluded-fraction reporting (coherence filter is a condition-dependent selection effect), alignment-threshold sweep {20,30,40}, mean score as secondary DV; (5) human gold set ~150–200/row (doubles as the MUST-1 calibration set) breaking the Haiku-vs-Sonnet same-family circularity, per-class precision/recall not just aggregate κ.

SHOULD (near-free): known-answer canaries per batch + judge snapshot pinning; confidence-based Haiku→Sonnet escalation (arXiv 2407.18370); k=3 self-consistency on the EM row (or a 10% subsample to quantify single-sample noise); one-off five-way option-order permutation check on the fact judge; rubric-with-worked-examples for the categorical Haiku judges (version the prompt v2 + re-run calibration, report both κs — no silent swap); per-verdict length/format metadata + clustered SEs by question id (arXiv 2411.00640).

SKIP (with reasons recorded): position-swap calibration (pairwise-only), panel-of-judges (Claude-judging-Qwen = lowest-risk cross-family quadrant), judge swaps to fine-tuned open judges (breaks pre-registration), 0–100→0–5 rescale on the EM row (replication fidelity dominates; threshold sweep covers it).

Cost: ~8–10K extra judge calls (~6% of volume), ~$50–120, ~600–800 human labels, analysis code.

## E. Consolidation actions (task hygiene)

- #524 stopped + merged into #537 phase 1 (directive posted on #524, 2026-06-09); its v6 stats machinery is already imported by v3 §6. Archive after v4 approval.
- #510 (rank-1 predictor) → absorbed by A1. #499 (P(training data | context)) → absorbed by A2. #512 (framing) → merges into #526 (theory). Archive all three after v4 approval.
- #526 remains the theory task: rank-1 derivation, nested-ladder logic, lit grounding (artifact: `tasks/proposed/526/artifacts/related_work_lit_scan.md`), zero-GPU retrodiction on existing #502/#489 matrices.

## G. Rank-1 mechanism tests (registered; derived from the v_b = M·v_c framing)

The leaderboard rows are scalar-leakage regressions; these two test the rank-1 update form itself.

**G1. ΔG_anti vs context-norm differences (zero GPU).** The rank-1 patch predicts a specific GEOMETRIC antisymmetry signature: `leak(i→j)/leak(j→i) = ‖v_j‖²/‖v_i‖²`, i.e. at matched implant strength the antisymmetric component in log space equals `2(log‖v_j‖ − log‖v_i‖)` — a quantitative slope prediction, not just a sign. Registered row: regress the per-behavior ΔG_anti (16×16 block, seed-split noise-corrected) on context-norm differences, with the strength-difference read (s_i − s_j; #524-v6 machinery) kept alongside as the competing explanation. Connects the norm-ratio ladder rung to the antisymmetric machinery; norms come from the P1 clouds.

**G2. Activation-delta parallelism (marker row registered ~0 GPU; judge rows +~4–5 GPU-h, exploratory-but-run).** v_b = M·v_c + rank-1 update predicts the behavior-direction change at EVERY target context is the SAME direction (v_b″ − v_b′), scaled by the projection coefficient — so trained−base residual deltas at the readout slot should be mutually parallel across eval contexts, with magnitudes ∝ the projection coefficient. NOTE the base clouds do NOT cover this — it needs TRAINED-model activations: (i) marker row: add a hidden-state hook to the existing Stage-1 (base) and Stage-2 (per-adapter) cross-eval forwards, dumping the residual vector at the post-response slot for layer subset {6, 14, 22, 27} (~1–2 GB; no new forwards). Registered reads: pairwise cosine of Δh(c_j) across the 28 eval contexts (parallelism), and ‖Δh(c_j)‖ vs projection coefficient (scaling). (ii) judge rows: one small batched teacher-forced HF pass per adapter immediately after its existing eval model-load (slot = first response token / taught-span first token), ~+4–5 GPU-h across 136 adapters; same reads, labeled exploratory (readout position is construct-cleanest for marker/fact). A parallelism FAILURE with a working scalar ladder = the leakage rule works for non-rank-1 reasons — that distinction is the mechanism deliverable.

## F. Budget delta summary

| Item | GPU-h | API/$ | Other |
|---|---|---|---|
| A1/A3/A7/A8 (from clouds) | 0 | — | storage ~3 GB |
| A2 training-completion prior | ~3 | — | — |
| A4 first-token cache | ~0.5–1 | — | — |
| A5 cheap tier (full panel) | ~2–4 | — | — |
| A5 canonical RB (16×16 block) | ~5–8 | — | branch merge first |
| A6 taught-span JS | ~1 | — | — |
| D judging upgrades | 0 | ~$50–120 | ~600–800 human labels |
| G1 ΔG_anti vs norm-diff | 0 | — | harness code only |
| G2 parallelism (marker via hooks / judge rows post-load TF) | ~0 + 4–5 | — | ~1–2 GB activation deltas |
| **Total** | **~16–22** | **~$80–180** | |

New v4 total estimate: ~190 GPU-h central (band ~155–280). Still parks at the >100 GPU-h gate by design.
