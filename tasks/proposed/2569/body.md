---
title: 'Linear-map theory battery: operator eigenstructure, null-space fibers, theory-bridge
  gates, SAE wiring, last-token SAE map, weight-update rank, shift regression, and
  a cross-model operator atlas'
kind: experiment
tags: []
created_at: '2026-08-25T06:45:51Z'
has_clean_result: false
parent_id: 1774
origin_prompt: 'User-approved 8-leg battery from the 2026-08-24 one-at-a-time theory
  walk: eigen/fixed-point analysis, gate-metric ladder (''ok this works''), wiring
  matrix (''sounds good''), last-prompt-token SAE + feature-to-feature map, weight-update
  rank (''let''s look at the rank of the update''), denoised shift regression, operator
  atlas + cross-model three-tier (''Sounds good''), null-space fibers + monitor certificates
  (''yes add it''). Filing batch answers: one umbrella task child of #1774, spawn
  one autonomous session, Llama-3.1-8B-Instruct for the atlas capture, no Overleaf
  writes.'
workflow: v1
goal: 'Exploit the linearity of the fitted residual context-answer mapping (banked
  963k-row layer-19 ridge map, Qwen2.5-7B-Instruct) to characterize in one phased
  battery: (1) operator eigenstructure, effective null space, fixed point, and their
  SAE-feature reads (ignored/copied/transcoded/damped anatomy); (2) whether the map''s
  Gram matrix WtW is the correct theory-paper context-gate metric and whether the
  closed-form ridge learning curve reproduces the empirical C1 scaling; (3) the feature-to-feature
  wiring of the map across SAE dictionaries including a trained last-prompt-token
  SAE; (4) the literal weight-update rank (dW spectra, intruder dimensions, factor
  alignment) across the recent organism fleet; (5) denoised reduced-rank structure
  of #1979''s banked write maps; (6) an aligned operator atlas with a cross-model
  (Llama-3.1-8B-Instruct) three-tier representation-vs-operator similarity report;
  (7) correlational validation of the effective null space via kernel-equivalent context
  pairs, plus closed-form monitor decision geometry and sensitivity certificates.'
---
# Linear-map theory battery: operator eigenstructure, null-space fibers, theory-bridge gates, SAE wiring, last-token SAE map, weight-update rank, shift regression, and a cross-model operator atlas

## Goal

Exploit the linearity of the fitted residual context-answer mapping (banked 963k-row layer-19 ridge map, Qwen2.5-7B-Instruct) to characterize in one phased battery: (1) operator eigenstructure, effective null space, fixed point, and their SAE-feature reads (ignored/copied/transcoded/damped anatomy); (2) whether the map's Gram matrix WtW is the correct theory-paper context-gate metric and whether the closed-form ridge learning curve reproduces the empirical C1 scaling; (3) the feature-to-feature wiring of the map across SAE dictionaries including a trained last-prompt-token SAE; (4) the literal weight-update rank (dW spectra, intruder dimensions, factor alignment) across the recent organism fleet; (5) denoised reduced-rank structure of #1979's banked write maps; (6) an aligned operator atlas with a cross-model (Llama-3.1-8B-Instruct) three-tier representation-vs-operator similarity report; (7) correlational validation of the effective null space via kernel-equivalent context pairs, plus closed-form monitor decision geometry and sensitivity certificates.

## Decision record (clarify gate, 2026-08-24 — answers are the user's, from the one-at-a-time discussion walk + the filing batch)

- Packaging: ONE umbrella task carrying all 8 legs, phased; child of #1774; parks once, promotes once (user-answer, filing batch).
- Launch: spawn one autonomous session for the whole task immediately after filing (user-answer).
- Atlas cross-model leg: Llama-3.1-8B-Instruct — cross-family at matched scale; one paired same-text capture round is the task's only mandatory GPU capture phase (user-answer).
- Theory-paper writes: NONE — results stay in the task body; the user integrates into the Overleaf theory paper himself (user-answer: "No, task bodies only").
- Each leg below was individually discussed and approved in the walk (user-answers quoted in ## Provenance).
- Explicitly PARKED, out of scope for this task (user): layer-to-layer predictive attribution lattice; turn-dynamics state-space system (A, B) — new direction requiring /deep-lit-review first; null-space CAUSAL steering rerun at corrected dose; fixed-point/drift multi-turn design; Der et al. nested turn+token SAE variant.
- Inherited/rule-pinned (not re-asked): everything stays LINEAR (linear-by-default rule; no MLP legs anywhere); any newly fitted map reports identity+learned-bias baseline + kNN retrieval (mapping-baselines rule); no GCV λ selection at n_train < d (#1887); group-level held-out folds where R²/ρ is quoted; judge = claude-sonnet-4-5-20250929 for any judged read (minimal judging in this task); runpod-first auto compute lane.

## Design sketch (for the planner — 8 legs, phased; 0-GPU legs first)

### Leg 1 — Operator eigenstructure, functional anatomy, fixed point (0 GPU, banked)

- Object: the banked 963k-row ridge map, weights local at `data/issue_2094/joint_transport/banked_maps/issue779_monitoring/n1m_readout/weights/L19/` (L14 + L26 as robustness replicates); constant offset convention b = ymu − W·xmu (standardized ridge).
- Full eigendecomposition (complex pairs kept) + biorthogonal expansion W = Σ λᵢ vᵢ uᵢᵀ: left eigenvector uᵢ = read direction, right vᵢ = write direction. "Maintained" requires BOTH real λ ≈ 1 AND cos(u, v) ≈ 1. Singular values reported alongside eigenvalues everywhere (regression-shrinkage caveat: λ conflates gain with predictability; σ is the honest amplification read).
- Two-sided SAE dashboards per top direction: read side named against the per-token layer-19 andyrdt dictionary (131,072 features, Neuronpedia labels `19-resid-post-aa`/trainer_1); write side against the #2476 turn-averaged matryoshka SAE decoder PLUS an encoder-pass third column (push the write direction through the #2476 encoder and read which features fire). Grain-matching rule: per-token dictionary for v_C-side reads, turn-averaged for v_A-side (#2476 bridge arm showed mismatch inverts tier profiles).
- Dual metric for nearest-feature reads: raw canonical basis AND side-matched whitened; every cosine quoted against the max-over-dictionary null floor (~0.08 for 131k features).
- Functional anatomy: classify context directions into ignored (effective kernel), copied (λ≈1, u≈v), transcoded (read strongly, written elsewhere), damped — with mass fractions per class.
- Fixed point x* = (I − W)⁻¹ b, guarded by the spectral-radius check (ρ = 0.910 < 1, #1774); nearest banked answer vectors to x* + its turn-averaged SAE decode. (#1774 found ‖x*‖ ≈ 45.0 ≈ answer-pool median.)
- CCA / mutual-information estimate between v_C and v_A through the map; W ≈ αI + rank-k structure test; ridge-posterior spectrum error bars (eigenvalue uncertainty under the fit's posterior).
- Phase-2 (pod, deferred within the task, runs only if phase-1 reads warrant): corpus-scale top-activating example retrieval for top eigen-directions + full ~1M-pool fixed-point nearest neighbors.

### Leg 2 — Theory bridge: gate-metric ladder + closed-form learning curve (0 GPU, banked)

- Gate-metric ladder for the theory predictor's context gate g_C: predict banked per-organism/per-context leakage (#1979 + #2474 arms) under similarity metrics ordered I → diag(Σ_c)⁻¹ → Σ_c⁻¹ → WᵀW → rank-truncated WᵀW → WᵀΣ_a⁻¹W. The algebraic anchor: through-map similarity = cᵀ(WᵀW)c′, i.e. the A10 gate metric IS the map's Gram matrix — this leg tests whether that resolves why the Σ⁻¹-whitened gate failed (it upweights rare directions over used ones).
- Closed-form ridge learning-curve prediction vs the empirical C1 scaling curve (R²/acc@1 vs n up to 963k).
- Results land in the task body ONLY (no Overleaf edits — user decision).

### Leg 3 — Wiring matrix + behavior receipts + prompt-attribution demo (0 GPU, existing dictionaries)

- T = E_turnavg · W · D_pertoken (mixed-grain: per-token decoder D on the context side, #2476 turn-averaged encoder E on the answer side): sparse feature→feature wiring reads; top in/out edges per behavior-relevant feature.
- Behavior-contraction receipts: for each trait/behavior direction, which context features drive which answer features through T.
- Optional Jacobian-agreement column (does T's edge structure agree with local Jacobian attributions?) — carried with the #1776 caveat: the map is a correlate, not the mechanism (Jacobian R² −0.001 vs 0.681); all wiring claims are map-level.
- Predictive prompt-attribution demo: attribute a predicted answer feature back through W to context positions/features on a handful of worked examples — the in-house analogue of Der et al.'s attribution graphs, at whole-context grain.

### Leg 4 — Last-prompt-token (context-side) SAE + feature→feature map (GPU, ~6–12 h)

- Train a context-vector SAE on the banked last-prompt-token v_C states (963k rows, layer 19), mirroring the #2476 recipe for comparability (matryoshka BatchTopK, k=100, width 65,536; k=200 twin optional if cheap).
- Fit the feature→feature map from last-prompt-token SAE activations to #2476 answer-SAE activations. Metrics (the full set discussed): per-feature held-out R²; firing AUROC + conditional-magnitude accuracy (predict WHETHER a feature fires and HOW MUCH given it fires); precision/recall@k on active-set prediction; Der-protocol evaluation on predicted feature lists (10-way matching / coverage) — reuse #2552's judged descriptions when that task parks, else run a bounded own auto-interp round with the pinned project judge.
- Mapping-baselines rule applies: identity+bias inapplicable across dictionaries (state it), kNN retrieval reported; shuffle nulls.

### Leg 5 — Weight-update rank: literal ΔW analysis (pod; download-heavy, CPU-dominant)

- The gap this fills: NO literal weight-matrix diff (θ_post − θ_base) analysis exists anywhere in the project (#667 was an activation panel; #1947 was per-row answer-shift stacks; #1979 prefix-grain shifts).
- LoRA organisms: effective rank within the rank budget (stable rank, participation ratio of singular values), intruder-dimension analysis per arXiv 2410.21228 (LoRA grows intruder dims; full FT spreads evenly).
- Full-FT organisms: ΔW spectra per weight matrix (the #1979 fleet's 18 banked checkpoints).
- Factor alignment: top ΔW singular vectors vs the behavior direction δ, readout r_B, context direction c_C, and gate directions.
- Fleet: recent organisms including #2379/#2474 inoculation organisms, #1979's checkpoints, #1947 arms. Downloads route to a pod (>10 GB rule), never the shared VM.

### Leg 6 — Denoised reduced-rank shift regression on #1979's write maps (0 GPU, banked)

- On the banked 16,400-row matched-base-text shift corpora + fitted write maps (36,400 cross-arm training rows): reduced-rank regression with split-half denoising; identify factors shared across arms vs arm-specific.
- Anchors to the existing rank record: on-policy rank-one top-1 share 0.14–0.19 (refuted), matched-text 0.595 vs the 0.6 criterion (borderline), #1979 prefix-grain ~0.43 — this leg asks what the DENOISED rank actually is and what the factors are.

### Leg 7 — Operator atlas + cross-model three-tier leg (0 GPU except one capture round)

- Embed all banked operators (#1979 write maps, #2474 arms, the n1m readout at L14/19/26, #2378's operators when that task parks, and this task's own leg-4 map) via aligned distances. Two mandatory corrections: anchor alignment (Procrustes to shared anchors before comparing) and split-half noise floors (else the atlas maps sample sizes, not operators). Similarity statistics follow the `scripts/issue1345_operator_comparison.py` conventions; every statistic states direction-aware vs spectrum/rotation-invariant-only.
- Cross-model leg (Llama-3.1-8B-Instruct — user choice): ONE paired same-text capture round (shared eval texts through both models, v_C + v_A captured at matched relative depth) — the task's only mandatory GPU capture phase. Then the three-tier report: tier 1 representation alignability per side (alignment-fit R² / CKA, per v_C and v_A separately); tier 2 operator similarity under fixed data-pinned alignments; tier 3 the decomposition (how much cross-model operator difference is representation misalignment vs genuine operator difference). Plus the shared-vs-side-specific correspondence test that determines whether cross-model eigenvalue comparison is even well-posed. Matched-capacity nulls throughout.

### Leg 8 — Null-space fibers + monitor robustness certificates (0 GPU, banked)

- Effective kernel = bottom singular subspace below a stated variance threshold (ridge has no exact kernel; same convention as #1774's validated-channel counts). All claims phrased "directions the map reads at < X% of typical gain."
- Kernel-pair validation (the headline of this leg — the correlational null-space test): mine the banked corpus for real context pairs far apart in context space whose difference lies mostly in the effective kernel (the map predicts near-identical answer states); check whether their REALIZED answers actually match, vs distance-matched control pairs. Zero steering, zero generation.
- Monitor decision geometry: for each C5 behavior readout r, the direction Wᵀr is the minimal context-space change that flips the predicted behavior; name it via the context-side SAE. Least-norm pre-images of target behavior levels, with the coset ambiguity stated (every pre-image is particular solution + kernel).
- Robustness certificates: worst-case score movement ε·‖Wᵀr‖ per monitor under bounded activation perturbation, compared across the monitor family (context probe vs mapped-answer probe vs persona-vector projection).
- Two binding caveats carried in every claim: (a) activation-space perturbations are not established to correspond to realizable text perturbations — this ships as sensitivity analysis and decision-geometry characterization, NEVER a security guarantee; (b) all "the map cannot distinguish" claims are map-level (#1776), and the kernel-pair validation is precisely the test of the stronger reading.

### Compute + sequencing

- Legs 1, 2, 3, 6, 8: 0 GPU, banked artifacts, VM/cpu pods; vectorized fits per the many-cell rules.
- Leg 4: ~6–12 GPU-h (SAE training + fits, per the #2476 measured recipe).
- Leg 7 capture round: ~4–8 GPU-h (paired capture on the shared eval-text subset, both models).
- Leg 5: pod-routed downloads + CPU spectra (cpu-bigmem or smallest GPU intent if co-located).
- Leg 1 phase-2 (corpus-scale examples): deferred, runs only if phase-1 warrants.
- Total estimate: ~10–20 GPU-h + CPU-pod time.

## Provenance / lineage

- Parent #1774: operator characterization on the same map — high-rank (763–2,932 validated channels), stable non-normal contraction (ρ 0.910, eigenvector condition number 4,519), traits rotated-and-shrunk (gain 0.60–0.64), fixed-point norm 45.0, null-space causal test VOIDED (under-dosed) — this task's legs 1 + 8 are its direct continuation.
- #1979: 16,400-row matched-base-text shift corpora, fitted write maps, 18 fleet checkpoints (legs 5–7 inputs).
- #2474 / #2379: recent organism fleet incl. inoculation organisms (legs 2, 5).
- #2476: in-house turn-averaged matryoshka SAEs (legs 1, 3, 4).
- #2552 (running): Der et al. replication + feature descriptions — soft dependency for leg 4's Der-protocol eval (reuse when parked; bounded own round otherwise).
- #1947 / #667: the prior rank-of-write reads this battery's leg 5 completes at weight-matrix grain.
- #1776: Jacobian correlate-not-cause — the map-level caveat binding legs 3 + 8.
- #2378 (interpreting): story-character transfer operators — atlas ingestion when parked.
- User approvals: each leg approved individually in the 2026-08-24 discussion walk ("let's look at the rank of the update"; "ok this works" [gate ladder]; "sounds good" [wiring]; "okay, let's do the last prompt token SAE, and then do the mapping from last prompt token SAE to answer vector SAE"; "Sounds good" [atlas + cross-model three-tier]; "yes add it" [fibers + certificates]; packaging/launch/model/paper-writes from the filing AskUserQuestion batch).
