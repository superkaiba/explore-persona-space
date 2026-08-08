# Analyses of the context→answer linear map W — methods survey (2026-07-06)

Chat-session lit survey (4 parallel agents: dynamical systems/control, Markov/transfer-operator/
spectral theory, statistics of estimated linear operators, LLM-interp prior art) on the question:
*we have a ridge-fit linear map W from context activations to answer activations (residual stream →
residual stream, L14, Qwen-2.5-7B, d=3584; #779/#813/#1073 line) — what are the ways to analyze it?*
Full agent reports are appended verbatim as Appendices A–D. All arXiv ids were agent-verified via the
arXiv MCP at search time (two exceptions flagged inline: Elhage et al. Transformer Circuits Thread and
Millidge & Black AF post have no arXiv id).

## Synthesis

### What W is (two exact identities, one affordance)

1. **W is exactly the exact-DMD / Koopman-EDMD operator with a linear dictionary.** The ridge fit
   `W = Yᵀ-side · (XXᵀ+λI)⁻¹`-style snapshot-pair regression is the operator DMD estimates, so the
   whole DMD toolkit transfers: mode amplitudes, residual-certified spectra (ResDMD, incl. the
   n<d dual form, 2403.05891), de-biased/bagged spectra with CIs (BOP-DMD 2107.10878, TDMD 1502.03854),
   and Procrustes-constrained fits (piDMD 2112.04307).
2. **W is an estimated conditional-expectation operator** (linear conditional mean embedding,
   x ↦ E[y|x]; Grünewälder 1205.4656, Mollenhauer & Koltai 2012.12917). The transfer-operator
   learning literature (Kostic et al. 2205.14027, 2302.02004) gives sharp spectral rates, shows
   ridge/EDMD has larger spectral bias than reduced-rank regression, and explains where spurious
   eigenvalues come from.
3. **Affordance:** input and output live in the SAME residual-stream space ⇒ W is an endomorphism ⇒
   eigen-analysis is licensed (the copying-detector read of Elhage et al.'s OV eigenvalues is the
   direct precedent). None of the inter-layer-lens papers have this; it is a genuine edge.

### Validity gates that come FIRST (all four agents converged on these)

- **Mechanical rank cap:** rank(W) ≤ n−1. At the averaged grain (n=50) every raw rank read is
  uninformative; d/n ≈ 72 puts most singular directions below the BBP detectability transition
  (math/0403022) — direction-level claims belong on the per-example grain (n≈2500).
- **λ sets much of the spectrum:** ridge filter factors σ²/(σ²+λ) shrink AND flatten. Report
  effective degrees of freedom df(λ)=Σσᵢ²/(σᵢ²+λ) next to every spectrum; re-run headlines at
  ~3 λ values; note GCV degenerates at n≈d (#779's observed failure).
- **Nulls are non-optional:** (a) pairing-permutation null — shuffle the (context, answer) row
  pairing, REFIT ridge (same λ, standardization inside the loop), recompute every statistic;
  (b) random-subspace null for any projection/overlap read (random k-dim subspaces of R^3584
  overlap a lot by chance); (c) Marchenko–Pastur / simulated noise edge for spectra
  (Martin & Mahoney 1810.01075; RMT cross-covariance cleaning Benaych-Georges et al. 1901.05543).
- **Eigen-reads are gated:** W is generically non-normal. Before interpreting eigenvalues, run
  (i) normality gap ‖WᵀW−WWᵀ‖, (ii) Σ-metric symmetric/antisymmetric split (reversibility index),
  (iii) pseudospectra / eigenvector condition number, (iv) ResDMD residual per eigenpair.
  Fail → report singular values/subspaces only.
- **Metric hygiene:** state the inner product. Two-sided whitening (Σy^{-1/2} W Σx^{1/2}) turns
  singular values into canonical correlations in [0,1] (comparable across layers/arms) but breaks
  the endomorphism; a single shared whitening preserves the endomorphism for eigen-reads.

### The menu, organized by question

**Q1 — What KIND of operator is it (identity / scalar / rotation / projection / symmetric)?**
Constrained-Procrustes fits over matrix manifolds (piDMD): solve min‖Y−WX‖ over
{αI, orthogonal, symmetric, skew, rank-r}; the held-out residual gap vs unconstrained ridge is one
comparable number per hypothesis. Companions: polar decomposition W=QP (rotation × stretch),
sym/antisym split in the Σ-metric, trace(W)/d (mean copying score), ‖W−I‖ and the spectrum of W−I
(near-identity / iterative-inference test), and the **identity baseline** (does W beat y≈x? —
the DMD-along-depth null, 2605.07556).

**Q2 — How many channels (the honest rank)?**
Primary: **held-out per-direction predictable-variance spectrum** (# of canonical/PLS components
with out-of-sample R²>0, group-level folds per ood-generalization-folds rule) — robust to λ and to
the Kornblith d≫n impossibility (1905.00414). Formal: reduced-rank regression + rank-selection
(RSC, Bunea et al. 1004.2995; CV rank; Bura–Cook). Soft: stable rank, spectral-entropy erank,
participation ratio — always as curves over λ with the null overlay. VAMP-2 score per layer/arm as
a cross-validatable "where is the map richest" selector (Wu & Noé line, 1904.07752).

**Q3 — WHICH directions (the projection/rank-of-projection asks)?**
- **Restricted maps** P_out·W·P_in for P ∈ {context-PCA, behavior span B, answer-PCA}: captured
  energy fraction ‖P_out W P_in‖_F²/‖W‖_F², operator norm, effective rank of the restriction —
  each vs the random-subspace + permutation nulls.
- **Trait gain matrix** G = U_Bᵀ W V_B (p×p, behavior basis in and out): diagonal = self-transfer
  of each trait, off-diagonal = cross-trait leakage — a literal pre-fine-tuning trait-transfer
  table, the most decision-relevant single artifact for the leakage-prediction goal
  (Observable Propagation coupling coefficients, 2312.16291). Orthonormalize / use the causal
  inner product first (persona vectors are not orthogonal).
- **Near-eigenvector test:** is W·v_trait ∝ v_trait (trait preserved) vs rotated vs killed?
  And does W map the context-side trait shift onto the answer-side trait direction?
- **Principal angles** between top singular subspaces of W and {B, context-PCA, answer-PCA};
  between prefix-arm and context-arm maps.
- **Nullspace read:** smallest right-singular directions = context geometry the answer discards;
  LEACE-erase (2306.03819) a trait direction from x and measure the answer-side drop (causal
  load-bearing test).

**Q4 — Eigen-structure (endomorphism-only, run AFTER the Q-gates pass)?**
Copying detector: fraction of eigenvalue mass with positive real part (Elhage OV precedent);
|λ|≈1 modes = persistent/integrating axes (line-attractor analogy, Maheswaranathan 1906.10720);
|λ|≈0 = forgotten context; complex pairs = rotational re-expression. W^k limit object (dominant
invariant subspace vs collapse; Geshkovski 2305.05465 as interpretive frame). Every retained
eigenpair residual-certified (ResDMD) + CI'd (BOP-DMD).

**Q5 — Causal / behavioral validation (from prediction to mechanism)?**
Port the LRE protocol wholesale (Hernandez et al. 2308.09124 — the closest published analogue:
fitted linear residual→residual relation maps, analyzed for faithfulness, LOW RANK, and causality):
(a) faithfulness = held-out cosine/R² of Wx vs y against predict-the-mean AND additive
task-vector baselines (Hendel 2310.15916, Todd 2310.15213); (b) causality = rank-reduced
pseudo-inverse steering — compute Δx = W⁺Δy, inject, measure on-policy judge-rate + log-P dual DV;
(c) stitching/transplant — inject Wx as the actual answer-position residual and measure behavior
recovery (Bansal 2106.07682). χ²-contraction ceiling: top centered whitened singular value² =
linear maximal correlation = upper bound on context→answer information transfer per layer
(Polyanskiy & Wu 1508.06025, Makur & Zheng 1510.01844).

**Q6 — Grains and arms?**
Averaged W = population operator; per-example Ws = a state-dependent operator FIELD — the spread
of per-example spectra around the averaged spectrum measures how much the single-operator
abstraction loses (nonlinearity/state-dependence); cluster per-example maps (Jacobian-switching
LDS view, 2111.01256). Prefix-arm vs context-arm: principal angles / Procrustes residual between
the two maps — small angle ⇒ the query adds little to the transfer operator.

### Must-reads (closest prior art)

- Hernandez et al., *Linearity of Relation Decoding in Transformer LMs* — https://arxiv.org/abs/2308.09124
- Aswani & Jabari, *DMD along Depth in Vision Transformers* — https://arxiv.org/abs/2605.07556
- Kostic et al., *Sharp Spectral Rates for Koopman Operator Learning* — https://arxiv.org/abs/2302.02004
- Golden, *Equivalent Linear Mappings of LLMs* (detached Jacobian; Qwen-validated) — https://arxiv.org/abs/2505.24293
- Benaych-Georges, Bouchaud, Potters, *Optimal cleaning for singular values of cross-covariance matrices* — https://arxiv.org/abs/1901.05543
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (OV eigenvalue copying read) — https://transformer-circuits.pub/2021/framework/index.html
- Colbrook, *ResDMD with fewer snapshots than dictionary size* (the n<d regime) — https://arxiv.org/abs/2403.05891

### Suggested first battery (0-GPU, on existing #779/#813 stores)

1. Validity layer: permutation-null + λ-sweep + df(λ) for the existing L14 spectra (extends
   `scripts/issue813_rank_spectrum.py`, which already computes factored spectra).
2. Q1 structure tests: distances to {αI, orthogonal, symmetric, rank-r} + identity baseline,
   held-out.
3. Q2 honest rank: held-out predictable-variance spectrum (group folds), both grains, both arms.
4. Q3 trait table: G = U_Bᵀ W V_B over the persona-vector dictionary + restricted-map energy
   fractions vs nulls.
5. Q4/Q5 only after 1–3: gated eigen-read + LRE-style faithfulness/causality on the per-example
   grain.

---

# Appendix A — Dynamical systems & control (agent report, verbatim)

# Dynamical-systems & control-theory framings for the ridge-estimated endomorphism W (context → answer, layer ~14, Qwen-2.5-7B)

Framing note used throughout: your ridge fit **W = Y Xᵀ(XXᵀ + λI)⁻¹** *is* the regularized least-squares one-step propagator between snapshot pairs — i.e. exactly the operator that exact-DMD and data-driven Koopman methods estimate. Because x (context activations) and y (answer activations) live in the same residual-stream space, W is an **endomorphism**, so eigen-analysis (not only SVD) is meaningful and the whole propagator/Koopman/control toolkit transfers. Two caveats recur and are load-bearing at your n≈d / n<d, λ≈1e3 regime, so I state them once and reference per technique: **(R1) ridge shrinkage** pulls singular values and eigenvalue moduli toward 0 (biased-low spectral radius/rank, biased-in eigenvalues); **(R2) partial identification** — with n<d, W is pinned only on the ≤n-dim row space of X (span of your contexts); its action off that subspace is set by λ, not data, so invariant-subspace / kernel claims must be restricted to the data-spanned subspace.

## Techniques / framings

### 1. Eigendecomposition of the one-step propagator (spectral radius, stable/unstable modes, W^k, fixed points)
**Computes:** eigenpairs (λᵢ, vᵢ) of W; classifies each mode as amplified (|λ|>1), preserved/integrating (|λ|≈1), or killed/forgotten (|λ|≈0); spectral radius ρ(W); behavior of iterated W^k and its fixed points/dominant invariant subspace.
**Recipe on W:** `eig(W)`; sort by |λ|. Modes with |λ|≈1 are directions the context→answer map carries over ~unchanged (candidate "persistent semantic axes"); |λ|≈0 modes are context content the answer discards; |λ|>1 modes are amplified. Iterate: does W^k converge to a rank-1/low-rank object (a single dominant eigenvector = a "leader"/collapse direction) or preserve a multi-dimensional invariant subspace? A right eigenvector with real λ≈1 near a data point ≈ an approximate fixed point.
**Refs:** Tu et al., *On DMD: Theory & Applications*, arXiv **1312.0041** (DMD = eigendecomposition of the best-fit linear operator); the value-matrix-spectrum→limit-object result of Geshkovski et al. **2305.05465** gives a theory for what W^k does.
**Pitfalls:** eigenvalues are the WRONG summary if W is non-normal (→ #3); (R1) shrinks ρ(W) and can spuriously push modes below |λ|=1; (R2) restrict invariant-subspace claims to span(X).

### 2. Exact-DMD / Koopman-operator identification (W as the snapshot-pair operator)
**Computes:** the finite-dimensional Koopman/DMD approximation A = Y X⁺ (your ridge W is its Tikhonov-regularized form) and its DMD modes, eigenvalues, mode amplitudes.
**Recipe on W:** treat every (context, answer) pair as a snapshot pair. DMD modes = eigenvectors of W; DMD amplitudes = projection of data onto them, ranking each mode by answer-set variance carried. For **per-example vs averaged**: the averaged map is a single Koopman operator; per-example maps are a *sample of operators* — analyze their spectral spread (a distribution over eigenvalues), not one spectrum. For **prefix vs context** arms, fit two operators and compare spectra/modes directly.
**Refs:** Tu et al. **1312.0041**; Williams, Kevrekidis & Rowley (EDMD), arXiv **1408.4408**; Korda & Mezić convergence, **1703.04680**; Schmid 2010 (*J. Fluid Mech.*, original DMD, textbook/venue). *(1312.0041 and 1408.4408 are among the sources flagged in the arxiv-mcp rule for embedded instruction-shaped text; cited on verified metadata only.)*
**Pitfalls:** "linear consistency"/rank-deficiency (Tu et al.) — at n<d the operator can trivially interpolate, so training-pair fit overstates the map; (R1)/(R2).

### 3. Non-normality: pseudospectra, departure-from-normality, numerical abscissa, transient growth
**Computes:** how badly eigenvalues mislead. Departure from normality ‖W*W − WW*‖; ε-pseudospectra Λ_ε(W)={z : ‖(zI−W)⁻¹‖≥1/ε}; numerical range/abscissa; transient-growth envelope ‖W^k‖ vs ρ(W)^k.
**Recipe on W:** a context→answer map is asymmetric ⇒ generically non-normal, so eigenvectors are non-orthogonal and ‖W^k‖ can grow (or singular values of W far exceed |λ|) even when ρ(W)<1. Compute ‖W^k‖ for k=1..K vs ρ(W)^k: a large gap means "spectrum lies; use SVD/pseudospectra to describe amplification." Overlay behavior/persona vectors on the pseudospectral map — a direction in a fat pseudospectral region is one where the map is highly sensitive.
**Refs:** Trefethen & Embree, *Spectra and Pseudospectra* (Princeton UP 2005, textbook); Fish & Bollt, non-normality in directed networks, **2202.00156**; Mohammadi et al., transient growth from non-normal dynamics (NN-adjacent optimization operators), **2103.08017**; Symon et al., resolvent/non-normality amplification classification, **1712.05473**.
**Pitfalls:** pseudospectra are O(grid×SVD) — expensive at d=3584; use ARPACK/randomized resolvent-norm sampling. Non-normality measures are themselves ridge-sensitive (R1).

### 4. Residual DMD (ResDMD) — spectral verification with error control, incl. the n<d dual formulation
**Computes:** a data-driven residual per candidate eigenpair certifying whether it is a true spectral feature vs a spurious artifact; pseudospectra and spectral measures with convergence guarantees.
**Recipe on W:** for each (λ, v) from #1, compute residual ‖(W − λI)v‖ / ‖v‖ in the appropriate inner product; keep only low-residual eigenpairs before interpreting any "invariant subspace / killed direction." Colbrook's **fewer-snapshots-than-dictionary** dual-least-squares variant is built for exactly your n<d regime (no train/quadrature split), and residual-based mode ordering beats |amplitude| ordering.
**Refs:** Colbrook, Ayton & Szőke, ResDMD, **2205.09779**; Colbrook, *ResDMD in the regime of fewer snapshots than dictionary size*, **2403.05891** (directly your n<d case); stochastic/variance version, **2308.10697**.
**Pitfalls:** residuals need a defined observable inner product (raw activation metric or a whitened one); still inherits (R1)/(R2) — a "verified" eigenpair is verified for the *regularized* operator on span(X).

### 5. Mode amplitudes + noise-robust / bagged DMD (optDMD, TDMD, BOP-DMD) for reliability at n≈d
**Computes:** dynamically-ranked modes with **uncertainty quantification** on eigenvalues/amplitudes, and de-biased eigenvalues that correct DMD's known noise bias.
**Recipe on W:** because least-squares/ridge DMD biases eigenvalues under snapshot noise, use total-DMD (TDMD) or optimized-DMD to de-bias, and **BOP-DMD** (bagging over resampled pairs) for confidence intervals on each λᵢ and mode. Report only spectral features whose CI excludes the artifact region — the direct antidote to "spectrum unreliable at low n."
**Refs:** Sashidhar & Kutz, BOP-DMD (UQ), **2107.10878**; Hemati et al., de-biasing/TDMD, **1502.03854**; Askham et al., optimized DMD, **1712.01883**.
**Pitfalls:** bagging assumes exchangeable pairs — fine for i.i.d. examples, questionable for per-example operators; UQ reflects sampling noise, not the λ-bias, so pair it with de-biasing (R1).

### 6. Structure-constrained / Procrustes DMD (piDMD) — a hypothesis test for "identity / projection / rotation / symmetric"
**Computes:** the best-fit W restricted to a matrix manifold — orthogonal (rotation), symmetric/self-adjoint, low-rank, shift/scale-equivariant, or ≈identity — each a closed-form Procrustes problem, plus the **residual gap** vs unconstrained W.
**Recipe on W:** solve `min_{W∈M} ‖Y − WX‖` for M ∈ {orthogonal, symmetric, skew, low-rank-r, α·I}. Compare each constrained residual to the unconstrained ridge residual: small gap for M=orthogonal ⇒ W is essentially a rotation (norm-preserving, information-conserving); small gap for M=symmetric ⇒ eigen-analysis trustworthy (normal operator); small gap for M=αI or low-rank-r ⇒ W ≈ scaled-identity or acts through an r-dim bottleneck. Turns the Goal's qualitative questions into quantitative, comparable numbers.
**Refs:** Baddoo et al., physics-informed DMD (piDMD as a Procrustes problem), **2112.04307**; invariant/consistent DMD constraints, **2312.08278**.
**Pitfalls:** constrained fits trade variance for bias — at n<d the *unconstrained* baseline overfits, so compare gaps on a held-out pair set.

### 7. Controllability/observability Gramians + balanced truncation (which input directions have persistent downstream influence)
**Computes:** for the discrete LTI system x_{k+1}=Wx_k, the finite-horizon Gramians and the balanced (jointly controllable+observable) subspace; a model-order reduction keeping directions that both receive and transmit signal under iterated W.
**Recipe on W:** one-step reachable set = column space / right singular vectors of W (context directions producing large answer-side responses); the finite-horizon controllability Gramian Σ_k W^k(W^k)* ranks directions by **persistent** influence across repeated application; balanced truncation identifies the low-dim subspace where the propagator "lives." Cross-Gramian gives a single-shot controllability∩observability read for square W. Compare the dominant balanced subspace to your persona/behavior directions.
**Refs:** Himpe & Ohlberger, cross-Gramian model reduction, **1606.03954**; Kramer & Willcox, balanced truncation for lifted nonlinear systems, **1907.12084**; nonlinear balanced truncation, **2604.23044**.
**Pitfalls:** Gramian sums assume ρ(W)<1 (else infinite-horizon Gramian diverges) — use finite-horizon or a shifted/scaled W; interpret "input directions" as context axes, not literal prompt controls.

### 8. Input–output amplification: leading singular vectors, resolvent, data-driven optimal-growth directions
**Computes:** the context directions that most excite the answer (top right singular vectors → their paired left singular vectors as responses), and — via the resolvent (zI−W)⁻¹ — mode-selective amplification for non-normal W. A **data-driven** variant computes optimal input→output directions directly from the (x,y) pairs without forming W.
**Recipe on W:** `svd(W)`: top right singular vectors = maximally-amplified context directions, paired left singular vectors = the answer directions they map to; singular-value spectrum quantifies rank/energy concentration. Test whether a behavior vector v is a high-gain input (large ‖Wv‖/‖v‖) or attenuated. The 2507.02525 approach gets the same optimal-amplification directions straight from data pairs — attractive at n<d because it sidesteps forming/regularizing W.
**Refs:** Symon et al., resolvent & non-normality, **1712.05473**; Kai, Frame & Towne, *Data-Driven Transient Growth Analysis*, **2507.02525**.
**Pitfalls:** SVD directions are basis-dependent under input standardization — de-standardize before interpreting; resolvent-norm sweeps expensive at d=3584.

### 9. Transformers as interacting particle systems / clustering dynamics (Geshkovski–Rigollet line)
**Computes:** a rigorous theory of what iterating an attention-type propagator does — tokens (particles on a sphere) cluster in "time" (depth), with the **type of limiting object governed by the spectrum of the value matrix**; metastability (long-lived multi-cluster states) before collapse.
**Recipe on W:** interpretive frame for W^k (#1): does your empirical propagator's spectrum predict representation collapse (→ single cluster) vs preserved multi-cluster structure? The value-matrix-spectrum result gives concrete predictions to test against W's measured eigenvalues; metastability says a near-collapse operator can still hold structure for many steps.
**Refs:** Geshkovski et al., *Emergence of clusters in self-attention*, **2305.05465**; *A mathematical perspective on Transformers*, **2312.10794**; Rigollet, *Mean-field dynamics of Transformers*, **2512.01868**; metastability, **2410.06833**; hardmax clustering & "leaders", **2407.01602**.
**Pitfalls:** the theory concerns the network's *own* value/attention matrices under idealized (sphere, time-invariant) dynamics — your W is a data-estimated cross-representation map, so this is a rigorous **analogy**, not an identity; don't import quantitative rates.

### 10. Residual networks / neural ODEs as discrete dynamical systems (iterative-inference & stability)
**Computes:** the residual stream x_{ℓ+1}=x_ℓ+f(x_ℓ) as an Euler step of a flow; stability/contraction of the linearized flow-map (Jacobian eigenvalues near +1 = preservation/integration, <1 = contraction).
**Recipe on W:** view the layer-14 context→answer map as one flow-map step and ask whether W behaves like `I + (small)` (near-identity refinement, the "iterative inference" picture) — measure ‖W − I‖ and the spectrum of W−I. Eigenvalues of W near +1 ↔ near-conserved directions; reconciles with #1 and #6 (identity-closeness) from the ResNet-ODE side.
**Refs:** Marion et al., implicit regularization of ResNets toward neural ODEs, **2309.01213**; Chang et al., multi-level ResNets from a dynamical-systems view, **1710.10348**; Cont et al., asymptotic analysis of deep ResNets, **2212.08199**; Haber & Ruthotto, *Stable architectures for deep neural networks* (Inverse Problems 2017, venue).
**Pitfalls:** your W is a *cross-representation* regression, not literal depth propagation of one token — the Euler/ODE identification is loose; use it for the near-identity test, not for claiming a continuous-depth flow.

### 11. Fixed-point / linearization reverse-engineering (Sussillo–Barak) + Jacobian switching LDS
**Computes:** fixed points of a (nonlinear) map and the local linearized dynamics around them — revealing line attractors, slow/marginal modes (|λ|≈1), and low-dim interpretable structure.
**Recipe on W:** W is already a *global* linearization, so its eigenstructure IS the local dynamics: eigenvectors with real λ≈1 are approximate line-attractor / integrating axes (directions the map accumulates rather than forgets) — directly interpretable as persistent semantic/persona axes; complex λ near the unit circle = slow rotations. For the **per-example** variant, the family of W's is a Jacobian-switching linear dynamical system — cluster the per-example Jacobians to find regimes. The sentiment-RNN precedent is a close NLP analogy (a task-relevant line attractor emerged and was human-interpretable).
**Refs:** Maheswaranathan et al., *line attractor dynamics in sentiment RNNs*, **1906.10720**; Smith, Linderman & Sussillo, Jacobian switching LDS, **2111.01256**; Rivkind & Barak, local dynamics in trained RNNs, **1511.05222**; Sussillo & Barak, *Opening the black box* (Neural Computation 2013, venue).
**Pitfalls:** a global linear W collapses genuinely state-dependent structure into one Jacobian — validate with the per-example ensemble; "line attractor" language requires λ≈1 confirmed after de-biasing (#5) and residual-checking (#4).

### 12. Equivalent-linear-operator / detached-Jacobian analysis — relating W to behavior/persona directions & steering
**Computes:** LLM inference as an input-dependent linear operator A(x)·x per token, whose low-dimensional singular vectors decode to semantic concepts and act as **steering operators**.
**Recipe on W:** treat your regression W as a data-averaged cousin of the exact detached-Jacobian operator and test the operator↔behavior-direction relationship the Goal wants: (a) cosine of each persona/behavior vector v with W's top left/right singular vectors and eigenvectors; (b) is v an approximate eigenvector of W (behavior-preserving) or rotated/killed?; (c) does W map the *context-side* shift for a trait onto the *answer-side* trait direction (Wv_context ≈ v_answer)? — a direct causal-propagation read. Connects the spectral picture to your mean-difference trait vectors and to steering.
**Refs:** Golden, *Equivalent Linear Mappings of Large Language Models* (detached Jacobian, per-token linear operators, low-dim semantic singular vectors as steering operators; validated up to Qwen-3-14B), **2505.24293**.
**Pitfalls:** the exact detached Jacobian is per-token and per-input; your averaged W blurs that — differences between W and the local Jacobian are informative, not noise. (R2): alignment meaningful only for v with support in span(X).

### 13. DMD-along-depth in transformers — the direct methodological precedent (+ Koopman-on-activations siblings)
**Computes:** an autonomous linear operator K fit from consecutive hidden-state pairs across depth, with K^p used to predict p steps ahead; systematic study of the regularization/rank/calibration budget needed for a *stable* fit, and where linearity holds vs fails.
**Recipe on W:** essentially your experiment done along depth in ViTs — reuse its protocol: sweep λ and calibration-set size, measure fitted-operator **rank** (it found early operators compress to rank ≪ d), track cosine of K^p·x vs the true endpoint map, and check whether an **identity baseline** becomes competitive (a critical null — if I predicts the answer as well as W, W's structure isn't doing work). Its finding that local linear fidelity does *not* transfer downstream is a direct warning for your interpretation. Siblings apply the same lens to LLM embedding trajectories (low-rank spectra; hallucination signature) and to residual snapshots (near-unit spectral mass diagnostic).
**Refs:** Aswani & Jabari, *DMD along Depth in Vision Transformers*, **2605.07556** (closest twin); Aswani et al., NN layers as linear ops via Koopman/DMD, **2409.01308**; Sugishita et al., Koopman replaces nonlinear layers, **2402.11740**; Akrout, *Representations Matter: DMD embedding modes of LLMs*, **2309.01245**; Kim et al., *Residual Koopman Spectral Profiling*, **2602.22988**.
**Pitfalls:** these fit *depth-autonomous* K (same operator reused across layers); your W is a single cross-layer map, so borrow the diagnostics (rank, identity-baseline, calibration budget) not the autonomy assumption.

## Closest prior work (verified ids)

- **2605.07556** — Aswani & Jabari, *DMD along Depth in Vision Transformers*. The direct methodological twin: fit K from consecutive hidden-state pairs, predict K^p, study rank/regularization/calibration + the identity baseline.
- **2505.24293** — Golden, *Equivalent Linear Mappings of Large Language Models*. Per-token input-dependent linear operators A(x)·x; low-dim semantic singular vectors as steering operators (Qwen-3 tested) — the operator↔behavior-direction bridge.
- **2602.22988** — Kim et al., *Residual Koopman Spectral Profiling*. Whitened DMD on layer-wise residual snapshots; "near-unit spectral mass" diagnostic; validated on LLaMA-2/GPT-2.
- **2309.01245** — Akrout, *Representations Matter*. DMD on LLM sentence-embedding trajectories; consistently low-rank spectra; mode-count signature of hallucination.
- **2403.05891** — Colbrook, *ResDMD in the regime of fewer snapshots than dictionary size*. Residual-verified spectra via dual least-squares — your exact n<d regime.
- **2305.05465** — Geshkovski et al., *The emergence of clusters in self-attention dynamics*. Limiting cluster geometry governed by the value-matrix spectrum — theory for what W^k does.
- **2112.04307** — Baddoo et al., *physics-informed DMD (piDMD)*. Procrustes-constrained operators (orthogonal/symmetric/low-rank) — the identity/rotation/projection hypothesis tests.
- **2409.01308** — Aswani et al., *Representing NN Layers as Linear Operations via Koopman*. Replacing a nonlinear layer by a finite linear operator; DMD-eigenvalue vs SVD analysis.
- **1312.0041** — Tu et al., *On DMD: Theory & Applications*. Foundational: W = eigendecomposition of the best-fit linear operator; linear-consistency pitfalls for rank-deficient data.
- **2107.10878** — Sashidhar & Kutz, *BOP-DMD*. Bagged/optimized DMD with UQ on eigenvalues — reliability at n≈d.

## Top-3 priority picks for your setting

1. **Non-normality lens first (pseudospectra + ‖W^k‖ envelope + optimal-amplification SVD/resolvent; Trefethen–Embree textbook, 2103.08017, 2507.02525, 1712.05473).** A context→answer map is asymmetric and therefore almost certainly non-normal, so the eigenvalue story from #1 will mislead about amplification and about how W acts on behavior directions. Establishing departure-from-normality up front tells you whether to trust eigen-analysis or switch to SVD/pseudospectral descriptions — and the optimal input→output directions are the cleanest link from W to your persona vectors.

2. **Reliability layer: ResDMD n<d dual verification + BOP-DMD UQ (2403.05891, 2107.10878, 1502.03854).** At n≈d/n<d with heavy ridge (λ≈1e3), your spectrum is partly determined by the regularizer, not the data (R1/R2). Residual-certify every eigenpair and put CIs on eigenvalues *before* interpreting any invariant/killed subspace — otherwise "rank," "spectral radius," and "|λ|≈1 axes" are artifacts of λ and n.

3. **piDMD/Procrustes structure tests (2112.04307).** They convert the Goal's central qualitative questions — is W close to identity / projection / rotation / symmetric? — into one comparable number each (the held-out residual gap of the nearest constrained operator vs unconstrained W). Most direct quantitative route to "characterize W's structure," and composes cleanly with #1 (a small orthogonal-fit gap would explain a near-unit-modulus, norm-preserving spectrum) and with the DMD-along-depth **identity-baseline** null from 2605.07556.

Must-read precedent before starting: **2605.07556** (DMD-along-depth in ViTs) is the closest existing recipe to your exact experiment — reuse its rank/calibration/identity-baseline diagnostics.

All arXiv ids above were returned with resolving abstracts via the arXiv MCP. Textbook/venue-only refs (Trefethen & Embree 2005; Schmid 2010 JFM; Sussillo & Barak 2013 Neural Comp.; Haber & Ruthotto 2017 Inverse Problems) are cited without ids by design.

---

# Appendix B — Markov chains / transfer operators / spectral theory (agent report, verbatim)

# Operator-theoretic framings for W: the context→answer map as an estimated conditional-expectation / transfer operator

**Orienting fact that organizes everything below.** With the linear feature map φ(x)=x, ridge regression gives Ŵ = Ĉ_yx (Ĉ_xx + λI)⁻¹ where Ĉ_yx = (1/n)Σ yᵢxᵢᵀ and Ĉ_xx = (1/n)Σ xᵢxᵢᵀ. So Ŵ is **exactly the linear-dictionary special case of EDMD / the empirical conditional mean embedding**: the population target is the L²-orthogonal projection of the conditional-expectation operator P f(x)=E[f(y)|x] onto the span of linear observables. Every technique below reads structure out of that estimated operator, and every one hinges on **which inner product you use** — the honest metric here is the covariance (whitened) inner product ⟨u,v⟩_Σ = uᵀΣv, not raw Euclidean, because that is the metric in which P is an L²(μ) operator and in which singular values become canonical correlations in [0,1]. Two whitening choices recur and must be kept distinct: (i) **two-sided** W̃ = Ĉ_yy^{-1/2} Ĉ_yx Ĉ_xx^{-1/2} for SVD/CCA reads (breaks the endomorphism — different left/right spaces, correct for singular values); (ii) **single shared** whitening by one Σ (e.g. Ĉ_xx, or pooled (Ĉ_xx+Ĉ_yy)/2) to preserve the endomorphism for eigen-analysis and reversibility tests.

## (1) Techniques and framings

**1. Conditional mean embedding / conditional-expectation operator (the master framing).**
*Computes:* the object Ŵ actually estimates — the linear least-squares approximation to x↦E[y|x]; population form C_yx C_xx⁻¹, ridge = Tikhonov on the ill-conditioned Ĉ_xx.
*Recipe for W:* nothing new to run; the payoff is knowing (a) the limiting object depends on λ (ridge shifts the target, not just the estimate), (b) CME theory gives the right error metric (HS/operator norm in the RKHS = your Frobenius-in-Σ-metric), (c) misspecification is expected because linear features are a coarse dictionary.
*Outcomes:* low effective rank of Ŵ ⇒ E[y|x] concentrates on few context directions; strong λ-sensitivity of the spectrum ⇒ you are reading the regularizer, not the map.
*Refs:* Grünewälder et al. arXiv:1205.4656 (CME = vector-valued regression); Mollenhauer & Koltai arXiv:2012.12917 (conditional-expectation operator, which limiting object the kernel estimate converges to, HS-approximation even when P is non-compact); Li, Meunier, Mollenhauer, Gretton arXiv:2208.01711 (optimal rates, misspecified regime).
*Pitfalls:* ridge shrinks Ŵ toward 0 (biases singular values down and unevenly); "E[y|x]" is only linear-in-x here, so nonlinearity leaks into residuals and per-example variation.

**2. EDMD / Galerkin-projection view — Ŵ is the compression P_V K P_V of the true operator onto linear observables.**
*Computes:* the best linear-dictionary surrogate; EDMD provably converges to the L²(μ)-orthogonal projection of the operator onto the dictionary span (here span{coordinates}).
*Recipe:* treat every eigenvalue/eigenvector of Ŵ as belonging to a **projected** operator; enrich the dictionary (quadratics, a few random features) to test whether spectral features are dictionary-stable.
*Outcomes:* eigenvalues stable as you enlarge the dictionary ⇒ intrinsic; moving ⇒ truncation artifacts.
*Refs:* Korda & Mezić arXiv:1703.04680 (EDMD→projected Koopman); Klus et al. review arXiv:1703.10112; Mezić arXiv:2010.05377 (extends Koopman to **static maps between different spaces** — directly licenses "context→answer" as an operator even though it is one-shot, not a time-iterate).
*Pitfalls:* spectral pollution from truncation (see #12); the "iterate Wᵏ" reading is *not* iterating the model — W is a static map, so powers describe the operator's own geometry, not multi-turn dynamics.

**3. Reduced-rank regression (RRR) + EDMD-vs-RRR spectral-bias comparison.**
*Computes:* rank-r risk-optimal operator (project onto top whitened cross-covariance subspace); a lower-bias spectral estimator than full ridge.
*Recipe:* fit Ŵ at several explicit ranks r and compare eigenvalues of ridge-Ŵ, RRR-Ŵ, and plain reduced-rank; eigenvalues agreeing across estimators are signal, those that move are artifacts. Report the "metric distortion" of estimated eigenvectors (departure from Σ-orthonormality) as a reliability flag.
*Outcomes:* identifies the trustworthy part of the spectrum; RRR typically resolves leading modes with less bias than EDMD/ridge.
*Refs:* Kostic, Novelli, Maurer, Ciliberto, Rosasco, Pontil arXiv:2205.14027 (RRR for transfer/Koopman operators in RKHS); Kostic, Lounici, Novelli, Pontil arXiv:2302.02004 (**sharp spectral rates: EDMD has larger bias than RRR, similar variance; explains spurious eigenvalues; introduces the eigenfunction metric-distortion functional**).
*Pitfalls:* rank choice interacts with λ; at n≈d both inherit covariance-estimation noise (see #13).

**4. Whitened operator / "the right metric" — singular values as canonical correlations in [0,1].**
*Computes:* W̃ = Ĉ_yy^{-1/2} Ĉ_yx Ĉ_xx^{-1/2}; its singular values are the **canonical correlations** between context and answer reps — the fraction of an answer direction linearly predictable from context — bounded, comparable across layers/arms, free of the arbitrary scaling that makes raw eigenvalues of Ŵ uninterpretable.
*Recipe:* always report the whitened SVD alongside any raw-metric read; use it as the canonical descriptor of context→answer coupling and its directions.
*Outcomes:* σ near 1 = a context direction almost deterministically fixes an answer direction; σ near 0 = decoupled.
*Refs:* Klus et al. arXiv:1703.10112 (unifies TICA/DMD/VAMP as whitened cross-covariance SVDs); Noé & Clementi arXiv:1506.06259 (kinetic-map scaling of whitened components).
*Pitfalls:* Ĉ_xx⁻¹, Ĉ_yy⁻¹ are the noisy objects at n≈d — whiten with a shrinkage/RIE estimate (see #13); two-sided whitening breaks the endomorphism (fine for SVD, wrong for eigenvalues).

**5. VAMP / VAMP-score & time-lagged CCA — SVD (not eigendecomposition) is the well-posed primitive for a non-self-adjoint map.**
*Computes:* for a general (non-reversible) operator the **singular functions** are the well-defined objects; the VAMP-r score = Σ_k σ_kʳ of the whitened cross-covariance measures how much predictable structure exists and is a cross-validatable selection criterion (layer, rank, λ).
*Recipe:* compute the VAMP-2 score of Ŵ per layer/arm to pick where the map is richest and choose rank without a free parameter; the top singular *directions* are the maximally-correlated context/answer axes to compare against persona/behavior vectors.
*Outcomes:* VAMP-2 peaking at a layer = that layer carries the most linearly-transferable context signal; a sharp drop after k singular values = effective rank k.
*Refs:* Klus, Husic, Mollenhauer, Noé arXiv:1904.07752 (kernel CCA = VAMP-score optimum = kernel transfer-operator SVD); Wu, Nüske, Paul, Klus, Koltai, Noé arXiv:1610.06773 (variational Koopman, model selection); Mardt, Pasquali, Wu, Noé arXiv:1710.06012 (VAMPnets, deep). Original VAMP: Wu & Noé, "Variational approach for learning Markov processes", *J. Nonlinear Sci.* 2020 (method used in the above).
*Pitfalls:* VAMP warns eigen-reads are meaningful only once near-reversibility (#6) is established; otherwise use singular values.

**6. Reversibility / self-adjointness test in the Σ-metric (detailed-balance analogue).**
*Computes:* whether Ŵ is self-adjoint w.r.t. ⟨,⟩_Σ, i.e. whether ΣŴ is symmetric. Self-adjoint ⇒ real spectrum, Σ-orthogonal eigenbasis, no oscillatory/rotational component ("equilibrium-like"). Asymmetry ⇒ complex-conjugate eigenpairs = a genuine rotational/cyclic component in how context is transformed into the answer.
*Recipe:* split M=ΣŴ into symmetric S and antisymmetric A; report ‖A‖/‖S‖ (Σ-metric) as an irreversibility index, and the magnitude/phase of complex eigenpairs as the size of the rotational part.
*Outcomes:* small ‖A‖/‖S‖ + real spectrum ⇒ W is essentially a symmetric "smoothing" of context into answer; large ⇒ directional/cyclic structure that CCA-symmetric summaries miss.
*Refs:* Paul, Wu, Vossel, de Groot, Noé arXiv:1811.12551 (TICA valid only under detailed balance; VAMP for non-equilibrium); Wu et al. arXiv:1610.06773 (reversible Koopman models / enforcing reversibility); Devergne, Kostic, Parrinello, Pontil arXiv:2406.09028 (time-reversal-invariant generator learning).
*Pitfalls:* finite-sample asymmetry is nonzero even for a truly reversible map — calibrate ‖A‖/‖S‖ against a bootstrap/label-permutation null before calling W "irreversible."

**7. Spectral gap → metastable / almost-invariant *subspaces* (PCCA+-style macro-structure).**
*Computes:* a cluster of eigenvalues near the leading one, with a gap below, signals near-block structure; the sign/soft-membership structure of the top-k (whitened) eigenvectors partitions context space into k "almost-invariant" macro-directions the map preserves coherently.
*Recipe:* eigendecompose Ŵ in the shared-whitening metric, find the k with the largest gap, run PCCA+ / soft-sign clustering on the top-k eigenvectors, check whether recovered macro-subspaces align with persona/behavior vectors.
*Outcomes:* recovered macro-subspaces = candidate persona/trait subspaces the map keeps invariant.
*Refs:* Klus et al. arXiv:1703.10112; Froyland et al. arXiv:2407.07278 (almost-invariant/coherent sets from transfer operators); Froyland, Murray, Stancevic arXiv:1012.2149 (second eigenfunction = split/escape); Klus & Bramburger arXiv:2507.18147 (transfer-operator spectral clustering, reversible reconstruction). Method origin: Deuflhard & Weber, "Robust Perron cluster analysis (PCCA+)", *Lin. Alg. Appl.* 2005 (off-arXiv).
*Pitfalls:* "metastable state" is a *time-iterated Markov* notion; here the precise claim is "almost-invariant **subspace of the one-shot map**," not "long-lived state." Don't import dwell-time language.

**8. Spectral radius / second eigenvalue — "mixing vs memory."**
*Computes:* for a stochastic operator, gap 1−|λ₂| is the mixing rate; for Ŵ the leading singular/eigenvalue measures how much context is *preserved* into the answer vs washed out.
*Recipe:* read the leading whitened singular value as a preservation/"memory" score and the spectrum decay as a "mixing/forgetting" profile across layers.
*Outcomes:* spectrum concentrated near 0 = the map forgets context (contractive/mixing); mass near 1 = context strongly determines answer (memory).
*Refs:* Makur & Zheng arXiv:1510.01844 (χ²-contraction = second singular value of the whitened transition operator); Polyanskiy & Wu arXiv:1508.06025.
*Pitfalls:* Ŵ is not row-stochastic and has no guaranteed λ=1 mode; interpret via singular values/contraction (#9), not by assuming a stationary eigenvalue.

**9. Contraction coefficients (Dobrushin / χ²) & data-processing ceiling — "how much context information survives into the answer."**
*Computes:* the χ²-contraction coefficient of the channel x→y equals the squared top *non-trivial* singular value of the whitened conditional-expectation operator = the (linear) **maximal correlation / HGR** between context and answer; it upper-bounds how much *any* downstream functional of the answer can depend on context.
*Recipe:* take the top centered whitened singular value ρ₁ of Ŵ; η = ρ₁² is a metric-free [0,1] ceiling on context→answer information transfer — compare across layers to localize where context influence peaks; corresponding singular vectors are the HGR maximal-correlation directions.
*Outcomes:* an operational, interpretable coupling measure readable as "at most this fraction of answer variation is context-driven at this layer."
*Refs:* Polyanskiy & Wu arXiv:1508.06025 (SDPIs, Dobrushin, χ² coefficient); Makur & Zheng arXiv:1510.01844 (χ² = maximal correlation; ordering of f-divergence coefficients); Asoodeh, Diaz, Calmon arXiv:2001.06546 (contraction-coefficient machinery). Foundational HGR/maximal-correlation is classical (Rényi 1959, off-arXiv).
*Pitfalls:* linear features give the *linear* maximal correlation, a lower bound on the true χ² coefficient; state it as such.

**10. Non-normality & pseudospectra — why eigenvalues can be misleading for a ridge-fit Ŵ.**
*Computes:* Ŵ is generically non-normal (ŴŴᵀ≠ŴᵀŴ), so eigenvalues are ill-conditioned and don't govern finite-power amplification; the ε-pseudospectrum {z: ‖(zI−Ŵ)⁻¹‖ ≥ 1/ε} shows how far eigenvalues can move under O(ε) perturbations — and ε is exactly your estimation-noise scale.
*Recipe:* compute the resolvent-norm surface on a grid (Σ-metric) and/or the eigenvector condition number κ(V); fat pseudospectra / large κ ⇒ individual eigenvalues are noise-dominated, report subspaces and singular values instead.
*Outcomes:* tight pseudospectra ⇒ eigenvalues trustworthy; fat ⇒ don't interpret eigenvalues one-by-one.
*Refs:* Trefethen & Embree, *Spectra and Pseudospectra* (Princeton 2005 — canonical, off-arXiv); Embree & Keeler arXiv:1601.00044 (pseudospectra in a physically-relevant norm; pencils); Fish & Bollt arXiv:2202.00156 (non-normality + pseudospectra for **directed networks** — a close ML analogue); Boullé, Colbrook, Conradie arXiv:2506.15782 (convergent pseudospectra for Koopman on RKHS with error control).
*Pitfalls:* pseudospectra must be computed in the same metric you interpret W in; raw-Euclidean pseudospectra of a whitened operator are meaningless.

**11. Numerical range / field of values — convex, estimation-robust spectral enclosure.**
*Computes:* W(Ŵ)={u*Ŵu: ‖u‖_Σ=1}, a convex set containing the spectrum whose barycenter is trace/d (mean of eigenvalues) and whose size vs the eigenvalue hull quantifies non-normality; its rightmost point bounds initial amplification.
*Recipe:* use the field of values as a stable summary that does not require trusting individual eigenvalues — a robust "where does W live" descriptor and non-normality gauge.
*Outcomes:* numerical range ≈ eigenvalue hull ⇒ near-normal, eigen-reads safe; much larger ⇒ non-normal, prefer SVD/pseudospectra.
*Refs:* Bögli & Marletta arXiv:1909.01301 (essential numerical range for pencils, spectral-pollution enclosure); general FoV/Toeplitz–Hausdorff theory (classical).
*Pitfalls:* convexity means it over-encloses (won't resolve fine structure); a screening tool, not a substitute for #3/#12.

**12. Spectral-pollution control via residuals (ResDMD) — filter spurious eigenvalues from real ones.**
*Computes:* an infinite-dimensional-consistent residual for each candidate eigenpair (λ,v) certifying it as genuine vs a truncation/finite-sample artifact; the dual-least-squares form handles n≲d directly.
*Recipe:* for each eigenpair of Ŵ compute ‖Ŵv−λv‖_Σ / ‖v‖_Σ (or the ResDMD residual on held-out data); keep only small-residual eigenpairs, confirm stability under a λ- and n-sweep.
*Outcomes:* a concrete pass/fail per eigenvalue — the most actionable defense against the ridge-artifact concern.
*Refs:* Herwig, Colbrook, Junge, Koltai, Slipantschuk arXiv:2507.16915 (**residual-based no-pollution spectra for transfer / Perron–Frobenius operators**, incl. protein folding; also shows spectral features can arise even when eigenfunctions leave the chosen space — a subtlety for "the true spectrum"); Colbrook arXiv:2403.05891 (ResDMD with **fewer snapshots than dictionary size**, dual least-squares — the n≈d regime); Davies & Plum arXiv:math/0302145 and Lewin & Séré arXiv:0812.2153 (classical spectral pollution).
*Pitfalls:* residuals need a genuinely held-out quadrature set or the dual form; reusing the fit data understates pollution.

**13. Low-rank + spiked structure: RMT cleaning of the cross-covariance (the n≈d de-biasing correction).**
*Computes:* W_pop = C_yx C_xx⁻¹ is built from covariances corrupted by sample noise (Marchenko–Pastur bulk + a few spikes); empirical singular values of the cross-covariance are systematically biased and singular vectors rotated. Optimal rotationally-invariant "cleaning" corrects the singular values and the **noise edge gives a principled threshold for the true number of slow modes**.
*Recipe:* replace plain ridge with optimal singular-value shrinkage on the whitened cross-covariance, then reconstruct Ŵ; count singular values above the RMT noise edge = operational rank; small rank ⇒ Ŵ is effectively a projection onto a few slow modes (answers domain (f) directly).
*Outcomes:* de-biased spectrum + a defensible effective dimension of the map, robust to the arbitrary λ.
*Refs:* Benaych-Georges, Bouchaud, Potters arXiv:1901.05543 (**optimal cleaning of singular values of the cross-covariance E[XYᵀ]** — exactly the C_yx object, RIE-optimal in Frobenius norm at finite n/d); Su & Wu arXiv:2207.03466 (eOptShrink for **colored/separable noise**, relevant since activations are strongly correlated); Bongiorno & Lamrani arXiv:2310.01963 (information lost by covariance cleaning, KL vs Frobenius).
*Pitfalls:* RIE assumes proportional asymptotics; at n≈50 you are below where guarantees are clean — treat the noise-edge rank as an estimate and bootstrap it.

**14. Averaged vs per-example maps: population operator vs a state-dependent operator field.**
*Computes:* the AVERAGED W is the population conditional-expectation operator; PER-EXAMPLE maps are a locally-linearized (state-dependent) operator field, and the spread of per-example operators around the average measures the true operator's non-linearity/non-normality.
*Recipe:* compare eigenvalue/singular-value clouds of per-example Ŵ(x) against the averaged Ŵ; small spread ⇒ the linear operator is a faithful global summary; large spread ⇒ the averaged eigen-structure is a mean of heterogeneous local maps, read cautiously.
*Outcomes:* quantifies how much the "single operator" abstraction is losing.
*Refs:* Mezić arXiv:2010.05377 (Koopman for static maps / representation eigenproblems, geometry of level sets); Lin, Tian, Perez, Livescu arXiv:2205.05135 (regression-based projection operators — linear vs nonlinear regression as the projection).
*Pitfalls:* per-example maps at n=1 output are extremely noisy; regularize heavily and interpret only aggregate statistics of the field.

## (2) Closest prior work (verified arXiv ids)

- **Mollenhauer & Koltai — Nonparametric approximation of conditional expectation operators.** arXiv:2012.12917
- **Grünewälder et al. — Conditional mean embeddings as regressors.** arXiv:1205.4656
- **Kostic et al. — Koopman Operator Regression in RKHS.** arXiv:2205.14027
- **Kostic et al. — Sharp Spectral Rates for Koopman Operator Learning.** arXiv:2302.02004
- **Klus et al. — Data-driven model reduction and transfer operator approximation.** arXiv:1703.10112
- **Klus, Husic, Mollenhauer, Noé — Kernel methods for detecting coherent structures.** arXiv:1904.07752
- **Herwig et al. — Avoiding spectral pollution for transfer operators using residuals.** arXiv:2507.16915
- **Colbrook — ResDMD with fewer snapshots than dictionary size.** arXiv:2403.05891
- **Benaych-Georges, Bouchaud, Potters — Optimal cleaning for singular values of cross-covariance matrices.** arXiv:1901.05543
- **Polyanskiy & Wu — Strong data-processing inequalities.** arXiv:1508.06025

(Supporting: Korda & Mezić arXiv:1703.04680; Mezić arXiv:2010.05377; Paul, Wu, Noé arXiv:1811.12551; Wu et al. arXiv:1610.06773; Makur & Zheng arXiv:1510.01844; Fish & Bollt arXiv:2202.00156; Boullé, Colbrook, Conradie arXiv:2506.15782; Deeptime library arXiv:2110.15013.)

## (3) Top-3 priority picks for your setting

1. **Whitened SVD + VAMP-score as the primary read, not eigendecomposition (#4–#5).** W is not self-adjoint, so its singular functions — canonical-correlation directions in [0,1] — are the well-posed, cross-layer-comparable primitives; eigenvalues are the fragile object, reported only after the reversibility test.

2. **ResDMD residual filtering + RRR-vs-EDMD spectral comparison + RMT cross-covariance cleaning (#12, #3, #13).** Jointly neutralize ridge shrinkage, n≈d noise, and spurious eigenvalues; the RMT noise-edge count answers "is W effectively a low-rank projection?"

3. **Σ-metric reversibility index + non-normality diagnostics (#6, #10).** The most interpretation-licensing test: tells you whether real-spectrum metastable-subspace language is admissible or W carries a genuine rotational component — and whether any eigenvalue is estimation-noise-dominated.

---

# Appendix C — Statistics / numerical linear algebra (agent report, verbatim)

# Statistics & numerical linear algebra of the estimated map W

**Setup used throughout.** Stack paired examples as rows: `X ∈ ℝ^{n×d}` (context or prefix mean-activations, columns standardized), `Y ∈ ℝ^{n×d}` (answer mean-activations), `d = 3584`. Ridge fit (dual): `Ŵ = Xᵀ(XXᵀ + λI_n)⁻¹Y`, a `d×d` endomorphism, `λ≈1e3`. Input covariance `Σ_x = XᵀX/n`; behavior/persona vectors `b_k` span subspace `B` (columns of `V_B ∈ ℝ^{d×p}`); context-PCA basis `U_C`, answer-PCA basis `U_A`.

**The one bound that dominates every read.** `Ŵ` lies in the row space of `X`, so `rank(Ŵ) ≤ rank(X) ≤ n−1`. At the averaged grain (`n=50`) that is **≤ 49, mechanically, for any λ**. So the *raw* rank of `Ŵ` or of any restriction `P W P` is uninformative — it just reports `min(subspace dims, n−1)`. Every technique below must report an **effective/predictable-rank** or an **energy fraction against a matched null**, never a bare rank count. Ridge with large λ additionally applies filter factors `f_i = σ_i²/(σ_i²+λ)` in the whitened SVD, shrinking and *flattening* the spectrum — pushing every naive effective-rank estimate upward (toward "full/high rank"). These two artifacts (n−1 cap; λ-flattening) are the recurring pitfalls, flagged per technique.

## (1) Techniques

### T1. Reduced-rank regression (RRR) + formal rank selection
**Computes:** the statistically-supported rank `r̂` of the coefficient matrix — the number of independent context→answer channels the data justifies, vs the mechanical `n−1`.
**Recipe:** RRR estimator `Ŵ_r = Ŵ_OLS · P_r`, `P_r` projecting onto the top-`r` right-singular directions of the fitted `ŶᵀŶ` in the `Σ_x` metric (= top-`r` CCA directions, T3). Select `r` by: (i) **Rank Selection Criterion (RSC)** — minimize `‖Y−XW_r‖_F² + μ·r·(n+d)` (Bunea–She–Wegkamp); (ii) **Bartlett/LR sequential test** on the smallest `d−r` canonical correlations (Anderson 1951); (iii) **Bura–Cook weighted-χ² rank test**; (iv) **CV rank** — pick `r` minimizing held-out `‖Y−XW_r‖`; (v) nuclear-norm path + read the elbow.
**Outcome meaning:** `r̂ ≈ p` (# behavior directions) → context→answer map is a low-rank readout aligned with persona axes. `r̂` large / ≈ `n−1` → no low-rank structure resolvable at this `n` (need more examples, not a stronger claim).
**Refs:** Bunea, She, Wegkamp (arXiv:1004.2995); Anderson 1951 (Ann. Math. Statist.); Izenman 1975 (JMVA); Reinsel & Velu 1998 (Springer); Bura & Cook 2003 (JMVA); Wen, Wang, Jiang *StARS-RRR* (arXiv:2207.00924).
**Pitfalls:** classic Bartlett LR assumes `n ≫ d`, breaks at `n≈d` — use RSC/CV. Ridge and RRR are different shrinkage; do the rank read on the CCA/predictable-variance spectrum (T3), not the ridge SVD. `r̂ ≤ n−1` always.

### T2. Singular-value spectrum + effective-rank measures of the (whitened) map
**Computes:** a soft, continuous "how many directions matter" from `Ŵ = UΣVᵀ`.
**Recipe:** three λ-report-alongside measures of `{σ_i}`: **stable/numerical rank** `sr = ‖W‖_F²/‖W‖₂² = Σσ_i²/σ_max²`; **spectral-entropy effective rank (erank)** `= exp(H)`, `p_i = σ_i/Σσ_j`, `H = −Σp_i ln p_i` (Roy–Vetterli); **participation ratio** `PR = (Σσ_i²)²/Σσ_i⁴`. Do it in the whitened output metric (`Σ_y^{-1/2} Ŵ`) so numbers reflect predictive channels, not answer-space anisotropy.
**Outcome meaning:** small `sr`/`erank`/`PR` (≈ a few) → map concentrates in a handful of directions (persona-controlled low-rank readout); values scaling with `n` → estimation noise.
**Refs:** Roy & Vetterli 2007 (EUSIPCO); Rudelson & Vershynin (arXiv:1301.2382); participation ratio — Gao et al. 2017 (bioRxiv); Jazayeri & Ostojic (arXiv:2107.04084).
**Pitfalls:** all three are **strongly λ-dependent** — large λ flattens `σ_i`, inflates erank/PR; sweep λ, report the curve. Uncentered `Y` puts a rank-1 mean spike in `σ_1` — center first. Report the whole spectrum overlaid with the null (T10), never one number.

### T3. CCA ↔ SVD-of-whitened-map + predictable-variance spectrum
**Computes:** canonical correlations `ρ_i` between context and answer spaces = singular values of the whitened regression operator; the estimation-robust "rank of the map."
**Recipe:** `M = Σ_x^{-1/2} Σ_xy Σ_y^{-1/2}` (regularize both covariances at λ); `SVD(M) = ρ_i`. Then the **predictable-variance spectrum**: per canonical (or PLS) component `k`, report **held-out** `R²_k` = variance in `Y` along direction `k` genuinely predictable from `X`.
**Outcome meaning:** # of `ρ_i` (or `R²_k`) clearly `>0` out-of-sample = the honest channel count. If only ~`p` components have held-out `R²>0` and align with `B`, the persona axes *are* the map.
**Refs:** Hotelling 1936; SVCCA (arXiv:1706.05806); projection-weighted CCA (arXiv:1806.05759); Kornblith et al. (arXiv:1905.00414) — **key warning:** any rotation-invariant statistic (incl. CCA) is meaningless when feature dim > n, exactly your `d=3584 ≫ n=50` regime.
**Pitfalls:** raw CCA at `d≫n` gives `ρ_i=1` spuriously; only the **held-out** `R²_k` version is trustworthy. Whitening with ill-conditioned `Σ_x` is unstable — use ridge-regularized covariances.

### T4. Restricted / projected map `P_out Ŵ P_in` — energy, rank, norm of the restriction
**Computes:** how much of the operator lives inside a chosen input×output subspace pair — the direct answer to "project onto context / behavior / answer subspaces and read the rank."
**Recipe:** pick input projector `P_in ∈ {P_C (context-PCA-k), P_B (behavior span), I}` and output projector `P_out ∈ {P_A (answer-PCA-k), P_B, I}`. Report per pair: **captured-energy fraction** `‖P_out Ŵ P_in‖_F² / ‖Ŵ‖_F²`; **operator norm** `‖P_out Ŵ P_in‖₂`; **effective rank of the restriction** (T2 on its SVD); and the small **block read** `G = V_Bᵀ Ŵ V_B` (a `p×p` matrix) — diagonal = each behavior direction mapping to itself, off-diagonal = cross-talk between persona axes. Scalar: `‖P_B Ŵ P_B‖_F² / ‖Ŵ‖_F²` = fraction of the map's action confined to persona space.
**Outcome meaning:** high behavior-space energy fraction + near-diagonal `G` → `Ŵ` acts as an (near-)diagonal gain on persona axes and little else. Diffuse energy → not persona-organized.
**Refs:** Golub & Van Loan (*Matrix Computations*); persona vectors (2507.21509) as applied anchor.
**Pitfalls:** `rank(P_out Ŵ P_in) ≤ min(dim in, dim out, n−1)` — mechanical; always divide by `‖Ŵ‖_F²` and compare to a **random-subspace null of the same dims** (T6). If `B` was estimated from the *same* activations, energy inside `B` is optimistically biased — estimate `B` on a disjoint split.

### T5. Principal angles / subspace overlap between input-driver, output-response, and behavior subspaces (with null)
**Computes:** geometric agreement between (a) the top right-singular subspace of `Ŵ`, (b) top left-singular subspace, (c) `B`, (d) context-PCA / answer-PCA subspaces.
**Recipe:** for orthonormal bases `Q_1,Q_2`, `SVD(Q_1ᵀQ_2)=cosθ_i` (Björck–Golub). Summaries: **mean cos²θ**, **Grassmann/chordal distance**. Ask "does the map read *from* context-PCA and write *into* answer-PCA?" and "are the driver/response subspaces = `B`?"
**Refs:** Björck & Golub 1973; random-subspace null — Aubrun (arXiv:2109.06535); JL angle-preservation (arXiv:1907.06166); common-subspace-in-DNNs (arXiv:2110.02863).
**Pitfalls:** in `d=3584`, two *random* k-subspaces already overlap substantially unless `k ≪ d` — subtract the random-subspace null band. Overlap is basis/whitening dependent.

### T6. Random-subspace & permutation nulls for restriction/overlap significance
**Computes:** the null distribution of every T4/T5 statistic under "no context→answer structure."
**Recipe:** **(a) Pairing-permutation null:** shuffle the row correspondence between `X` and `Y`, refit ridge, recompute → empirical null for `σ_i`, energy fractions, subspace overlaps. **(b) Random-subspace null:** replace `B` (or `P_C`) with Haar-random subspaces of the same dim. Report the observed statistic's percentile.
**Refs:** Ding, Denain, Steinhardt (arXiv:2108.01661); random-subspace geometry (arXiv:2109.06535).
**Pitfalls:** the permutation null must refit ridge (same λ) *inside each permutation*, with the standardization fit inside the loop, or you leak the true covariance. Use ≥1000 draws.

### T7. Structural decomposition: polar, symmetric/antisymmetric, distance-to-canonical-forms
**Computes:** what *kind* of operator `Ŵ` is — rotation vs stretch, symmetric vs skew, and distance from identity/scalar/orthogonal/projection.
**Recipe:** **polar** `Ŵ = QP`: `Q` = rotational part, `P` = directional stretch. **Sym/skew split**; ratio `‖Ŵ_skew‖_F/‖Ŵ‖_F`. **Distances:** `‖Ŵ−I‖_F`, `min_c‖Ŵ−cI‖_F` (optimal `c = tr(Ŵ)/d`), `min_{Q∈O(d)}‖Ŵ−Q‖_F` (= `Σ(σ_i−1)²`), `‖Ŵ−Ŵ²‖_F` (projection-like?).
**Refs:** Higham 1986; Golub & Van Loan; Schönemann 1966.
**Pitfalls:** all distances are **scale-sensitive** — compute in a common whitened metric. Ridge shrinks `σ_i<1`, biasing toward 0 and inflating `‖Ŵ−I‖` — compare to the shrinkage-matched null.

### T8. Commutator with input covariance: `[Ŵ, Σ_x] ≈ 0`?
**Computes:** whether `Ŵ` acts as a *filter in the input-PCA basis* vs mixes principal directions.
**Recipe:** `C = ŴΣ_x − Σ_xŴ`; report `‖C‖_F / (‖Ŵ‖_F‖Σ_x‖_F)`; or off-diagonal energy of `Ŵ` in the `Σ_x` eigenbasis.
**Refs:** Golub & Van Loan; Dobriban & Wager (arXiv:1507.03003).
**Pitfalls:** ridge already diagonalizes the *estimator* w.r.t. `Σ_x`; test the commutator of the **cross-map** `Σ_x^{-1}Σ_xy`, not the trivially-filtered ridge artifact. Restrict to the top-`n−1` reliable PCs.

### T9. Orthogonal Procrustes / shape metrics between context and answer bases
**Computes:** the best rigid alignment of context to answer activations, and the residual — a companion to "how close to a rotation is the map."
**Recipe:** `min_{Q∈O(d)}‖Y − XQ‖_F` → `Q = UVᵀ` from `SVD(XᵀY)`; compare residual to the ridge residual. Embed in **generalized shape metrics**.
**Refs:** Schönemann 1966; Williams et al. (arXiv:2110.14739); stochastic extension (arXiv:2211.11665); stitching/affine-matching (arXiv:2110.14633).
**Pitfalls:** Procrustes at `d≫n` also overfits — do it in a top-PCA-reduced common space, validate on held-out rows.

### T10. Marchenko–Pastur / ridge-noise null for the singular spectrum
**Computes:** the bulk of singular values expected from pure noise at your `(n,d,λ)`.
**Recipe:** simulate `Y_null = X W_0 + E` with `W_0=0`; fit ridge; collect the null `σ_i` bulk edge. Or the **MP law** + **optimal hard threshold**; RIE cleaning for the oracle shrinkage.
**Refs:** Marchenko & Pastur 1967; Bouchaud & Potters (arXiv:0910.1205); Gavish & Donoho 2014; Dobriban & Wager (arXiv:1507.03003).
**Pitfalls:** MP assumes near-isotropic noise; your answer noise is anisotropic → prefer the **simulated** null with the empirical noise covariance. `λ=1e3` moves the edge — recompute per λ.

### T11. Spiked-model / BBP analysis: are the top singular directions trustworthy?
**Computes:** whether the leading singular vectors reflect true signal or are noise-tilted, as a function of SNR — plus the expected **overlap** between estimated and true top direction.
**Recipe:** BBP: a planted direction is detectable only above a critical SNR (`~√(d/n)`-type); above it, squared overlap has a closed form (Paul 2007). Estimate per-direction SNR from the gap between `σ_i` and the null edge; read off expected `cos²`(estimate, truth).
**Refs:** Baik, Ben Arous, Péché (arXiv:math/0403022); Johnstone 2001; Paul 2007; Miolane (arXiv:1806.04343); Cai, Han, Pan (arXiv:1711.00217).
**Pitfalls:** `n=50, d=3584` → `d/n≈72`, an *extreme* undersampling regime where **only very strong spikes are recoverable** — the single strongest reason to prefer the `n≈2500` per-example grain for direction-level reads.

### T12. Bootstrap CIs on singular values AND singular subspaces
**Computes:** sampling uncertainty of `σ_i` and of the top singular subspaces.
**Recipe:** resample rows of `[X|Y]`, refit ridge, recompute. **Values:** percentile CIs. **Subspaces:** principal angle between each bootstrap top-`k` subspace and the full-sample one. Also bootstrap the T4 energy fractions and T5 overlaps.
**Refs:** Ding et al. (arXiv:2108.01661); CKA reliability (arXiv:2210.16156).
**Pitfalls:** `n=50` bootstrap is coarse. Bootstrap **subspaces** (principal angles), not individual vectors (sign/rotation ambiguity).

### T13. Preprocessing & λ sensitivity: GCV/PRESS, effective d.o.f., pooling/centering
**Computes:** how much every read above is an artifact of λ, standardization, centering, mean-pooling.
**Recipe:** GCV / PRESS-LOOCV for λ; report **effective degrees of freedom** `df(λ)=tr H_λ = Σ σ_i²/(σ_i²+λ)` alongside every spectrum. Redo the headline read at λ∈{GCV-opt, 10×, 0.1×} and under {standardized vs raw, centered vs uncentered `Y`, mean- vs last-token pooling}.
**Refs:** Golub, Heath, Wahba 1979; Liu & Dobriban (arXiv:1910.02373); Dobriban & Wager (arXiv:1507.03003).
**Pitfalls:** at `n≈d` GCV can be unstable/degenerate (#779's observed note); when GCV is flat, pin λ by held-out `R²` and **report `df(λ)`**. Uncentered `Y` injects a rank-1 mean; mean-pooling is itself a linear operator that can manufacture low-rank appearance.

## (2) Closest prior work (verified arXiv IDs)

- **arXiv:1004.2995** — Bunea, She, Wegkamp, RSC + effective rank of a regression coefficient matrix.
- **arXiv:1905.00414** — Kornblith et al., CKA; rotation-invariant statistics meaningless at d > n.
- **arXiv:1706.05806** — Raghu et al., SVCCA.
- **arXiv:1806.05759** — Morcos et al., projection-weighted CCA.
- **arXiv:2110.14739** — Williams et al., Generalized Shape Metrics.
- **arXiv:2108.01661** — Ding, Denain, Steinhardt, statistical testing for representation similarity.
- **arXiv:math/0403022** — Baik, Ben Arous, Péché, BBP transition.
- **arXiv:1507.03003** — Dobriban & Wager, high-dim ridge asymptotics.
- **arXiv:1910.02373** — Liu & Dobriban, ridge structure + CV.
- **arXiv:0910.1205** — Bouchaud & Potters, RMT toolkit.
- **arXiv:2107.04084** — Jazayeri & Ostojic, effective dimensionality.
- **arXiv:2210.16156** — Davari et al., CKA reliability.

## (3) Top-3 priority picks

1. **Predictable-variance / CCA spectrum with held-out R² (T3), not raw rank (T1).**
2. **Restricted-map energy fractions vs random-subspace + pairing-permutation nulls (T4 + T6).**
3. **BBP/spiked-model gating of any direction-level claim (T11), reported at both grains.**

**Cross-cutting warning:** report `df(λ)` and re-run headlines at ≥3 λ values (T13) — at `λ=1e3, n≈50` the shrinkage, not the data, may be setting the apparent rank.

---

# Appendix D — LLM / NN interpretability prior art (agent report, verbatim)

# Analyzing a linear map W: R³⁵⁸⁴→R³⁵⁸⁴ (context→answer ridge operator) — literature-grounded techniques

Framing note: our W is a **data-level regression operator**, not a network weight matrix. Two consequences recur: (i) its spectrum entangles the true map with the input activation covariance — whiten first or interpret in a covariance-aware inner product; (ii) with n ≈ 50–2500 against d = 3584, a large fraction of singular directions are estimation noise, so a **null model is not optional**. Because in-space = out-space, eigen-analysis is licensed (the one genuine affordance the inter-layer-lens literature lacks).

## Part 1 — Techniques

### 1. Eigen-decomposition as a copying/transport detector
- Elhage et al.'s OV-circuit analysis: **positive real eigenvalues of W_OV = a "copying" signature**; negative/complex = anti-copying/rotation. Direct precedent for eigen-analysis of a residual→residual endomorphism.
- **Recipe:** `eig(W)`; fraction of eigenvalue mass with positive real part; complex-plane plot; decode top eigenvectors by logit lens + cosine to persona vectors. Eigenvalues near +1 with persona-aligned eigenvectors = "trait copied verbatim into the answer."
- **Refs:** Elhage et al., *A Mathematical Framework for Transformer Circuits* (Transformer Circuits Thread, 2021 — no arXiv id); Millidge & Black (below).
- **Pitfalls:** ridge W is **non-normal** — pair with field-of-values / shuffled-pairs null; spectrum reflects Cov(x) too.

### 2. SVD: singular spectrum + interpretable singular directions
- Right singular vectors = context directions the map reads; left = answer directions it writes; Millidge & Black showed transformer weight singular vectors decode to interpretable token directions.
- **Recipe:** decode top-k right/left vectors via logit lens + cosine to persona vectors; report (v_i → u_i, σ_i) triples.
- **Refs:** Millidge & Black, AI Alignment Forum 2022 (no arXiv id; https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight); Martin & Mahoney (1810.01075).
- **Pitfalls:** Σ depends on input whitening; ridge shrinks σ non-uniformly. Report raw + whitened.

### 3. Effective rank / low-rank structure
- LRE's central finding: faithful relation maps are **low-rank**.
- **Recipe:** sweep truncated-SVD rank, re-measure faithfulness; report the plateau rank.
- **Refs:** Hernandez et al. (2308.09124).
- **Pitfalls:** apparent rank **confounded by ridge λ** — fix λ by CV, report rank/λ jointly, compare to permuted-pairs null.

### 4. Faithfulness (predictive R² / cosine of Wx vs true y)
- **Recipe:** held-out cosine(Wx, y) and R²; benchmark against **predict-the-mean** and additive task-vector baselines.
- **Refs:** Hernandez et al. (2308.09124); Akyürek et al. (2211.15661).
- **Pitfalls:** use group-level held-out folds (leave-persona/topic-out), not pointwise LOO.

### 5. Causal / behavioral faithfulness via low-rank pseudo-inverse editing
- LRE's causality test: edit the subject representation with the **rank-reduced inverse** to change the predicted object.
- **Recipe:** Δx = W⁺Δy (rank-reduced), inject at context position; or inject W·v_persona at answer position. Measure on-policy behavior with judge-rate + log-P dual DV.
- **Refs:** Hernandez et al. (2308.09124); ROME (2202.05262); ActAdd (2308.10248); Persona Vectors (2507.21509).
- **Pitfalls:** W⁺ ill-conditioned near n≈d — rank-reduce first; measure off-target leakage.

### 6. Bilinear projection onto persona/concept subspaces (the "gain matrix")
- G = UᵀWV: entry G_ji = gain from "trait i in context" to "trait j in answer" (Observable Propagation coupling coefficients applied to a fitted map).
- **Recipe:** build G over the trait dictionary; heatmap. Diagonal dominance = self-transfer; off-diagonal = cross-trait leakage.
- **Refs:** Dunefsky & Cohan (2312.16291); Elhage et al.; Park et al. (2311.03658) for the causal inner product.
- **Pitfalls:** persona vectors not orthonormal — orthonormalize or use the causal inner product first.

### 7. Logit-lens / vocabulary decoding of the map's action
- **Recipe:** decode top singular/eigen directions and Wx outputs to token space; prefer a **tuned lens** at L14 (raw logit lens is brittle mid-stack).
- **Refs:** Belrose et al. (2303.08112); Pal et al., Future Lens (2311.04897); Patchscopes (2401.06102).
- **Pitfalls:** the answer is generated content, not a next-token distribution — cross-check with judge-scored behavior.

### 8. Spectral / random-matrix null
- **Recipe:** ESD vs Marchenko–Pastur bulk for (n, d); **permute the (context, answer) pairing and refit** — any structure must exceed this null.
- **Refs:** Martin & Mahoney (1810.01075).
- **Pitfalls:** the highest-priority safeguard at n≈d; at n≈50 the MP bulk swamps almost everything.

### 9. ICL-as-ridge-regression framing + additive-baseline / compositionality tests
- **Recipe:** test W against the additive task-vector baseline (answer ≈ query-processing + a mean context offset); test compositionality W(x_a+x_b); test whether persona vectors are near-eigenvectors.
- **Refs:** Hendel et al. (2310.15916); Todd et al. (2310.15213); Akyürek et al. (2211.15661); von Oswald et al. (2212.07677).
- **Pitfalls:** the additive baseline is often surprisingly strong — report the delta honestly.

### 10. Symmetric/antisymmetric split, normality, trace-copying score
- **Recipe:** trace(W)/d, ‖sym‖/‖antisym‖, normality gap ‖WᵀW−WWᵀ‖; decode leading symmetric eigenvectors.
- **Refs:** Elhage et al. (no arXiv).
- **Pitfalls:** on strongly non-normal W report the numerical range, not eigenvalues alone.

### 11. Concept-erasure / nullspace probing of the map's input
- **Recipe:** (a) right null space / smallest singular directions = context geometry the answer discards; (b) LEACE-erase a persona direction from x, re-apply W, measure the answer-trait drop.
- **Refs:** LEACE (2306.03819); INLP (2004.07667).
- **Pitfalls:** linear-only guarantees; measure collateral.

### 12. Stitching / transplant test (behavioral sufficiency)
- **Recipe:** inject Wx as the actual L14 residual at the answer position; measure downstream loss/behavior recovery.
- **Refs:** Lenc & Vedaldi (1411.5908); Bansal, Nakkiran & Barak (2106.07682); Csiszárik et al. (2110.14633).
- **Pitfalls:** injection must match the exact layer/position the map was fit on.

### 13. Cross-map comparison: prefix-W vs context-W (and across layers/traits)
- **Recipe:** principal angles between top-k right-singular subspaces of prefix-W and context-W; CKA between outputs; Procrustes residual.
- **Refs:** CKA (1905.00414); SVCCA (1706.05806); relative representations (2209.15430); Mikolov (1309.4168); MUSE (1710.04087).
- **Pitfalls:** CCA-family similarity meaningless when d > n (Kornblith).

### 14. SAE / crosscoder feature-basis read of W
- **Recipe:** with an SAE at L14 (encoder E, decoder D), score E W D between dictionary atoms → sparse context-feature → answer-feature transfer graph.
- **Refs:** Transcoders (2406.11944); Dedicated Feature Crosscoders (2602.11729); Delta-Crosscoder (2603.04426).
- **Pitfalls:** requires a trained SAE; overcomplete basis complicates the bilinear score.

## Part 2 — Closest prior work (all ids MCP-verified)

1. **2308.09124** — Hernandez et al., *Linearity of Relation Decoding* — the single closest published analogue to W.
2. **2310.15213** — Todd et al., Function Vectors.
3. **2310.15916** — Hendel et al., ICL Creates Task Vectors.
4. **2303.08112** — Belrose et al., Tuned Lens.
5. **2311.04897** — Pal et al., Future Lens.
6. **2401.06102** — Ghandeharioun et al., Patchscopes.
7. **2312.16291** — Dunefsky & Cohan, Observable Propagation.
8. **2202.05262** — Meng et al., ROME (+ MEMIT 2210.07229).
9. **2306.03819** — Belrose et al., LEACE (+ INLP 2004.07667).
10. **2209.15430** — Moschella et al., Relative Representations.
11. **2211.15661** — Akyürek et al., ICL ≈ closed-form ridge.
12. **2507.21509** — Chen et al., Persona Vectors.

Non-arXiv but central: Elhage et al. (Transformer Circuits Thread 2021); Millidge & Black (AI Alignment Forum 2022).

## Part 3 — Top-3 priority picks

1. **LRE (2308.09124) as the methodological template** — port its faithfulness + causality protocol wholesale (with group-level folds and the dual DV).
2. **Elhage-framework eigenvalue-copying analysis** — the endomorphism is the unique affordance; positive-real persona-aligned eigenvalues would directly evidence "context traits are copied into the answer."
3. **Bilinear persona-projection G = UᵀWV gated by an MP/permuted-pairs null** — the most decision-relevant artifact (a pre-fine-tuning trait-transfer table), hallucination-prone at n≈d without the null.

---
---

# Round 2 (2026-07-06/07): prior-art deep dive + novelty assessment

Second sweep, prior-art-focused (3 parallel agents): (E) the exact object — context-summary →
answer-summary maps + pre-fine-tuning predictors of fine-tuning effects; (F) token-axis
h_t → h_{t+1} activation maps; (G) layer-axis h_ℓ → h_{ℓ+1} maps. Full reports verbatim in
Appendices E–G. Verification caveat: the arXiv MCP was 429-throttled for parts of the run, so some
ids (esp. 2026-dated ones) were verified via live arxiv.org/abs pages rather than MCP; the agents
flag per-id status. Re-verify flagged ids before citing in a paper.

## Novelty assessment (synthesis)

### Per-axis verdicts

**The exact object — pooled-context → pooled-answer, same layer, same model: none found.**
No published work fits W: R^d→R^d from mean-pooled context activations to mean-pooled
generated-answer activations at the same layer and analyzes W as an operator; and no work uses
the structure of such a fitted map to predict fine-tuning effects. Closest neighbors, each missing
a defining element:
- *Activation Transport Operators* (2508.17540) — fitted regularized linear operators between two
  activation sites of the SAME model, with structural analysis — but the sites are different
  LAYERS at the same token (depth transport, SAE-feature diagnostic), not different spans.
- *Belief-state geometry* (2405.15943; mechanism 2502.01954) — the conceptually deepest precedent:
  the residual stream linearly encodes a context-summary (belief state) that determines the
  future-output DISTRIBUTION — but toy HMM generators, a distribution target (not an
  answer-activation vector), a decode claim (not a fitted operator), no spectrum/rank.
- *LRE* (2308.09124) — fitted affine subject→object relation maps — but Jacobian-local,
  relation-specific, token-grain.
- *Persona vectors* (2507.21509 A3.3) + *persona features* (2506.19823) + EM-susceptibility line
  (2606.20225 etc.) — predict fine-tuning-induced behavior shifts from base-model geometry — but
  every one is direction→SCALAR; none is an operator-valued predictor.
- Transferability estimation (LogME/LEEP/Task2Vec) — the classic "predict fine-tuning outcome from
  pre-FT features" lineage — scalar/task-level, degrades after full fine-tuning.

**Token axis — essentially no operator characterization exists.** EAGLE (2401.15077) TRAINS the
next-feature map for speed and remarks only qualitatively that features are "more regular than
tokens"; EAGLE-3 (2503.01840) reports feature-prediction is a scaling CONSTRAINT (the strongest
published hint the map is nontrivial) — neither analyzes the map. The closest characterization is
sentence-grain: a switching linear dynamical system over hidden-state trajectories (2506.04374 —
rank-40 projection, 4 latent regimes). Real rank/spectrum reads of a per-token state operator
exist ONLY where the architecture hands you the operator explicitly (linear-attention/SSM state
matrices, 2602.02195) — wrong architecture for a softmax transformer's residual stream. Trajectory
GEOMETRY (DMET 2505.20340, curvature/attractors 2502.15208 in text space) exists; the operator
does not. **Open gap.**

**Layer axis — crowded periphery, empty center.** Many same-position layer-map artifacts exist:
the lens family (tuned lens 2303.08112; layer→final shortcuts 2303.09435), stitching translators,
Secretly Linear (2405.12250 — fits the ℓ→ℓ+1 map but reduces it to ONE Procrustes scalar, 0.99,
and its own ablation shows the near-linearity is residual-dominated i.e. trivially expected),
layer-redundancy metrics (2403.17887 angular distance; ShortGPT cosine; Painters 2407.09298
reordering), depth-stages (2406.19384), and local-Jacobian spectra (2505.24293 detached-Jacobian
SVD; 2602.12384, 2605.14258). But NO work fits a population-regression ℓ→ℓ+k map and reports
rank/spectrum/invariant subspaces with estimation-noise discipline, and NOTHING operates at a
span-pooled grain. Two inherited cautions: (i) frame any layer-axis map against the identity
baseline and characterize the deviation-from-identity operator (near-identity is trivial there);
(ii) SVD misses rotational structure carried by complex-eigenvalue pairs (2603.13259) — read the
eigen/Schur spectrum too.

### Overall impression of novelty

1. **The genre is not novel; the object is.** "Linear maps between activations of one LLM" is a
   busy genre (lenses, stitching, secretly-linear, Koopman/DMD-on-activations, EAGLE). What is
   consistently missing — across all three axes — is treating the FITTED map as the object of
   study: operator-level characterization (rank, spectrum, invariant subspaces, normality) with
   estimation-noise discipline. That center is empty even on the crowded layer axis.
2. **The specific object appears genuinely new** on three independent sweeps: the span-pooled
   grain (context summary → generated-answer summary) has no published instance on ANY axis; the
   span pair (context vs own answer, same layer) has none; and the downstream use (an
   operator-valued predictor of fine-tuning-induced leakage, generalizing the direction→scalar
   persona-vectors/persona-features predictors) has none.
3. **A useful structural point for positioning:** on the layer axis, near-identity structure is
   trivially expected (the residual path). The context→answer map has NO identity path — the two
   spans are disjoint token ranges — so identity-likeness, if found, is a substantive finding
   there, not an artifact. The same fact cuts the other way: reviewers steeped in the layer-axis
   literature will assume triviality and must be shown the difference.
4. **The novelty is object + use, not mathematics.** The characterization toolkit (DMD/Koopman,
   RRR, CME, RMT cleaning) is mature and imported; a Koopman-community reviewer will read this as
   an application. The defensible claim shape: "we introduce and characterize the context→answer
   transfer operator of an LLM at the span grain, and show its structure predicts
   fine-tuning-induced leakage" — with ATOs, LRE, belief-state geometry, Secretly Linear, EAGLE-3,
   and persona vectors/features as the six named nearest neighbors to position against.
5. **Confidence: HIGH** that the exact object is unpublished, after the targeted check below was
   executed (upgraded from MODERATE-HIGH on 2026-07-07). Residual caveat: Semantic Scholar's
   citation index lags ~1–2 months for the newest preprints, and the LW/AF gray literature moves
   fast — re-run the citation screen at paper-submission time.

### Citation-graph check — EXECUTED 2026-07-07 (no competitor found)

- **ID re-verification:** all 30 flagged / load-bearing arXiv ids returned by the agents were
  re-verified against the live arXiv API (`export.arxiv.org` `id_list` batch) — 30/30 resolved
  with titles matching the agents' claims. No fabricated or mis-attributed ids.
- **Citation screen:** all Semantic-Scholar-indexed citing papers of the three anchor papers were
  pulled and every 2026-dated citer screened by title, with abstracts fetched for the 8
  competitor-shaped ones. ATOs 2508.17540: 0 indexed citations. Belief-state 2405.15943: 60
  citations, 26 from 2026 — none fit a context→answer or span-pooled activation map. LRE
  2308.09124: 193 citations, 57 from 2026 — none fit the object.
- **New neighbors worth knowing (surfaced by the screen, not competitors):**
  - **"As X, Do Y: How Persona and Task Combine in Instruction-Tuned LLMs" (2605.23147)** — the
    closest NEW neighbor. Finds an ADDITIVE persona+task decomposition of the residual stream at
    the prompt→answer transition (last prompt token + first two generated tokens, early/mid
    layers, Qwen-2.5-1.5B/3B + Gemma-2-2B), and a NEGATIVE result: the role prompt canNOT be
    compressed into one cached residual vector — persona-conditioned generation keeps attending
    back to the distributed prompt/KV positions. Deltas vs our object: additive shift, not a
    fitted map; 3 token positions, not span-pooled; no operator read; no FT prediction. Bears
    directly on open question 1.1 (context-as-a-vector): evidence AGAINST single-site
    single-vector sufficiency for reproducing full persona-conditioned generation — note our
    map's target (answer-side MEAN activation) is a much weaker sufficiency claim, so the two
    results can coexist; cite and distinguish.
  - **"Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning"
    (2606.07631)** — tracks drift along 7 trait directions across finetuning checkpoints
    (7–9B models), finds a low-dim EM drift axis (65.5% variance), builds a checkpoint monitor
    (AUROC 0.990). The monitoring-USE neighbor: during-FT, direction-based, post-hoc drift —
    not a pre-FT predictor and no context→answer operator. Cite in the leakage-prediction
    positioning alongside persona vectors / persona features.
  - **"Relational Linearity is a Predictor of Hallucinations" (2601.11429)** — uses a PROPERTY
    of LRE-style relation maps (their linearity) to predict a behavior (hallucination vs
    refusal, r ∈ [.58, .84]). Precedent for the claim SHAPE "structure of a fitted linear map
    predicts behavior" — at relation grain, different behavior; strengthens rather than
    threatens the novelty of an operator-valued leakage predictor.
  - Also catalogued: 2605.22532 (KL-based relational-linearity probing, LRE-method successor);
    2603.19664 (per-token residual vectors deterministically fix KV — "residual stream is the
    sole state," relevant to context-as-code framing); 2606.32022 (SemRF depthwise-trajectory
    formalism, layer axis); 2605.17084 (readout-aligned geometry vs scale); 2602.04863
    (log-linear subliminal data-selection mechanism, EM-line relevant).

---

# Appendix E — Prior art: context→answer maps (agent report, verbatim)

# Prior-art search: context-summary → answer-representation maps

**Verification note:** arXiv MCP `get_abstract` returned HTTP 429 on every call this session. Every arXiv id below was instead verified by fetching its `arxiv.org/abs/<id>` page and reading the live title/authors/abstract. None are fabricated; where I could not resolve a clean arXiv id I say so and give title+venue.

## (1) Ranked closest prior work

**1. Activation Transport Operators (ATOs)** — Szablewski & Masiak, arXiv **2508.17540** (Aug 2025). https://arxiv.org/abs/2508.17540
- *What they fit:* explicit **regularized linear maps** from **upstream residual-stream activations to downstream residuals k layers later**, evaluated in SAE-feature space; used to test whether a feature is *linearly transported* vs *nonlinearly re-synthesized* across depth.
- *Delta:* Single closest published object on the "fit an explicit linear operator between two activation sites of the *same* model and characterize where it is/isn't linear" axis. **Matches:** same-model, explicit fitted linear map between two activation sites, structural (feature-space) analysis. **Differs:** the two sites are **different layers, same token** (depth transport), NOT **different spans (context vs generated answer) at the same layer**; per-token, not span-pooled; feature-transport diagnostic, not an operator whose rank/spectrum is characterized and then used to predict fine-tuning leakage.

**2. Transformers Represent Belief State Geometry in their Residual Stream** — Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers, arXiv **2405.15943** (NeurIPS 2024). https://arxiv.org/abs/2405.15943
- *What they characterize:* the residual stream (a running summary of the *context*) **linearly encodes a belief state** — a distribution over the data-generating process's hidden state — and that belief state is exactly the sufficient statistic that **determines the future-output distribution**. Linear decodability of the (often fractal) mixed-state-presentation geometry.
- *Delta:* Strongest conceptual precedent for "a context representation linearly determines what comes next." **Matches:** context-summary representation in the residual stream, linearly related to the future output. **Differs:** decodes context→**belief-state / next-token distribution** (a probability object), not context-vector→**answer-activation-vector**; a *decoding* claim (probe exists), not a *fitted operator W* between two pooled activation summaries; tiny synthetic HMM generators, not a 7B chat model over (persona, question, answer) triples; no operator rank/spectrum analysis; no fine-tuning-prediction use.

**3. Constrained Belief Updates Explain Geometric Structures in Transformer Representations** — Piotrowski, Riechers, Filan, Shai, arXiv **2502.01954** (Feb 2025). https://arxiv.org/abs/2502.01954
- *What they characterize:* the *mechanism* by which context maps to the internal belief representation — attention implements a constrained/parallelized Bayesian belief update in the probability simplex, producing distinctive geometric structure.
- *Delta:* Same lineage as #2; characterizes the context→belief-representation map's geometry, but still HMM-scale, belief-distribution target, no fitted vector→vector operator, no fine-tuning-prediction use.

**4. Persona Vectors: Monitoring and Controlling Character Traits** — Chen, Arditi, Sleight, Evans, Lindsey, arXiv **2507.21509** (2025, *known/sibling — position only*). https://arxiv.org/abs/2507.21509
- *What they fit:* the **projection-difference predictor** (appendix): project training-data responses onto a base-model persona direction; that scalar projection **predicts the post-fine-tuning trait shift** (and the A3.3-style `E ≈ r_Bᵀv` read-out predictor).
- *Delta:* Closest on the "use pre-fine-tuning geometry to predict fine-tuning behavior" axis. **Matches:** base-model activation geometry → prediction of fine-tuning-induced behavior. **Differs:** predictor is **activation-direction → scalar** (trait dose), not an **operator** over answer representations; reads the *training data's* projection onto one direction, not a fitted **context→answer map** characterized as an operator.

**5. Persona Features Control Emergent Misalignment** — Wang, Dupré la Tour, Watkins, Makelov, Chi, …, **Mossing**, arXiv **2506.19823** (2025, *known/sibling — position only*). https://arxiv.org/abs/2506.19823
- *What they do:* SAE "misaligned persona features" whose activation **predicts whether a model will exhibit misalignment** after narrow fine-tuning; toxic-persona feature particularly predictive; steering these features controls EM.
- *Delta:* activation-feature → **scalar behavioral prediction** of a fine-tuning outcome. No context→answer map, no operator characterization; predicts a behavior label, not an answer representation.

**6. In-context Vectors (ICV)** — Liu, Ye, Xing, Zou, arXiv **2311.06668** (2023). https://arxiv.org/abs/2311.06668
- *What they do:* a forward pass on demonstrations yields a single **in-context vector from the latent embedding**; adding it shifts latent states on a new query, steering the output (with vector arithmetic over ICVs).
- *Delta:* **context→one vector** that causally shapes the output — a *summary of context used generatively*. **Differs:** additive steering shift, not a **fitted map to the answer representation**; no operator rank/spectrum; not used to predict fine-tuning.

**7. Task Vectors in In-Context Learning: Emergence, Formation, and Benefit** — Yang, Lin, Lee, Papailiopoulos, Nowak, arXiv **2501.09240** (Jan 2025). https://arxiv.org/abs/2501.09240
- *What they do:* study how a single **task vector** (context compressed to a vector) emerges and can be strengthened via a task-vector prompting loss.
- *Delta:* characterizes the *context→vector* compression and its localization; the vector steers the answer but they do not fit/characterize a **context-rep → answer-rep operator**.
- *(Same bucket, KNOWN — position only:* function vectors 2310.15213; task vectors/arithmetic 2310.15916.)*

**8. Learning to Compress Prompts with Gist Tokens** — Mu, Li, Goodman, arXiv **2304.08467** (NeurIPS 2023). https://arxiv.org/abs/2304.08467
- *What they do:* train the LM (via attention-mask tricks) to **compress a prompt into a few "gist" token vectors** cached and reused at generation (up to 26×).
- *Delta:* strongest "context → vector summary that then drives generation" example, but a **trained soft-prompt distillation**, not a **fitted post-hoc linear map** between two activation summaries; no operator analysis; no fine-tuning prediction.

**9. Code Correctness Signals in LLM Hidden States: Pre-Generation Probing and Repair Geometry** — Di Cicco, arXiv **2606.14530** (Jun 2026). https://arxiv.org/abs/2606.14530
- *What they do:* linearly decode **first-attempt code correctness from the prompt-final hidden state** *before generation* (held-out AUC 0.931), with **residualization** to remove prompt-length confounds; honestly reports a negative result on a repair-direction.
- *Delta:* prompt-activation → **scalar** correctness. Methodologically the *closest kin to the project's own habits* (residualization confound control, honest negative), but target is a label, not an answer representation; single prompt-final token, not a pooled span; no operator.

**10. No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes** — Moreno Cencerrado, Padrés Masdemont, Gonzalvez Hawthorne, Africa, Pacchiardi, arXiv **2509.10625** (Sep 2025). https://arxiv.org/abs/2509.10625
- *What they do:* extract activations **after the question, before generation**; a linear probe onto an "in-advance correctness direction" forecasts answer correctness, generalizing 7–70B and across datasets.
- *Delta:* prompt-side activation → **scalar** accuracy. Same near-miss shape as #9; no answer-representation target, no operator.

**11. Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations** — Collu, Conte, Giaretta, Kleyko, Conti, Zavatteri, Confalonieri, arXiv **2605.28553** (May 2026). https://arxiv.org/abs/2605.28553
- *What they do:* linear probes on residual-stream activations at each block **predict refusal before decoding**.
- *Delta:* activation → **scalar** refusal label. Near-miss (d); no context→answer map.

**12. Looking Inward: Language Models Can Learn About Themselves by Introspection** — Binder, Chua, Korbak, et al., arXiv **2410.13787** (2024). https://arxiv.org/abs/2410.13787
- *What they do:* fine-tune a model to **predict properties of its own future behavior**; a model predicts itself better than another model can (privileged access), hypothesized "self-simulation."
- *Delta:* **behavioral** self-prediction (output-property), not a representation→representation map; no operator; near-miss on "predict what the answer will be" at the behavior level.

**13. Future Lens: Anticipating Subsequent Tokens from a Single Hidden State** — Pal, Sun, Yuan, Wallace, Bau, arXiv **2311.04897** (2023, *known — position only*). https://arxiv.org/abs/2311.04897
- *What they do:* a **single hidden state** approximates the model's prediction of tokens at positions ≥ t+2 (>48%).
- *Delta:* single-state → **future tokens** (not answer *representations*, not span-pooled), decoding not a fitted operator. Known anchor for bucket (a).

**14. Actionable Activation Directions for Detecting and Mitigating Emergent Misalignment Across Model Families** — Syed, arXiv **2606.20225** (Jun 2026). https://arxiv.org/abs/2606.20225
- *What they do:* difference-in-means direction separates aligned/misaligned activations (99.6%) at the final layer **after** identical fine-tuning; cross-family transfer (Qwen/Gemma donors, Llama receiver); base-prior-adjusted membership-inference metrics predict susceptibility (AUC 0.849).
- *Delta:* activation direction → **scalar** EM detection/susceptibility; post-hoc detection, not a pre-FT context→answer operator. Relevant EM-prediction neighbor to bucket (e).

**15. (bucket-e neighbors, catalogued, not central):** *Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer* (arXiv **2605.12798**); *Model Organisms for Emergent Misalignment* (arXiv **2506.11613**, Turner et al. lineage); *Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning* (arXiv **2602.00298**). All predict/organize **misalignment susceptibility or transfer** from data/activation priors — scalar/behavioral targets, no context→answer operator. *(IDs surfaced via web search; abstracts not individually WebFetch-verified this session — verify before citing in a paper.)*

## (2) Near-miss taxonomy by bucket

**(a) Predicting response *representations* from prompt representations.** The pure form — fit y = W·x where both x and y are *pooled activation summaries* of the same model (context span vs generated-answer span) — **does not exist in the literature I could find.** Neighbors predict the *future output distribution/tokens* from a state (Future Lens 2311.04897; belief-state 2405.15943), or forecast the *next latent* as a training objective ("Next-Latent Prediction Transformers Learn Compact World Models," OpenReview forum Lh4ayjJIAW — verified as an OpenReview entry, no clean arXiv id resolved; provably converges to belief states). None fit a post-hoc pooled-span→pooled-span operator. Generic QA "predict the answer embedding from the question embedding" exists only in pre-LLM sentence-embedding/skip-thought regression work (weak, cross-encoder, not residual-stream, not operator-characterized).

**(b) Context-compression-to-vector, used predictively.** Rich and mature: ICV (2311.06668), function vectors (2310.15213, known), task-vector arithmetic (2310.15916, known), task-vector emergence (2501.09240), gist tokens (2304.08467), and the broader implicit-ICL / soft-prompt-distillation line. All compress **context → one vector** that *causally steers* the answer. **What's missing:** none *fit a map from the context vector to the answer vector* and *characterize that map as an operator* (rank, spectrum, subspace). They treat the context vector as an additive intervention, not the domain of a learned operator whose range is the answer representation.

**(c) Computational mechanics / belief-state geometry.** Closest conceptual match to "context vector determines answer vector": the residual stream linearly encodes a belief state that is the sufficient statistic for the future output (2405.15943; mechanism in 2502.01954). **Gap:** the target is a *distribution over futures* (belief state), not the *activation representation of the realized answer*; the map studied is context→belief-geometry (a decode), not a fitted context-summary→answer-summary operator; results are on small synthetic generators, never a 7B chat model over (persona, question, answer) triples at a chosen layer.

**(d) Predicting *behavior* (scalar) from prompt-side activations before generation.** Densest near-miss bucket: correctness (2606.14530, 2509.10625), refusal (2605.28553), hallucination pre-generation probing (multiple), plus behavioral self-prediction (2410.13787). All are **activation → scalar/label**, typically the prompt-final token, sometimes with residualization (2606.14530 is methodologically closest to the project). **Gap:** none produce an answer *representation*; the codomain is a 1-D behavior axis, not R^d, and there is no operator.

**(e) Predicting *fine-tuning outcomes* from pre-fine-tuning representations.** Two sub-lines. (i) Transferability estimation (LogME, LEEP, Task2Vec, encoder-selection e.g. arXiv 2210.11255) — predicts *transfer performance* from frozen features, explicitly *degrades after full fine-tuning*, scalar/task-level, never a context→answer operator. (ii) Persona/EM predictors: persona-vectors projection-difference (2507.21509) and OpenAI-lineage persona features (2506.19823) predict *post-FT trait/misalignment* from base-model directions; EM-direction/susceptibility work (2606.20225, 2605.12798, 2602.00298, 2506.11613). **Gap:** every one maps activation-direction/data-projection → **scalar** behavior shift. None construct an **operator W over representations** and use *its* structure (rank/spectrum/subspace) to predict fine-tuning-induced leakage — exactly the project's proposed mechanism.

**(f) Fitting a linear map between two *span-pooled* activation summaries of the same model and analyzing its rank/spectrum.** Most specific bucket, and the emptiest. The only same-model fitted linear operator with structural analysis is ATOs (2508.17540) — but cross-layer, per-token, feature-transport. Model-stitching / relative-representations work (Lenc–Vedaldi; Moschella et al.) fits linear maps between representation spaces but across **different models/networks**, not context-span→answer-span within one model. Direct searches for "linear map between mean-pooled hidden states of two token spans, rank/spectrum" returned **no on-target hit**.

## (3) Closeness verdict

**Single closest published object:** **Activation Transport Operators** (Szablewski & Masiak, arXiv 2508.17540). It is the only work that fits an *explicit, regularized linear operator between two activation sites of the same model* and characterizes *where that map is linear vs not*. **Gap statement:** its two sites are different *layers* at the *same token* (a depth-transport diagnostic in SAE-feature space), whereas our object is a map between two *different spans at the same layer* — the pooled **context** (prefix / prefix+query) and the pooled **generated answer** — fit over 50–2500 (context, answer) pairs and characterized as an operator (rank, spectrum, subspace) that is then used to **predict fine-tuning-induced behavior leakage**. On the *conceptual* axis the closest is the belief-state line (2405.15943 / 2502.01954): a context summary in the residual stream linearly determining the future output — but that determines a *distribution over futures* on toy HMMs, not a fitted pooled-context→pooled-answer activation operator on a 7B chat model.

**Honest bottom line — has anyone (i) fit + characterized a pooled-context → pooled-answer same-layer linear map, or (ii) used such a map to predict fine-tuning effects?**
- **(i) None found.** No published work fits W: R^d→R^d from mean-pooled *context* activations to mean-pooled *generated-answer* activations at the same layer and analyzes W as an operator (rank/spectrum/subspace). The nearest same-model fitted-linear-operator work (ATOs) is cross-layer per-token feature transport; the nearest conceptual work (belief-state geometry) targets a distribution, not an answer-activation vector, and is not a fitted operator.
- **(ii) None found for the *operator* form.** Predicting fine-tuning effects from base-model geometry exists, but only as **direction→scalar** predictors (persona-vectors projection-difference 2507.21509; OpenAI-lineage persona features 2506.19823; EM susceptibility 2606.20225 and neighbors). **No one uses the structure of a fitted context→answer *map* to predict fine-tuning-induced leakage.** That combination — the object *and* its predictive use — appears genuinely novel; position primarily against ATOs (2508.17540, "operator, but cross-layer not context→answer"), the belief-state line (2405.15943 / 2502.01954, "context linearly determines the future, but a distribution on toy data, not a fitted answer-activation operator"), and the persona/EM predictor line (2507.21509 / 2506.19823, "predict FT effects from base geometry, but direction→scalar, not a map").

---

# Appendix F — Prior art: token-axis activation→activation maps (agent report, verbatim)

# Prior art: token-axis activation→activation maps in LLMs

**Framing of the target object:** a *characterization* (rank / spectrum / operator structure / linearity score — not merely a trained-for-speed predictor) of the map from an activation at sequence position/time *t* to the activation at *t+1* (or *t+k*), **same layer, along the sequence/generation axis**, in a standard softmax-attention transformer's residual stream. All arXiv IDs below are verified against their abstract pages unless flagged "id unverified."

## (1) Ranked closest prior work

**1. A Statistical Physics of Language Model Reasoning — 2506.04374** (Carson & Reisizadeh, Jun 4 2025). Fits a **Switching Linear Dynamical System (SLDS)** to *sentence-level hidden-state trajectories* along the generation axis: a rank-40 projection captures ~50% variance, four latent reasoning regimes, drift-diffusion + regime switching, validated across 8 models / 7 benchmarks, and used to simulate trajectories cheaply. **DELTA:** the single closest thing to "fit a linear operator to token-axis activation dynamics and characterize it." But it operates at *sentence* granularity (not token t→t+1), on a low-rank *projection*, and reports a phenomenological drift-diffusion+SLDS with regimes/rank-of-projection — not the rank/spectrum of an explicit residual-stream h_t→h_{t+1} operator.

**2. Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control — 2604.19018** (Skifstad, Yang, Chou, Apr 21 2026). Shows layer-wise dynamics are "well-approximated by locally-linear models," models inference as a **linear time-varying (LTV) system** built from layer-wise Jacobians, and does LQR control. Most explicit *operator* treatment (Jacobians of an LTV system) in the set. **DELTA:** the operator is along the **depth/layer axis within one forward pass**, not the token/sequence axis. Wrong axis for the target, but the closest methodological template (fit local-linear Jacobian operators to activation dynamics + treat them as a control operator).

**3. Latent Trajectory Dynamics in LLMs: A Manifold Evolution Framework (DMET) — 2505.20340** (Zhang, Dong, Li, May 24 2025). Models generation as a controlled dynamical system on a low-dimensional semantic manifold, formalizes a first-order ODE with a semantic potential, and characterizes **token-axis trajectory geometry** via three metrics (state continuity, attractor clustering quality, topological persistence); online monitoring drives an adaptive decoder. **DELTA:** characterizes trajectory *geometry* (smoothness / basins / topology) along the generation axis via proxy metrics, not the rank/spectrum/linearity of the transition operator itself.

**4. State Rank Dynamics in Linear Attention LLMs — 2602.02195** (Sun et al., Feb 2 2026). Directly characterizes the **rank and spectrum** of the compressed per-token *state matrix* in linear-attention models — "State Rank Stratification," a spectral bifurcation across heads (near-zero-rank vs saturating), argued to be an intrinsic learned property; low-rank heads matter for reasoning. **DELTA:** a rank/spectrum characterization of a per-token state-update operator — but for **linear-attention / SSM-style state**, not the residual-stream h_t→h_{t+1} of a softmax transformer (Qwen-2.5-7B). Right question (rank/spectrum of a per-step operator), wrong architecture and wrong state object.

**5. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty — 2401.15077** (Li et al.). *Trains* a lightweight head to predict the next **second-to-top-layer feature** (hidden state) from the current feature + a one-step-advanced token embedding. Key structural claims: feature-level autoregression is "more straightforward than at the token level" (feature sequences more regular than discrete tokens), but "inherent uncertainty" constrains it (resolved by injecting the sampled token). **DELTA:** learns the h_t→h_{t+1} map *for speed* and asserts its relative regularity — but never characterizes it as an operator (no rank, spectrum, or linearity score; the map is a trained MLP/decoder-layer treated as a black box).

**6. EAGLE-3: Scaling up Inference Acceleration via Training-Time Test — 2503.01840** (Li et al.). Reports that EAGLE's **feature-prediction is a constraint** that caps data-scaling gains, and *abandons feature prediction* for direct token prediction with multi-layer feature fusion. **DELTA:** the strongest published signal that the naive h_t→h_{t+1} feature map is *limited/hard to scale* — a negative result about the map's learnability — but again not a characterization of its structure; a design pivot, not an operator analysis.

**7. Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential — 2507.11851** (Samragh et al., Jul 16 2025). Multi-token prediction from a single hidden state via masked-input joint prediction + gated LoRA + a learned sampler; ~5× (code/math), ~2.5× (chat) speedups. **DELTA:** predicts future *tokens* (multi-token-ahead) from current activations; it's about decodable future content, not a fitted activation→activation transition operator.

**8. ParaScopes: What Do LM Activations Encode About Future Text? — 2511.00180** (Pochinkov et al., Oct 31 2025). Residual Stream Decoders probe how much *future text* is linearly recoverable from current activations; finds ≥5 tokens of future context decodable in small models. **DELTA:** measures forward-looking *information content* (a decoding/probe question), not the map h_t→h_{t+k} as an operator with rank/spectrum. The paragraph-scale successor to Future Lens.

**9. Efficient Joint Prediction of Multiple Future Tokens (JTP) — 2503.21801** (Ahn, Lamb, Langford, Mar 24 2025). Enriches hidden states via joint multi-token prediction through a representation bottleneck; achieves a short-horizon **belief-state representation**. **DELTA:** a training objective that shapes the hidden state to carry future info; connects to belief-state geometry (#12) but does not fit or characterize an activation→activation map.

**10. Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries (FSP) — 2510.14751** (Mahajan et al., Oct 2025 / Mar 2026). Auxiliary head predicts a **compact representation of the long-term future** (bag-of-words or a reverse-LM embedding); improves long-horizon reasoning at 3B/8B. **DELTA:** predicts a *future summary embedding* from the present — conceptually adjacent to a "summary→future-summary" map, but it's a pretraining objective and the summary is of future *text*, not of the answer-side activation; no operator characterization.

**11. Transformers for Dynamical Systems Learn Transfer Operators In-Context — 2602.18679** (Bao, Lai, Gilpin, Feb/Apr 2026). Trains small transformers to forecast dynamical systems and shows they implement a **transfer-operator (Koopman-flavored) strategy in-context** — delay-embed to recover the manifold, then forecast invariant sets/attractors. **DELTA:** the transfer operator is over *external time-series data*, and attention is the mechanism learning it — a data-level transition operator, not the model's own activation→activation map. Strong conceptual analogue for "attention learns a transition operator."

**12. Transformers Represent Belief State Geometry in Their Residual Stream — 2405.15943** (Shai et al., NeurIPS 2024). Belief-updating over the data-generating HMM's states (the mixed-state presentation / MSP meta-dynamics) is **linearly represented** in the residual stream, including fractal geometry; belief states carry whole-future information. **DELTA:** characterizes the geometry of the *belief-state manifold* (the DGP's transition/update dynamics as reflected in activations), not a fitted h_t→h_{t+1} activation operator. The deepest theoretical link between token-axis transition structure and residual-stream geometry.

**13. Attention with Markov — 2402.04161** (Makkuva et al.). Transformers trained on Markov chains form induction heads that estimate the in-context bigram/transition kernel; single-layer loss-landscape theory (bigram global min vs unigram local min). **DELTA:** the transition operator learned is the **data-level Markov kernel**, characterized via loss landscape — position against activation-level maps.

**14. Unveiling Attractor Cycles in LLMs: A Dynamical Systems View of Successive Paraphrasing — 2502.15208** (Wang et al., Feb 2025). Iterated text→text mapping converges to **2-period attractor cycles**; a genuine dynamical-systems characterization (fixed points / limit cycles) of the generation loop. **DELTA:** dynamics are in **text space** (paraphrase iteration), not activation space; no operator/spectrum on activations.

**15. Relational Rank Geometry in Transformers — 2605.29634** (Kobrosly, May 28 2026). Rank-indexed geometry of relations among token tuples in hidden states (Plücker sign entropy), plus rank-frame steering. **DELTA:** a rank/geometry object over token-tuple *relations* at a position, not a token-axis transition map; relevant only as evidence that rank-structured hidden-state geometry is a studied object.

**Supporting/known:** Future Lens — 2311.04897 (multi-token-ahead decodability from a single hidden state; origin of this line, as flagged in the brief); DMD on LLM embeddings — 2309.01245 (Akrout — DMD spectrum of *sentence embeddings across paragraphs* for hallucination detection; token/sentence-axis DMD but on embeddings, for detection, not operator characterization — the "known" DMD paper); Local Intrinsic Dimensions of Contextual LMs — 2506.01034 (mean local dimension of latent space tracks training/overfitting/grokking — geometry of activations, not a transition map).

## (2) Near-miss taxonomy by bucket

- **(a) Feature-level autoregression for speculative decoding.** EAGLE **2401.15077**, EAGLE-3 **2503.01840** (verified); EAGLE-2 (id 2406.16858, unverified here). Hydra, GLIDE, Medusa (Medusa is logits/token-level as flagged in the brief; Hydra adds sequential draft heads; GLIDE reuses target KV — ids unverified, cited by name). **Common gap:** all *train* the next-feature/next-token map for speed; the map's rank/spectrum/linearity is never the object of study. EAGLE's "features more regular than tokens" and EAGLE-3's "feature prediction is a constraint" are the only structural remarks — qualitative, not quantitative.

- **(b) Predicting future hidden states / tokens from current state.** Future Lens **2311.04897**, Your-LLM-Knows-the-Future **2507.11851**, ParaScopes **2511.00180**, JTP **2503.21801**, FSP **2510.14751**, What's-the-Plan **2601.20164** (Maar/McDougall/Nanda — implicit-planning metrics + end-of-line steering vectors for rhyme/QA, planning universal from 1B), plus Anthropic's *On the Biology of a Large Language Model* (transformer-circuits.pub, 2025 — poetry rhyme planning / forward-planning gray literature). **Common gap:** these ask "what future info is present/decodable" or "how to shape it," not "what is the operator mapping this activation to the next."

- **(c) Dynamical-systems characterization of token-axis trajectories.** SLDS **2506.04374**, DMET **2505.20340**, attractor cycles **2502.15208**, DMD-on-embeddings **2309.01245**, local intrinsic dimension **2506.01034**, Curved Inference **2507.21107** (Manson — residual-stream trajectory *curvature/salience* under a pullback metric; **depth-axis geometry within a prompt**, not sequence-axis). This bucket is where the target lives; SLDS + DMET are the two nearest, both at sentence/manifold granularity with proxy metrics rather than an explicit h_t→h_{t+1} operator.

- **(d) Attention/OV as per-step transition operator.** Attention-with-Markov **2402.04161**, Transformers-learn-transfer-operators-in-context **2602.18679**, plus the induction-heads-on-Markov-chains line (Evolution of Statistical Induction Heads / Variable-Order Markov in-context — ids 2402.11004 / 2410.05493, unverified here). Belief-state geometry **2405.15943** sits between (d) and (f). **Common gap:** the transition operator these study is over *data* (Markov kernels, external time series), learned/represented by attention — not the model's own activation-space transition.

- **(e) Recurrent/SSM reformulations exposing a per-token state-update map + spectral analysis.** State Rank Dynamics **2602.02195** (verified — rank/spectrum of linear-attention state matrices). The "Transformers are RNNs" / linear-attention-as-RNN line (Katharopoulos et al. 2020, id 2006.16236 unverified here) and Mamba/SSD spectral analyses of the state-transition matrix A (Mamba 2312.00752; SSD/Mamba-2 — ids unverified). **Common gap:** spectral characterization here is of an *architecturally explicit* linear state operator (linear attention / SSM), not the effective h_t→h_{t+1} of a softmax-attention residual stream. This is the bucket that has genuinely *characterized* per-token operators — but only where the architecture hands you one for free.

- **(f) Directly computing rank/spectrum/linearity of an h_t→h_{t+1} map in a trained transformer.** No verified paper does this for the residual stream of a standard softmax transformer. Nearest partial hits: State Rank Dynamics **2602.02195** (rank/spectrum, but linear-attention state), Local Linearity/LTV **2604.19018** (Jacobian operator, but depth axis), Relational Rank Geometry **2605.29634** (rank geometry, but token-tuple relations at a position). **Gap confirmed.**

## (3) Closeness verdict

**Single closest work:** *A Statistical Physics of Language Model Reasoning* (**2506.04374**), which fits a switching linear dynamical system to hidden-state trajectories along the generation axis and characterizes them via a rank-40 projection and four latent regimes. **Gap:** it works at sentence granularity on a low-rank projection and delivers a phenomenological drift-diffusion + SLDS with discovered regimes — not the rank, spectrum, or linearity score of an explicit position-*t*→position-*t+1* activation-transition operator, and not on a same-layer residual stream at token resolution. The context-summary→answer-summary map is a *summary→summary* operator, which is even more specific than the adjacent-token map and is untouched by any of these.

**Honest answer to "has anyone CHARACTERIZED (rank/spectrum/operator structure, not just trained-for-speed) the token-axis activation→activation map in LLMs?"** — **Essentially none found for the exact object.** The pieces exist but nobody assembles them: (i) EAGLE *learns* the map for speed and only qualitatively remarks that features are "more regular than tokens" (2401.15077) and that feature-prediction is a scaling "constraint" (2503.01840) — no rank/spectrum. (ii) SLDS/DMET (2506.04374, 2505.20340) fit dynamical operators to *sentence/manifold-level* generation trajectories and characterize regimes/geometry, but not a token-level residual-stream transition operator's spectrum. (iii) The one place a per-token operator's **rank/spectrum is actually computed** is linear-attention/SSM state matrices (2602.02195), which is the wrong architecture (not softmax-transformer residual stream). (iv) The one place an activation-transition **operator (Jacobian/LTV) is fit and characterized** is the **depth axis** within a forward pass (2604.19018), not the sequence axis. So: characterizing the rank/spectrum/linearity of the *same-layer, token-axis* h_t→h_{t+1}/h_{t+k} residual-stream map in a standard transformer (Qwen-2.5-7B) — as an operator, treated as the object of study — appears to be an **open gap**, and the summary→summary variant more so.

**Verification note:** 16 IDs verified directly against arxiv.org abstract pages (2506.04374, 2604.19018, 2505.20340, 2602.02195, 2401.15077, 2503.01840, 2507.11851, 2511.00180, 2503.21801, 2510.14751, 2602.18679, 2405.15943, 2402.04161, 2502.15208, 2605.29634, 2601.20164, 2506.01034; plus 2309.01245 confirmed via search). IDs explicitly flagged "unverified" (EAGLE-2 2406.16858, induction-head-Markov 2402.11004/2410.05493, Transformers-are-RNNs 2006.16236, Mamba 2312.00752, Hydra/GLIDE/Medusa) were not abstract-fetched — treat those ids as provisional.

---

# Appendix G — Prior art: layer-axis activation→activation maps (agent report, verbatim)

# Prior art: characterizing layer→layer (depth-axis) activation maps

**Target object for delta comparison:** a *fitted* (population-regression) linear map from the activation at layer ℓ to the activation at layer ℓ+k, *same token position*, characterized **as an operator** (rank, singular/eigen-spectrum, invariant subspaces), at a **span-pooled / summary grain**, with **estimation-noise discipline** on the fitted spectrum.

**Verification key:** ✅MCP = abstract confirmed via arXiv MCP this session; 🌐WEB = arXiv abs page + title confirmed via web (MCP was 429-throttled for much of the run). No IDs were fabricated; every 🌐WEB item resolved to a live `arxiv.org/abs/` page. The 2602–2606 IDs are 2026 papers (this environment is dated July 2026) — resolvable but not MCP-confirmed, so marked 🌐WEB.

## (1) Ranked closest-prior-work list

**1. "Your Transformer is Secretly Linear" — Razzhigaev, Mikhalchuk, Goncharova, Gerasimenko, Oseledets, Dimitrov, Kuznetsov. arXiv 2405.12250 (ACL 2024). ✅MCP**
Fits a map between *sequential layer embeddings, same position* — exactly the depth axis — and characterizes it with a **Procrustes similarity score of 0.99**. Closest work on the "fit a same-position layer→layer map and measure it" axis.
**DELTA:** Procrustes is an *orthogonal* (rotation-only) fit → recovers neither rank, nor a full singular/eigen-spectrum, nor invariant subspaces of a *general* linear operator; collapses to one scalar similarity. Grain is per-token embedding, not span-pooled. They themselves report the near-linearity is **residual-dominated** ("linearity decreases when the residual component is removed due to a consistently low output norm") — i.e. the trivially-expected identity-plus-small-update structure. No estimation-noise treatment of a fitted regression map.

**2. "Equivalent Linear Mappings of Large Language Models" — James R. Golden (sole author). arXiv 2505.24293. ✅MCP** (HF mirror titled "Large Language Models are Locally Linear Mappings")
Closest work on the **operator-level spectrum/rank/subspace** axis. Detaches the input-dependent `A(x)` terms so the Jacobian becomes an *exactly equivalent* linear operator (reconstruction rel-error <10⁻¹³); does **SVD of this detached Jacobian**, shows LLMs "operate in extremely low-dimensional subspaces where the singular vectors decode to interpretable concepts," **per layer and per module (attn/MLP)**, on Qwen 3 / Gemma 3 / Llama 3 (up to Qwen 3 14B) — same family as the project's Qwen-2.5-7B.
**DELTA:** this is the *local, per-input, exact* linearization (detached Jacobian), **not a fitted population-regression map** → no estimation-noise regime, no across-examples generalization. Layer→output-embedding / per-module, per-token — not an arbitrary ℓ→ℓ+k summary-grain map. But it is the field's clearest precedent for reading rank/spectrum/singular-vectors off a linear operator inside a trained LLM.

**3. "Eliciting Latent Predictions from Transformers with the Tuned Lens" — Belrose, Furman, et al. arXiv 2303.08112. 🌐WEB** (known)
Fits a **per-layer affine map** (translator `A_ℓ h_ℓ + b_ℓ`), same position — a fitted, learned, per-layer linear+bias operator.
**DELTA:** codomain is the *logits* (composed with the unembedding to a token distribution), not the next layer's activation; the affine map is a *means to decode*, never characterized as an operator (no rank/spectrum/invariant-subspace read). Per-token.

**4. "Jump to Conclusions: Short-Cutting Transformers With Linear Transformations" — Yom Din, Karidi, Choshen, Geva. arXiv 2303.09435. 🌐WEB**
Fits **linear transformations casting a hidden representation directly to the final-layer representation** (a learned layer→final linear map), across model scales.
**DELTA:** layer→*final* (a shortcut to the head), not general ℓ→ℓ+k; purpose is prediction accuracy / peeking, not operator characterization; per-token.

**5. "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State" — Pal, Sun, Yuan, Wallace, Bau. arXiv 2311.04897 (CoNLL 2023). 🌐WEB**
Trains a **linear model predicting future hidden states** (and tokens) from one hidden state — a fitted hidden→hidden linear map.
**DELTA:** the axis is *token position* (predict state at position ≥ t+2), not depth at fixed position; evaluated by decode accuracy, not spectrum/rank; per-token.

**6. "Linearity of Relation Decoding in Transformer Language Models" — Hernandez et al. arXiv 2308.09124 (ICLR 2024). 🌐WEB**
Fits an **affine operator `R(s)=W s + b`** (a first-order Jacobian approximation from a single prompt) mapping a subject representation to the relation's object prediction — a fitted linear map spanning layers.
**DELTA:** relation-specific and prompt-local (one `W` per relation), not a generic ℓ→ℓ+k map; characterized by faithfulness + a low-rank inverse ("attribute lens"), which touches rank but not spectrum/invariant-subspaces as the object; per-token subject rep.

**7. "Transformer Layers as Painters" — Sun, Pickett, Nain, Jones. arXiv 2407.09298 (AAAI 2025). ✅MCP**
Middle layers are near-interchangeable: skippable, reorderable, runnable in parallel with graceful degradation — strong evidence the layer map is close to identity / permutation-invariant in the middle band.
**DELTA:** measures *behavioral robustness to reordering*; never fits a map or reads a spectrum. Positions the near-identity assumption, doesn't characterize the operator.

**8. "The Unreasonable Ineffectiveness of the Deeper Layers" — Gromov, Tirumala, Shapourian, Glorioso, Roberts. arXiv 2403.17887 (ICLR 2025). ✅MCP**
Picks prunable blocks by **angular distance between layer ℓ and ℓ+n representations** (near-identity ⇒ removable), heals with QLoRA; up to ~half the layers removable.
**DELTA:** a scalar similarity (angle) used for pruning under a near-identity assumption; no fitted map, no spectrum/rank.

**9. "ShortGPT: Layers in LLMs are More Redundant Than You Expect" — Men et al. (arXiv 2403.03853). 🌐WEB**
**Block Influence = 1 − cos(input, output)** per layer; low-BI layers pruned.
**DELTA:** scalar redundancy metric, near-identity framing; no operator characterization.

**10. "The Remarkable Robustness of LLMs: Stages of Inference?" — Lad, Lee, Gurnee, Tegmark. arXiv 2406.19384. ✅MCP**
Layer delete/swap interventions → the four depth-stages taxonomy (detokenization / feature engineering / prediction ensembling / residual sharpening).
**DELTA:** intervention-based depth-*dynamics* characterization, not a fitted map or its spectrum. Best cite for the "what the depth axis is doing" framing.

**11. "Sparse Crosscoders for Cross-Layer Features and Model Diffing" — Lindsey, Templeton, et al. Transformer Circuits, Oct 2024 (transformer-circuits.pub/2024/crosscoders). 🌐WEB (gray lit)**
A dictionary that *reads and writes multiple layers at once*, tracking feature persistence/evolution through the residual stream across depth.
**DELTA:** a *sparse-feature* cross-layer lens (resolves cross-layer superposition, "jumps" identity connections), not a dense linear operator's rank/spectrum; complementary framing of cross-layer structure.

**12. "Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints" — Saurez, Lee, Har. arXiv 2602.09783 (submitted ICML 2026). 🌐WEB**
Closest on the **"invariant subspaces"** keyword and the *why-linearity-is-expected* question: an "Invariant Subspace Necessity" theorem — features decoded through linear interfaces (OV circuits, unembedding) must occupy context-invariant linear subspaces.
**DELTA:** about *feature* subspaces decoded through linear interfaces, theoretical; not the invariant subspaces of a *fitted ℓ→ℓ+k map*. But it is the most direct published argument for why your object should have low-rank invariant structure by construction.

**13. Local-Jacobian spectra across depth — "Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment," arXiv 2602.12384 (ICML 2026 poster) 🌐WEB; and "Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology," arXiv 2605.14258 🌐WEB.**
Operator-level **singular-value spectra / eigenstructure of the Jacobian across depth**: depth-induced exponential SV scaling, effective low-rank, later layers lower-rank ("representational contraction at depth").
**DELTA:** the *local Jacobian* spectrum (analysis/theory of the block linearization), not a fitted population regression map; no summary grain, no fit-noise question.

**14. "How Transformers Reject Wrong Answers: Rotational Dynamics of Factual Constraint Processing." arXiv 2603.13259. 🌐WEB**
Reports depth updates as **rotation on an approximately constant-norm manifold** (norm ratios within 3% of unity, cosine drops to ~0.62–0.69), and explicitly notes **complex-conjugate eigenvalue pairs = 2-D rotate-and-scale subspaces "invisible to SVD."**
**DELTA:** trajectory-dynamics of activations, not a fitted map's operator characterization — but a **direct methodological warning** for your spectrum read: an SVD of the layer map will miss rotational (complex-eigenvalue) structure; use an eigendecomposition/Schur view too.

**15. Trajectory / representation geometry across depth — "Trajectory Geometry of Transformer Representations Across Layers," arXiv 2606.09287 🌐WEB; Valeriani et al., "The Geometry of Hidden Representations of Large Transformer Models," NeurIPS 2023 🌐WEB.**
Depth trajectory metrics (length, curvature, layerwise cosine, intrinsic dimension across layers).
**DELTA:** geometry of the *trajectory / representations*, not the map operator between them.

**16. Model-stitching / cross-layer & cross-model linear alignment — "Transferring Linear Features Across Language Models With Model Stitching" (OpenReview Qvvy0X63Fv) 🌐WEB; "Characterizing Linear Alignment Across Language Models," arXiv 2603.18908 🌐WEB.**
Fit affine "translators" between residual streams; the OpenReview note explicitly says stitching "works for different layers inside a single model as well."
**DELTA:** the object is *cross-model* (or cross-layer as a byproduct) alignment *quality*, low-rank-least-squares vs task-loss stitching — not a within-model ℓ→ℓ+k operator's spectrum/invariant subspaces at summary grain.

**Folklore basis (position against, do NOT treat as discoveries):** residual-stream norm growth — **Heimersheim & Turner, "Residual stream norms grow exponentially over the forward pass," LessWrong/Alignment Forum, May 2023** (🌐WEB gray lit); iterative inference — **Jastrzębski et al., "Residual Connections Encourage Iterative Inference," ICLR 2018** (🌐WEB). Together these are *why* ℓ→ℓ+1 ≈ identity + small update, hence why a fitted linear map is near-identity and near-linearity is trivially expected. The Secretly Linear paper's residual-removal ablation is the empirical confirmation of exactly this point.

## (2) Near-miss taxonomy per bucket

**(a) Secretly-linear / linearity-score family** — 2405.12250 (Procrustes 0.99), plus "A Primer on the Inner Workings of Transformer-based LMs" (2405.00208, survey context). **Who flagged that the residual makes near-linearity trivial:** the Secretly Linear authors themselves (residual-removal ablation), and structurally the iterative-inference (Jastrzębski) + norm-growth (Heimersheim & Turner) literature. I found **no standalone LessWrong/AF post whose thesis is a critique of that paper as trivial** — stated as a "none found" (the point is folklore, absorbed into the residual-stream framing rather than a single rebuttal post).

**(b) Lens family beyond tuned lens** — logit lens (nostalgebraist, LessWrong 2020, gray lit); Tuned Lens 2303.08112 (per-layer affine → logits); Jump to Conclusions 2303.09435 (layer→final linear); Future Lens 2311.04897 (hidden→future-hidden linear); Logit Prisms (neuralblog.github.io/logit-prisms, gray lit); "Analyzing Transformers in Embedding Space" 2209.02535. **All map to the vocabulary/logit space or to the final representation — none fits and characterizes a general ℓ→ℓ+k activation operator.**

**(c) Layer redundancy / interchangeability** — Unreasonable Ineffectiveness 2403.17887 (angular distance); ShortGPT 2403.03853 (Block Influence, 1−cos); Painters 2407.09298 (reorder/parallel); adjacent: "Ghosted Layers" (2605.15491 🌐WEB), "The Curse of Depth in LLMs" (2502.05795 🌐WEB), "Inverse Depth Scaling From Most Layers Being Similar" (2602.05970 🌐WEB). **What they measure about the layer map:** cosine/angular *similarity* or behavioral robustness — never rank/spectrum/linearity of a fitted operator.

**(d) Depth-dynamics characterization** — Stages of Inference 2406.19384; residual norm growth (Heimersheim & Turner 2023); iterative refinement (Jastrzębski 2018; Loop-Residual 2409.14199); rotation-vs-scaling (2603.13259); trajectory geometry (2606.09287; Valeriani NeurIPS 2023); feature evolution via crosscoders (Lindsey et al. 2024). **Characterize norm/angle/curvature/intrinsic-dim trajectories and depth stages — not the map operator.**

**(f) Rank / spectrum / eigenstructure of a fitted layer-map in a trained transformer** — **the sparsest bucket.** Operator-level SVD/rank exists only for the *local exact Jacobian*: 2505.24293, 2602.12384, 2605.14258, with 2603.13259 warning that SVD misses complex-eigenvalue rotation. Fit-quality-with-rank appears in stitching (OpenReview Qvvy0X63Fv) and LRE's low-rank inverse (2308.09124). **No work fits a population ℓ→ℓ+k regression map and reports its rank/spectrum/invariant subspaces with estimation-noise discipline. Span-pooled / summary-grain layer→layer maps: none found.**

## (3) Closeness verdict

**Single closest work — a tie between two, neither of which does your exact thing:**
- **2405.12250 (Secretly Linear)** is closest in *object* — it fits a same-position sequential ℓ→ℓ+1 map — but characterizes it with only a scalar orthogonal-Procrustes similarity, so it never recovers rank, full spectrum, or invariant subspaces, and its "linearity" is admittedly residual-dominated (trivially expected).
- **2505.24293 (Equivalent Linear Mappings, Golden)** is closest in *method* — it reads rank/singular-vectors/low-dim subspaces off a linear operator inside an LLM, per layer, on the Qwen family — but that operator is the *exact local detached Jacobian per input*, not a *fitted population regression map*, so there is no across-examples estimation-noise regime and no summary/pooled grain.

**Gap statement:** No published work fits a *population-regression* linear map along the depth axis (activation ℓ → activation ℓ+k, same position) and characterizes it *as an operator* — rank, singular/eigen-spectrum, invariant subspaces — **at a span-pooled / summary grain, with discipline about the noise a finite-sample fit injects into that spectrum.** The adjacent axes are individually well-trodden (the lens family, layer-redundancy/pruning, depth-dynamics, and local-Jacobian spectra), but this specific intersection is open.

**How crowded is the space, honestly:** the *layer-axis map* space is **crowded on the periphery and empty at the center you're aiming for.** There are many same-position layer-map artifacts (lenses, stitching translators, Procrustes fits) and many depth-similarity/redundancy metrics, so a reviewer will say "layer maps are well-studied" — pre-empt that. But the **operator-level characterization of a *fitted* map (spectrum/rank/invariant subspaces + estimation-noise control), and the span-pooled/summary grain, are genuinely un-done** — the only operator-level spectra published are of the *exact local Jacobian* (2505.24293 and the 2602/2605 Jacobian-spectra papers), not of a learned population map, and the only fitted same-position layer→layer map that has been "characterized" (2405.12250) was reduced to a single Procrustes scalar. Two methodological cautions inherited from the crowd: (i) the residual stream makes a near-identity linear fit and high linearity trivially expected (Secretly Linear's own ablation; Heimersheim & Turner; Jastrzębski) — frame the operator characterization *against* the identity baseline, i.e. characterize the *update / deviation-from-identity* operator, not the raw map; and (ii) an SVD will miss rotational structure carried by complex-eigenvalue pairs (2603.13259), so read the eigen/Schur spectrum, not just singular values.

---
---

# Round 3 (2026-07-28): null spaces of W — the two reads, and what current fits support

Chat-session consolidation (2026-07-28). All quoted numbers below were re-verified against the named artifacts at compose time (`eval_results/issue_1092/p7/read2_grain_rank.json`; `eval_results/issue_722/structural/rank_structure.json`; `eval_results/issue_1310/`; `docs/theory_assumption_test_plan.md:162`).

Short answer: there are two null spaces and they buy different things — the input-side kernel tells you which context differences are provably invisible to the answer, the output-side co-kernel gives a hard ceiling on pre-generation predictability. But with the fits banked today, almost everything you'd call a "null space" belongs to the ridge estimator, not to the model.

## The two reads

**Right null space (kernel, input side)** — context directions that produce zero predicted answer movement. This is the "what does the answer discard" read, and it's the one that pays the leakage program: if a persona or trait direction on the context side sits in the kernel, the theory predicts no propagation to the answer from that direction. That's a negative prediction, which is cheap to falsify and hard to get by accident, so it's worth much more than another R². It's also directly causal-testable in both directions — LEACE-erase a kernel direction from the context state and behavior should not move; inject along one and nothing should happen (which doubles as a null-intervention control validating the steering rig), while the same operations on the top singular directions should move behavior.

**Left null space (co-kernel, output side)** — answer directions unreachable from the context. This is the ceiling: project each trait read-out r_B onto range(M_{C,A}) versus its complement, and the complement fraction is a per-behavior upper bound on how well any linear context-side monitor can ever do. That speaks straight to the paper's framing question ("exactly how much of the answer can be predicted from the context") as a bound rather than a fitted number. The sharper version is in the round-1 synthesis above: the top centered-whitened singular value squared is the linear maximal correlation, an information-theoretic cap on context→answer transfer per layer.

Because the map is roughly square (3584→3584) and non-normal, these two subspaces are genuinely different objects — you can't read one off the other.

## Why the current fits can't deliver either yet

For ridge, β = Xᵀ(XXᵀ+λI)⁻¹Y, so its columns live in the span of the observed context vectors and its rows in the span of the observed answer vectors. With n observed pairs, both the kernel and the co-kernel contain at least d − n dimensions for free — directions nobody ever showed the estimator. Ridge also never zeroes anything exactly; it shrinks, so "the null space" is really a soft small-singular-value tail whose extent is set by λ. Our λ is GCV-chosen and keeps saturating a grid edge (1000 in #1092/#813/#823, 0.01 in #722, 0.001 in #779's n=1M fits), so tail sizes aren't comparable across issues. `docs/theory_assumption_test_plan.md:162` already says this out loud: at d≫n the unregularized null space dominates and prediction-transfer stays the trustworthy primary.

What the measured spectra actually support:

- #1092 is the one legitimate regime — 12,121 fit rows against d=3584. The ambient context→answer maps at layers 14/18/19 are numerically full rank (all 3584 singular values nonzero) but effectively low rank: stable rank 3.7–24, k50 = 10–89, k90 = 355–767, top-50 directions carrying 40–67% of Frobenius energy. So there is no exact kernel — there's a heavy tail of ~3,000 directions carrying under 10% of the map's energy. That's the honest form of "the answer discards most of the context's geometry" (`eval_results/issue_1092/p7/read2_grain_rank.json`).
- The prefix arm's apparent 3,000-dimensional kernel is a corpus artifact. Its ambient maps show only 492–544 nonzero singular values, which looks like a dramatic finding until you notice the corpus crosses just 1,145 distinct prefixes with a 1,397-query bank. The prefix-side design matrix is rank-limited by the number of distinct prefixes, not by the model. Worth flagging before that number gets narrated as a prefix-vs-context rank gap.
- #722 and #813's spectra can't carry a null-space claim at all — participation ratio 16.95, stable rank 3.00, k90 = 19, all from n=50 conditions at d=3584. Their "null space" is 3,534 unobserved directions. (#813 at least ran matched-n control draws, which is the right guard.)
- #1310's lesson generalizes here. Its singular-spectrum cosine came in at 0.9896 against its own shuffled-fit null of 0.9976 — i.e. below the null, no discriminating signal — while direction-aware reads did separate (raw operator cosine 0.2265 vs null −0.0016). A null space is a subspace claim, so it needs the direction-aware treatment plus a matched null, not a description of a singular-value tail.

## What would make it a result

Fit where the number of distinct inputs greatly exceeds d (#779's up-to-1M-row fits and #1092's context arm qualify; the prefix arm needs a wider prefix pool). Then report the held-out predictable-variance spectrum — how many canonical or PLS components have out-of-sample R² above zero under novel-prefix group folds — rather than the fitted-coefficient spectrum, since that's a rank claim robust to λ and immune to the estimator's own null space. Gate it against permuted (context, answer) pairing refits and matched-n draws. Then make it causal with the LEACE erase-and-measure test, because a kernel that isn't causally inert is just a small singular value.

The round-1 menu above already scopes both reads (Q3's nullspace item, and Appendix D item 11 on concept-erasure probing), and the empirical paper's introduction lists "the null space/eigenvectors/fixed points of this mapping" as a planned characterization with `\subsection{What Is This Mapping Bad at Predicting?}` as its natural home — currently a two-bullet stub, so this is an open slot rather than a redo.
