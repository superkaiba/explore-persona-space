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
