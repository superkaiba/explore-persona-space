# Theoretical analysis plan — the four context→answer maps at the large-n grain

*2026-07-28. Scope: the fitted maps from the #779 (up to 1M rows) and #1092 (21,193-row crossed real-conversation corpus) lines. Grounding: the methods survey (`docs/ideas/2026-07-06-context-answer-map-analyses.md`), the consolidated results (`docs/results_summaries/2026-07-22-prefix-query-context-answer-map-consolidated.md`), and the empirical paper's M_{C,A} object. Analysis plan over banked artifacts plus a small set of new fits; any execution routes through the standard task pipeline.*

## 1. The object: one conditional expectation, four conditionings

Every row has generative structure a = F(p, q, ε): answer summary a from prefix p, user query q, decode noise ε. The four maps estimate conditional expectations of this one object:

| Arm | Input | Estimand | Plain reading |
|---|---|---|---|
| context | c(p,q), pooled prefix+query | E[a \| p, q] | the full-information map |
| prefix-end | e(p), last prefix token (pre-query) | E[a \| p] | query-marginalized: what the persona alone fixes |
| bare query | v(q), query with no prefix | E[a \| q] | prefix-marginalized: what the task alone fixes |
| query-averaged | v̄_P = mean_q c(p,q) | E[a \| p] | same estimand as prefix-end, richer input |

**The deviation from the snapshot literature.** DMD/Koopman fits assume an autonomous system: the state at t determines the distribution at t+1. Here an exogenous input — the user turn — enters between input and output. The correct template is DMD *with control*: a = A·p + B·q + interaction + noise (Proctor–Brunton–Kutz DMDc; verify arXiv id before paper use). The estimator-level toolkit (ResDMD certification, bagged CIs, constrained fits) imports unchanged because it is about regression operators, not physics; the *dynamical* readings (iterating the map, attractors, mixing times) do not. The conditional-expectation framing (Grünewälder 1205.4656; Mollenhauer & Koltai 2012.12917; Kostic 2302.02004) is what absorbs the user turn cleanly: for the prefix arms the query is marginalized, so it is structured noise with three knowable consequences — an R² ceiling equal to the between-prefix variance share, attenuation of the fitted operator, and anisotropic residuals that break vanilla Marchenko–Pastur nulls. Each consequence is a testable prediction, not a nuisance.

**Measured facts the theory must explain** (consolidated writeup, all held-out, novel-prefix folds):
1. Context map R² 0.74–0.81 per-context; prefix-end R² 0.07–0.10 per-context, 0.35–0.58 of its own ceiling even on averaged targets.
2. Additivity to 91%: disjoint prefix + bare-query forwards stitch to R² 0.833 vs 0.910 full.
3. Variance shares: query 79% / prefix 11% / interaction 10%.
4. The prefix and context operators share output subspaces (angles 22–34° vs ~78° null) but read near-orthogonal inputs; the prefix arm's predictions are a global ~0.6× shrinkage of the context arm's.
5. The independently-fit averaged map is a noisier estimate of the context map, not a second mechanism (operator-coincidence check).

## 2. Q1 — Are the four maps one object? (algebraic consistency)

Three relations the one-object view predicts, all computable from banked stores:

- **Jensen/commutativity gap.** For a linear true map, averaging inputs then mapping equals mapping then averaging: M_ctx(v̄_P) = mean_q M_ctx(c(p,q)). Compute both sides per prefix; the gap is a direct curvature witness — how nonlinear the true map is across each prefix's query cloud — obtained with zero new fits. *Interprets as:* where (which prefixes, which answer directions) the linear abstraction bends.
- **Factorization of the prefix-end deficit.** Fact 1 says e(p) reaches only 0.35–0.58 of ceiling. Two rival explanations: the pre-query state does not *contain* the information (state deficit), or it contains it in a form the linear readout misses (readout deficit). Fit the intermediate map e(p) → v̄_P and chain it: if R²(e→v̄_P→a) ≈ R²(e→a), the deficit is in the state; if chaining through v̄_P recovers skill, the information is present but linearly inaccessible from e. *Interprets as:* locating WHERE persona information is lost before the query arrives — the single most monitoring-relevant unknown in the line.
- **Marginal vs joint prefix operator.** In the crossed design p ⊥ q, so the prefix block A of the joint (stitch) fit and the marginal prefix-end map should coincide (omitted-variable term B·Cov(q,p) vanishes). Compare them. *Interprets as:* a validity check on the crossing — divergence measures confounding from sparse-crossing imbalance, which would contaminate every cross-arm comparison.

## 3. Q2 — What does the user turn do? (the DMDc layer)

- **Low-rank bilinear interaction.** Extend the additive stitch with a rank-r interaction term: a = A·p + B·q + Σᵢ (uᵢᵀp)(vᵢᵀq)·wᵢ, sweeping r. Fact 2 caps the interaction at ~0.08 R²; the question is its *shape*. *Interprets as:* how many degrees of freedom of prefix-modulation the query has — r=1 means the query is a single gain knob on the persona; small r means a few named modulation channels (inspect uᵢ, vᵢ, wᵢ against the trait dictionary); diffuse means genuine mixing only attention explains.
- **Where the interaction lives.** Project the interaction component onto the persona-vector dictionary and answer PCs. *Interprets as:* whether prefix×query interaction is persona-bearing (bad news for linear monitors: persona expression is query-gated) or stylistic residue.

## 4. Q3 — Channel structure per arm

For each of the four arms, on the same folds and targets:

- **Held-out predictable-variance spectrum** (whitened; number of canonical/PLS components with out-of-sample R² > 0). *Interprets as:* the honest channel count — how many independent things each input determines about the answer. Robust to λ and to the estimator's mechanical rank cap.
- **Linear maximal-correlation ceiling ρ₁² per layer.** *Interprets as:* the hard cap on any linear readout of the answer from that arm's input, comparable across arms and layers; the arm-to-arm ρ₁² gap is "how much the query unlocks," as a bound rather than a fit.
- **Per-trait-direction held-out R² table** (project true and predicted answers onto each persona/trait direction; directions fixed on train folds). *Interprets as:* which behaviors ride which arm — a trait predictable from prefix-end is pre-query monitorable; a trait predictable only from context requires seeing the query; a trait predictable from neither is (linearly) spontaneous. This is the leakage-relevant deliverable.
- **Cross-arm subspace geometry.** Principal angles between arms' predictive input and output subspaces (extends Fact 4 to all four arms, with spectrum-matched nulls). *Interprets as:* whether the query adds new channels or reweights existing ones — Fact 4 says reweighting for prefix-vs-context; test whether bare-query behaves the same.

## 5. Q4 — Ceilings and null spaces

Import of the Round-3 null-space program, now legitimate on the n ≫ d arms (context, bare-query; the prefix arm stays rank-limited by 1,145 distinct prefixes — widen the pool before any prefix-side kernel claim):

- **Co-kernel ceiling per behavior:** fraction of each trait read-out r_B outside range(M), per arm. *Interprets as:* a per-behavior upper bound on any linear context-side monitor; the cross-arm differences say how much of each behavior's expression is query-gated.
- **Kernel causal tests:** LEACE-erase or inject along candidate kernel directions (predicted: nothing happens — the null-intervention control) vs top singular directions (predicted: behavior moves). *Interprets as:* upgrades "small singular value" to "causally inert direction" — the cheap-to-falsify negative prediction that distinguishes a mechanism claim from a fit description.

## 6. Q5 — Where linearity ends (kernel and MLP)

- **Detection before estimation.** HSIC / distance correlation between each arm's input and its linear-map residuals, group-respecting permutation p-values. *Interprets as:* is there ANY structure the linear map missed — answered exactly, before spending on nonlinear estimators.
- **Estimation ladder per arm**, identical folds and nested-CV tuning on every rung: ridge → random-features/Nyström kernel ridge (exact kernel methods do not scale past n ~ 10⁴; RFF/Nyström do) → MLP. Note the regime reversal: at n = 21K–1M, MLPs are legitimate estimators (unlike the n≈2.5K per-example grain, where optimizer variance dominates); report seed spread. The **nonlinearity gain** per arm = held-out R²(nonlinear) − R²(linear), both against the identity+bias baseline and kNN-retrieval reads per the standing mapping rules. *Interprets as:* (a) which of the four relations are genuinely linear vs linear-only-as-approximation; (b) the prefix-arm gain is the monitoring number — persona→answer transport that NO linear probe can see; (c) if the MLP's gain over linear-on-context is closed by the rank-r bilinear model of Q2, the "nonlinearity" is *named* — it is the prefix×query interaction, not a black box.
- **Decode-noise ceiling.** K ≥ 5 answer draws per context on a subset; per-direction noise floor. *Interprets as:* separates "the map fails here" from "nothing could predict this" — required before reading any per-direction R² table (Q3) or nonlinearity gain as a map property.

## 7. Validity gates (binding on every read above)

Group-level folds by prefix id; λ discipline (df(λ) reported, headline reads at ~3 λ values, same λ refit inside every permutation draw); permutation + matched-n nulls for every spectrum and subspace overlap; non-normality gate before any eigen-read; identity+bias baseline + kNN retrieval for every fitted map; directions selected on train folds only; n_train vs d stated per fit. Full text: methods survey §"Validity gates".

## 8. Compute

Mostly 0-GPU on banked #1092/#779 stores (Q1, Q3 spectra + trait tables, Q4 co-kernel, Q2 bilinear fit — CPU, batched per the vectorize-first rule). New compute: multi-draw answers for the noise ceiling (small GPU); RFF/MLP ladder (cheap GPU band); kernel causal tests (GPU, steering rig). Each execution round is a task through the standard pipeline; nothing here pre-authorizes a run.

## 9. Summary: computation → what it tells us

| Computation | What it tells us about the mapping |
|---|---|
| Jensen gap (avg-then-map vs map-then-avg) | curvature of the true map over query clouds — where linearity bends |
| e(p)→v̄_P chain | whether persona information is absent from the pre-query state or just linearly unreadable |
| Joint-vs-marginal prefix operator | whether the crossed design cleanly identifies the prefix's own effect |
| Rank-r bilinear interaction | how the user turn modulates the persona: gain knob, few channels, or diffuse mixing |
| Predictable-variance spectra (4 arms) | honest channel count per input |
| ρ₁² per arm/layer | hard linear-monitor ceiling; what the query unlocks, as a bound |
| Per-trait R² table (4 arms) | which behaviors are pre-query monitorable, query-gated, or spontaneous |
| Cross-arm principal angles | query adds channels vs reweights them |
| Co-kernel fraction per trait | per-behavior monitor ceiling from unreachable answer directions |
| LEACE/injection on kernel vs top directions | which directions are causally inert vs load-bearing |
| Residual HSIC/dCor | whether anything nonlinear remains, exactly |
| RFF/MLP nonlinearity gain per arm | the persona transport invisible to any linear probe; whether nonlinearity = the named interaction |
| Multi-draw noise ceiling | what no map could ever predict |
