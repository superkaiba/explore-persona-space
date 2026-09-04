# Theoretical analysis of the context→answer map: compiled report

*2026-09-02, revised the same day to add the full eigenvector, null-space, and inverse-map record, on 2026-09-03 with the whitened eigen dashboards and null-space interpretation, and on 2026-09-04 with the variance decomposition and PCA/SAE basis views. Compiled from the task clean-results, the docs listed in §7, the June leakage-theory paper, and Thomas's Obsidian notes. Numbers are quoted from the cited artifacts; nothing here is a new measurement.*

## 1. Summary

- **The map is one high-rank linear operator, and we can predict its own learning curve from first principles.** The closed-form ridge learning curve matches the measured curve to within 0.006 R² over two decades of training size, and the 963k-row map sits at 99% of the population linear ceiling (#2569). Feeding prefix and query as two separate inputs loses R² to their interaction, and a rank-32 bilinear term recovers 93% of that loss (#1775). The single-draw answer-state variance is now split into 72.6% linear signal, 9.22% sampling noise, no resolved whole-context contribution beyond the last prompt-token state, and about 18.2% nonlinear or otherwise unread linear-state signal (leg 10, 2026-09-04). An MLP recovers 5.6–5.9 percentage points beyond ridge on its pinned evaluation, but local kNN does not beat the linear map.
- **The operator has been dissected four times and has no compact structure.** Its top directions are not feature-shaped once the answer covariance is divided out: raw cosines of 0.3–0.6 between write directions and answer-SAE features fall to the null floor under whitening (2026-09-03 rerun). Half of the context directions are effectively ignored and carry 1% of the map's energy; the other half are rotated and rescaled; nothing is copied through. The spectrum is 98% complex, strongly non-normal, and the fixed point is a dense, unreadable state (#1092, #1774, #779, #2569). Thomas's own reading in Obsidian: "Understand structure of mapping: I think done, not really any structure."
- **The null space now has a reading, and it is unproven as mechanism.** The map discards 83% of real context variance in 55% of its raw-coordinate read directions (1.5× what a random split would discard); the standardized-coordinate PCA gives the same 83.3% answer with a 59% geometric kernel, so the result is not a units artifact. What it discards is what the conversation is about and how it is dressed: topic wording, boilerplate mass, formality and length. What it reads is which language, which reply template, and the safety register. The persona directions themselves sit mostly in the ignored half (kernel share 0.71–0.83 against a random 0.55) (2026-09-03 run). Context pairs that differ only along low-gain directions produce closer answers (0.61× controls), and 19–27% of each trait direction lies outside the map's range. But every causal test of "the map ignores this" is either under-dosed or negative, and the worst-predicted answer directions sit in the map's high-gain subspace, so the null space does not explain what the map gets wrong (#1774, #1482, #2569). Refusal context differences (one-word harmful swaps, China vs another country) sit in the ignored half at the same rate as random context pairs (0.81 vs 0.81), yet the 14 to 19% the map keeps reproduces the answer shift (cosine 0.80), ranks refusal rates (Spearman 0.77), and transfers to the China questions without refitting (leg 9, 2026-09-03).
- **Inverting the map is the wrong way to go back to the context.** The pseudoinverse pre-image of a persona vector picks sensible contexts (#1615) but cannot steer at the context vector (#2254, #2225) and is a poor context predictor (R² 0.14). A directly fitted answer→context map recovers held-out contexts at R² 0.75 and points in a different direction (cosine 0.3–0.4) (#2618).
- **The map describes a correlation. It is not the causal mechanism.** Jacobians of the true forward map recover none of its predictive power and full-state substitution at the map's input slot moves nothing (#1776).
- **The bridge to the leakage theory is weak in behavior space and real in activation space.** No gate metric, including the map's own Gram matrix, separates from its panel (#2569 leg 2). The coherence condition holds on constructed contexts and on natural data only under the whitened metric (#658, #1092).
- **The map is the same object across settings up to a change of coordinates, and only partly shared across model families** (#825, #1345, #1639, #1902, #2569 leg 7).

## 2. The object

| Term | Meaning |
|---|---|
| Context vector v_C | Residual-stream state at the last prompt token (end of the user turn), layer 19 of Qwen2.5-7B-Instruct unless stated. |
| Answer vector v_A | Mean residual-stream state over the answer tokens, same layer. |
| The map W (also A, M) | Ridge regression v_A ≈ W v_C + b, fit on 963,444 real LMSYS + WildChat conversations (#779). Held-out R² 0.75 single-draw, 0.80 with five-draw targets. Smaller fits: #1092 (12k rows, layers 14/18/19), #1774 (17k rows, layer 14). |
| Endomorphism | W maps a space to itself (same layer, same width 3,584), so eigenvalues, a fixed point x* = (I − W)⁻¹ b, and iterating the map are all defined. |
| Spectral radius ρ | Largest eigenvalue magnitude. ρ < 1 means iterating the map converges to the fixed point. |
| Non-normal | Eigenvectors far from orthogonal (condition number κ(V) in the thousands). Singular values then overstate per-step growth, and eigen reads need a gate. |
| Kernel (right null space) | Context directions the map sends to zero: context differences invisible to the predicted answer. Ridge never zeroes anything, so in practice "kernel" means the low-gain tail below a stated threshold. |
| Co-kernel (left null space) | Answer directions the map cannot reach from any context: a ceiling on any linear context-side predictor of that direction. |
| Pre-image, pseudoinverse | M⁺ r_B: the minimum-norm context direction the fitted map sends to a persona direction r_B. Distinct from a separately fitted answer→context regression (#2618). |
| Leakage theory (June paper) | Predictor L̂ = strength × (read-out ᵀ training displacement) × context gate. The context→answer map is its assumption A3 (A4/A5 in the assumption map), and the gate is a whitened context similarity. |

## 3. What we did

| When | Item | Where |
|---|---|---|
| June 2026 | Leakage-theory paper: assumptions A1–A11, the boxed predictor, cosine predictor as the isotropic special case, an assumption ladder from a rank-one key-value memory. | Overleaf `6a2df2d2`, `docs/notes/leakage_model_stepwise.tex` |
| July 6 | Methods survey: Koopman/DMD, transfer operators, reduced-rank regression, non-normality gates, validity gates for operator reads. | `docs/ideas/2026-07-06-context-answer-map-analyses.md` |
| July | Persona-vector pre-image: which contexts does M⁺ r_B pick out? Pseudoinverse direction as a monitoring probe. | #1615, #779 |
| July 15–16 | Next-token affine map: fixed point and slow eigen-shell. | #922 |
| July 22 | Consolidated prefix vs query vs context results; operator principal angles; coherence condition on natural data. | `docs/results_summaries/2026-07-22-...consolidated.md`, #1092, #923, #658 |
| July 28 | Four-arm theoretical analysis plan (Jensen gap, channel counts, kernel, eigen reads, fixed point, nonlinearity ladder) and the round-3 null-space consolidation (kernel vs co-kernel; what ridge fits can and cannot support). | `docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md`, `docs/ideas/2026-07-06-...md` §Round 3 |
| July 29 | #1774 linear operator characterization (channels, co-kernel, eigen reads, fixed point, LEACE erasure), #1775 nonlinear ladder, #1776 Jacobian test. | task bodies |
| July 30 | What the map is bad at predicting: residual SVD, worst directions vs the high-gain subspace, context×direction interaction. | `docs/results_summaries/2026-07-30-what-is-the-map-bad-at-predicting.md`, #1482 |
| Aug 1–19 | #1945 information ceiling, #2091 target averaging, #1895 predictable subspace vs SAE subspace, #1902 post-training operator rank, #2254 and #2225 pre-image steering. | task bodies |
| Aug 24–29 | Theory walk → #2569 eight-leg battery (19 figures) plus cross-model own-answers follow-up; #779 operator dissection and joint embedding round; #2618 fitted reverse map vs pseudoinverse. | #2569, #779, #2618 |
| Sep 3–4 | Data-weighted kernel and refusal-pair reads; four-way answer-state variance decomposition; raw/standardized PCA and exact context-SAE variance accounting. | #2569 legs 8–11 and the branch artifacts in §7 |

## 4. Results by question

### 4.1 Is one linear map the right description? Yes, to within a small named remainder.

| Read | Number | Source |
|---|---|---|
| Closed-form ridge learning curve vs measured, five sizes 4.5k→500k | mean abs gap 0.0058 R², same sign at every point | #2569 leg 2 |
| Held-out R² at n=500k vs population linear ceiling | 0.719 vs 0.726 (99.1%) | #2569 leg 2 |
| Residual error structure across context pairs | peak interaction R² 0.0013 against a 0.10 materiality floor | #1945 |
| Gain from averaging five answer draws | 0.06–0.10 R², of which 0.05–0.095 is target averaging, not the map | #2091 |
| Nonlinear gain over ridge at 963k rows | ≈ +0.05 R²; all nonlinear families converge near 0.81 | #779, #1901 |
| Single-draw answer-state variance carried by the population linear map | 0.726 | #2569 leg 10 |
| Within-prompt sampling variance | 0.0922 (LMSYS/WildChat banks mixed 61:39 to match the 100k sample) | #2569 leg 10 |
| Whole-context contribution beyond the last prompt-token state | unconstrained estimate −0.0029; effectively zero, because the independently transferred sampling term slightly exceeds the nearest-neighbor intercept | #2569 leg 10 |
| Nonlinear/remainder component | 0.1847 under the exact unconstrained identity; 0.1818 after enforcing the theoretical boundary W ≥ 0 | #2569 leg 10 |
| Local kNN vs linear map on the 100k sample | best kNN R² 0.6426 vs banked-map R² 0.7277; no positive kNN lower bound on the nonlinear component | #2569 leg 10 |
| Additive prefix+query linear map vs pooled-context linear map | 0.833 vs 0.914 R² | #1775 |
| Share of that additive gap closed by a rank-32 bilinear prefix×query term | 93% on novel prefixes | #1775 |

Reading: the linear map is the population-optimal linear predictor, but its remaining single-draw error is not all per-pair noise. Sampling different answers to the same prompt accounts for 9.22% of total answer-state variance. The nearest-neighbor extrapolation does not resolve any further contribution from the full text beyond the last prompt-token state: its raw-coordinate intercept is 122.97 [120.71, 125.29] absolute variance units, slightly below the separately transferred sampling term of 126.96, hence the small impossible negative W estimate. Treating that mismatch as a boundary estimate W = 0 leaves 18.18% as nonlinear or otherwise unread signal determined by the last-token state. The independent #1901 MLP recovers 5.6–5.9 percentage points beyond ridge, whereas kNN remains below ridge; neither closes that remainder. Separately, the interaction between prefix and query, when they are fed as two inputs, is low-rank: 32 channels through which the query modulates the prefix's effect recover 93% of what an additive two-input map loses. The pooled context vector already carries that interaction.

### 4.2 Eigenvectors, singular directions, and the fixed point

Four dissections of the operator, on two fits. The small fits (#1092, #1774) used 12k–17k rows at layer 14. The large fit (#779 dissection, #2569 leg 1) used the 963k-row banked maps at layers 14, 19, 26.

**Rank and spectrum**

| Read | L14 | L19 (map layer) | L26 | Source |
|---|---|---|---|---|
| Effective rank (participation ratio), 963k map | 794 | 973 | 1,279 | #779 dissection |
| Rank for 90% / 99% of squared singular mass | 411 / 1,325 | 547 / 1,608 | 794 / 2,000 | #2569 leg 1 |
| Frobenius energy in the top 128 directions | 0.70 | 0.65 | 0.58 | #2569 leg 1 |
| Spectral radius ρ | 1.66 | 1.21 | 0.92 | #2569 leg 1, #779 dissection |
| Fraction of eigenvalues with modulus below 1 | 0.990 | 0.997 | 1.000 | #779 dissection |
| Median eigenvalue modulus | 0.16 | 0.12 | 0.10 | #779 dissection |
| Eigenvector condition number κ(V) | 7,650 | 4,261 | 7,452 | #2569 leg 1 |
| Complex eigenvalue pairs / real eigenvalues at L19 | | 1,751 / 82 | | #2569 leg 1 |
| Direction-aware cosine with the identity map | 0.12 | 0.22 | 0.39 | #779 dissection |
| Best fit W ≈ αI + low-rank at L19: α, residual share | | α = 0.13, residual 98% of Frobenius norm | | #2569 leg 1 |
| Split-half stable singular directions at L19 | | 3,425 of 3,584 | | #2569 leg 1 |

Earlier small-fit reads agree on the shape: #1092 found the layer-14/18/19 maps numerically full-rank but effectively low-rank (stable rank 3.7–24, 90% of energy in 355–767 directions), and #1774 counted 763–2,932 held-out-validated channels depending on convention.

**Direction classes at L19** (#2569 leg 1; each of the 3,584 singular directions labelled by gain and input-output alignment):

| Class | Directions | Share of directions | Share of map energy |
|---|---|---|---|
| Ignored (gain below the kernel threshold) | 1,976 | 55% | 1.0% |
| Rotated and rescaled | 1,602 | 45% | 83% |
| Transcoded (read strongly, written elsewhere) | 6 | 0.2% | 16% |
| Copied (gain ≈ 1, output ≈ input) | 0 | 0 | 0 |
| Damped | 0 | 0 | 0 |

Reading: the map copies nothing through. Almost all of its energy goes into rotating and rescaling about half the context directions; the other half is effectively discarded. The near-zero fraction of real eigenvalues says the same thing: the operator is rotational, not a set of independent gains.

**Trait directions under the map**

| Read | Result | Source |
|---|---|---|
| Three persona directions (evil, sycophancy, hallucination), small fit, L14 | gain 0.60–0.64, cosine with own direction 0.07–0.41, 99.6% of output mass leaves the trait span | #1774 |
| Four trait directions (apathetic, humorous, impolite, optimistic), 963k map, L14 | cosine 0.43–0.69, share in trait span 0.21–0.49 | #779 dissection |
| Same four, 963k map, L19 | cosine 0.75–0.86 (random null 0.22), share in trait span 0.58–0.75 (null 0.03) | #779 dissection |
| Top-20 singular directions, small fit | gain 4.1–10.7 at near-zero alignment with their input | #1774 |
| Top-16/64 singular subspace vs trait span, 963k map | one side aligned (principal cosines 0.7–0.9), the other side not (0.1–0.2) | #779 dissection |
| Top output direction | aligned with the hallucination direction, |cos| 0.67 | #1774 |

Reading: "traits are rotated away" (#1774) is a small-fit, layer-14 result. On the 963k map at the map layer, trait directions pass through mostly preserved in direction. The asymmetry read (trait span aligned with the singular subspace on one side of the map only) is recorded with opposite side labels in the events log and the JSON, which is why #2571 (a row/column-convention check for linear-map plans) was filed. Treat the side as to verify.

**Left and right eigenvectors**

For a non-normal operator the two differ: the right eigenvector v_i is the direction mode i writes into (W v_i = λ_i v_i), the left eigenvector u_i is the direction it reads (u_iᵀ W = λ_i u_iᵀ), and they are biorthogonal (u_iᵀ v_j = δ_ij, W = Σ λ_i v_i u_iᵀ). A direction is "maintained" only if λ_i is real and near 1 AND cos(u_i, v_i) is near 1. Both fits ran the full non-symmetric eigendecomposition with the left eigenvectors taken from the inverse of the right-eigenvector matrix.

| Read | Result | Source |
|---|---|---|
| Biorthogonality check, 963k map L19 | max error 1.8e-11; κ(V) 4,261; 1,751 complex pairs, 82 real eigenvalues | #2569 leg 1 |
| Eigenvalues near 1 (candidates for maintained directions), small fit L14 | 0 of 3,584; 68 near 0; trace/d = 0.036 (mean copying score ≈ 0); positive-real eigenvalue mass 0.56 | #1774 |
| cos(u_i, v_i) over the top-32 singular pairs, 963k map | median 0.26, max 0.36, none above 0.5; top pair 0.084 | #2569 leg 1 |
| Fold stability of the top-64 eigenmodes, small fit | all 64 stable (matched across 6 folds) | #1774 |
| Trait gain matrix G (evil, sycophancy, hallucination in and out), small fit | diagonal 0.25 / 0.08 / 0.13, off-diagonal up to 0.16, 99.6% of energy outside the trait span | #1774 |
| Read side (u) of the top-32 singular directions vs 131,072 per-token context-SAE features | median max cos 0.135, max 0.16, against a null floor of 0.085: no single-feature alignment | #2569 leg 1 dashboards |
| Write side (v) of the top-32 singular directions vs 65,536 turn-averaged answer-SAE features | median max cos 0.31, max 0.63; 32 of 32 above the null floor; each write direction fires about 165 features through the encoder | #2569 leg 1 dashboards |
| Read / write side of the top-32 eigen directions vs the same dictionaries | read median 0.155 (max 0.29); write median 0.26 (max 0.41) | #2569 leg 1 dashboards |
| Feature labels for any of the above | absent (both description sources unavailable at run time) | #2569 leg 1 dashboards |
| Tuned-lens token decode of left and right singular vectors 0–7, small fit | unreadable (code fragments, CJK pieces, punctuation); same for the fixed point | #1774 |
| Slow eigenmodes of the next-token map: subspace and coordinates | the slow subspace (right eigenvectors, modulus above 0.95) holds trait and context content 30–125× above a random-subspace null; the slow coordinates (left eigenvectors) persist along real answers 1.7–2.3× vs null 0.64 | #922 |

Reading: nothing is maintained. No eigenvalue sits near 1, every top read direction is nearly orthogonal to its paired write direction (cos ≤ 0.36), and 98% of the modes are complex pairs, so the map acts as rotations between unrelated directions rather than gains on shared ones. The one asymmetry with content: the map's top output directions line up with individual answer-SAE features (cos up to 0.63) while its top input directions line up with no single context feature. Nobody has said what those answer features mean, because the label sources were absent when the dashboards ran. Two caveats bind the eigen dashboards: complex eigenvectors were dashboarded through their real part only, and the median imaginary fraction is 0.65–0.74, so most of each mode was left out; and the whitened-cosine companion was deferred. The only eigenvector read that found legible structure is #922's slow shell on the next-token map.

Convention note: plan v3 of #2569 mixed row and column action on this non-normal operator, so its Gram gate, wiring edges, kernel mining, and gate direction had computed the transpose map's geometry (three probes confirmed, including |cos(u₁, v₁)| = 0.084 vs the expected near-1 for a self-consistent read). Plan v4 fixed the row action before the run. The #779 dissection's singular-subspace read (one side aligned with the trait span at 0.7–0.9, the other at 0.1–0.2) carries the same side-label ambiguity in its events note, and #2571 is the filed convention check.

**Rerun with the imaginary part and whitening (2026-09-03, branch `issue-2569-eigen-v2`)**

Each complex eigenvector pair was read as its real invariant 2-plane instead of its real part alone, and every cosine was recomputed in the whitened metric (angles after dividing out the context or answer covariance, shrinkage 1e-2 with 1e-3 as a check). Null floors are the 95th percentile of random directions or random planes: about 0.08 raw, about 0.10 whitened. A second read-side dictionary was added: the #2569 leg-4 SAE trained on the map's own input states.

| Direction set, dictionary | raw median / max cosine | raw above floor | whitened median / max | whitened above floor |
|---|---|---|---|---|
| Singular read, andyrdt per-token | 0.135 / 0.162 | 30/32 | 0.093 / 0.111 | 6/32 |
| Singular read, trained context SAE | 0.109 / 0.200 | 32/32 | 0.092 / 0.113 | 6/32 |
| Eigen read, andyrdt | 0.150 / 0.304 | 32/32 | 0.109 / 0.154 | 15/32 |
| Eigen read, trained context SAE | 0.154 / 0.284 | 32/32 | 0.118 / 0.176 | 25/32 |
| Singular write, answer SAE | 0.309 / 0.626 | 32/32 | 0.088 / 0.164 | 8/32 |
| Eigen write, answer SAE | 0.289 / 0.418 | 32/32 | 0.115 / 0.196 | 26/32 |

Reading: the imaginary axis carries 38% (read) to 64% (write) of each best match, yet including it raised the maxima by a median of only 0.02–0.04, so the topic labels found earlier stand. Whitening is what changes the story. The singular write directions' matches with single answer features are second-moment structure: they collapse to the floor and the nearest feature changes for 25 of 32 directions, the same variance-driven overlap #1895 reported. Eigen planes keep a modest alignment beyond covariance (write side 26 of 32 above the whitened floor, median 0.115 vs 0.108). The read side is not feature-aligned in any metric with either dictionary, and each read direction fires about 180 features of the grain-matched SAE. Labels: singular write directions match answer-format features (repetitive loops, "acknowledge task and ask for input", greetings, structured documents, step-by-step guides, one-word answers, lists); eigen write directions match topic features (nutrition and chemistry, finance, sports and law). Both readings are raw-coordinate facts.

**Fixed point**

| Read | Result | Source |
|---|---|---|
| Small fit, L14 | ρ = 0.91, so iterating converges; x* norm 45.0 equals the answer-pool median; nearest banked answers cosine 0.66–0.67; token decode unreadable | #1774 |
| 963k map, L14 / L19 | x* exists algebraically (relative residual 1e-14) but ρ > 1, so iterating diverges from it; at L19 ‖x*‖ = 153 and 10,302 of 65,536 context-SAE features fire | #2569 leg 1 |
| 963k map, L26 | ρ = 0.92, the only layer where the iterated-map reading is valid | #2569 leg 1 |
| Next-token affine map (a different object: token state → next token state, per layer) | fixed point is a norm-shrunken mean answer state (cosine 0.985–0.999 at 55–68% of typical norm), decodes to generic answer scaffolding; ρ 0.98–0.99 | #922 |

The one place eigenvectors earned their keep is the next-token map (#922): its slow shell (8–18 modes with |λ| > 0.95, time constants 48–96 tokens) holds up to 50% of the hallucination direction's energy and 32% of the leading between-context direction against a 0.4% random-subspace null, persists along real answers (1.7–2.3× vs null 0.64), and is one subspace through depth (adjacent-layer principal cosines 0.83–0.96 vs null 0.11). Logit-lens decodes of those modes read as language and domain axes, not traits (#1415 Result 6).

One discrepancy to reconcile: #1774 reported ρ = 0.91 at layer 14 (a contraction) on the 17k-row map, while the 963k-row map has ρ = 1.66 at layer 14 in two independent reads. Different corpus, n, and λ; the "stable contraction" headline from #1774 did not carry over to the large map at L14 or L19.

### 4.3 Null space, co-kernel, and what the map gets wrong

**The framing** (round-3 consolidation, 2026-07-28). There are two null spaces. The kernel (input side) says which context differences the answer ignores; a trait direction in the kernel predicts no propagation, which is the negative prediction the leakage program wants. The co-kernel (output side) gives a hard ceiling on any linear context-side monitor of a behavior. Because the map is square and non-normal the two are different objects. The same note lists why most ridge fits cannot support either claim: with n < d rows the estimator has d − n free null dimensions; ridge shrinks rather than zeroes, so the "kernel" is a λ-dependent soft tail; the prefix arm's apparent 3,000-dimensional kernel is a corpus artifact (1,145 distinct prefixes); the n=50 spectra of #722 and #813 cannot carry a null-space claim at all; and #1310 showed a spectrum-shape cosine sitting below its own shuffled null while direction-aware reads separated. Only the n ≫ d fits (#779's 963k rows, #1092's context arm) qualify.

| Read | Number | Source |
|---|---|---|
| Co-kernel: share of each trait direction's answer-side mass outside the map's k90 range | 19–27% (λ-sweep spread 0.06–0.10) | #1774 |
| Kernel-direction context pairs: answer displacement vs matched controls | 0.607× (CI 0.602–0.615); 1.23× the residual floor vs 2.01× for controls; 99.9% of pairs inside the training rows | #2569 leg 8 |
| Kernel steering by addition | dose 0.92 units vs decode noise ≈ 7.8: under-dosed, no verdict | #1774 |
| LEACE erasure of trait directions from the context state | state moved 1.8–3.0× reference; erase-sycophancy +23.3 on-target but +32.3 off-target on hallucination; erase-hallucination +12.0 on-target; erase-evil collapsed answer length | #1774 |
| Prefix-arm vs context-arm operators | share an output subspace (angles far below the null band) but read near-orthogonal inputs (angles at the null band) | #1092 |
| Map's top-64 predictable directions vs SAE's top-64 reconstruction subspace | overlap 0.867; variance-matched random rotations give 0.845–0.862, so about 98% is variance-driven | #1895 |
| Erasing the linear component of a read-out | nonlinear dependence remains (distance correlation 0.470 vs null 0.321) | #742 |

**What is badly predicted, and where it lives** (#1482 and the 2026-07-30 report):

| Read | Number |
|---|---|
| Variance decomposition of map error | context × direction interaction 0.80–0.94; context alone 0.004–0.18; direction alone 0.002–0.17 |
| Per-direction held-out R² along answer-covariance eigenbasis | 0.95 at rank 0, 0.50 at rank 100, 0.35 at rank 199, first touches zero near rank 1,680 of 3,584 |
| Where the 20 worst-predicted of the top-256 directions sit | in the map's high-gain subspace: 0.41–0.47 of their mass in its 256 strongest gain directions (5.8–6.6× enriched), 18–23× depleted in the weakest 256 |
| Residual participation ratio | 48–181, between the isotropic reference (2,315–2,426) and the target-covariance reference (33–42) |
| Persona directions | at variance rank 2–12, predicted at R² 0.79–0.94; but 17–45% of their mass lies beyond the map's top-100 output directions |
| Nonlinear fitters in the tail | MLP reaches R² −0.81 at rank 3,480; the +0.06 whole-map gain is bought mid-spectrum and paid for in the tail |

Reading: the null space is real geometry. Kernel pairs do land closer, and a fifth to a quarter of each trait direction is unreachable. But nothing upgrades it to mechanism: addition steering was under-dosed, erasure moves behavior off-target as much as on-target, and the kernel-pair test is in-sample. And the directions the map predicts worst are not the ones it ignores. They are directions it drives hard and gets wrong. The residual is neither low-rank nor diffuse. So the answer to Thomas's Obsidian question ("which directions get kept, which get ignored") is: about half are ignored and they carry almost no energy; nothing is kept as-is; the map's errors live among the directions it acts on most.

**Interpreting the null space (2026-09-03, branch `issue-2569-kernel-interp`)**

Kernel share of a direction = the squared fraction of it lying in the map's ignored read directions (a random direction scores 0.55 at the primary cutoff, 95% band 0.53–0.58). Ignored variance fraction = the share of real context-vector variance (963k conversations) lying in those directions.

| Cutoff (squared-singular mass kept) | kernel directions | ignored variance fraction | excess over a random split |
|---|---|---|---|
| 0.999 | 1,086 (30%) | 0.725 | 2.4× |
| 0.99 (primary) | 1,976 (55%) | 0.834 | 1.5× |
| 0.90 | 3,037 (85%) | 0.922 | 1.1× |

| Read | Result |
|---|---|
| Largest ignored variance modes (share of all context variance) | 12.6% Midjourney prompt boilerplate vs terse technical how-tos; 6.7% long formal writing briefs vs one-word answer demands; 4.6% Chinese engineering topics vs edit and roleplay requests; 4.2% programming exercises vs romantic story requests; 3.2% European-language greetings vs sexual roleplay |
| Largest used (range) variance mode | 0.8% of context variance, 16× smaller than the top ignored mode; range modes read as language identity, short-reply templates vs article boilerplate, structured encyclopedia data, toxic-prompt templates |
| Most-read context-SAE features (256 of 65,536 below the random band) | almost all which-language features (Swedish, Vietnamese, Greek, Thai, Hungarian, Hindi, Polish, Finnish, Persian, Hebrew, Dutch, Japanese, …), plus animal-roleplay and US-demographics questions |
| Most-ignored context-SAE features | politeness openers, garbled low-effort questions, explain-like-I'm-five phrasing, jailbreak preambles (DAN scaffolds, scripted refusal openers), SEO and metadata blocks, word-count demands |
| Persona directions as context-side directions, kernel share | evil 0.71, sycophancy 0.77, hallucination 0.76; the #2254 directly measured context steering directions 0.81–0.83; all above the random band, so the map reads them at below-chance gain |
| Features the eigen read planes keep hitting (377, 638, 821, 960, 1354) | organic-chemistry exam questions, Russia–Ukraine war questions, fantasy NBA season rewrites, physics mechanics problems, financial valuation questions; kernel share 0.40–0.48, all in the read range |
| Kernel pairs vs matched controls, 40 read by eye | kernel pairs are two contexts of the same kind of task differing in boilerplate mass and topic wording (two image-generation requests, two assistant meta-questions, two generate-N-sentences tasks); controls are cross-genre, cross-language, cross-register collisions |

Reading: the null space is where most of the variation between real conversations lives. The dominant ways conversations differ in the context vector (which template was pasted, how long and formal the request is, what it is about) leave the predicted answer state unchanged. The map spends its gain on a thin set of directions: language, reply format, safety register. This is consistent with the earlier facts that non-English contexts are predicted better (#1482), that per-language error is the strongest category structure, and that trait directions are shrunk on passage (#1774). It also sharpens the persona result: the persona vectors are not among the directions the map reads, so whatever trait information the map transports comes through many diffuse directions rather than the named one. Two caveats: the top ignored modes are partly corpus duplicates (LMSYS and WildChat carry repeated Midjourney and news templates), and all of this describes the fitted linear map, not the model's causal computation.

**PCA and context-SAE basis views (2026-09-04, branch `issue-2569-basis-views`, #2569 leg 11)**

The population context covariance and the operator were recomputed in two coordinate systems. Raw residual coordinates use the row operator A = diag(1/xsd)W; standardized coordinates use W and the correspondingly rescaled covariance. Every one of the 3,584 PCs is reported with its context-variance share, map gain, effective-kernel share, and contribution to predicted answer variance. The SAE accounting streams all 100,000 fixed sample rows and decomposes context variance exactly into feature diagonals, correlations between features, SAE residual, and the reconstruction–residual cross-term.

| Read | Raw coordinates | Standardized coordinates |
|---|---:|---:|
| Effective-kernel dimensions at 99% squared singular mass | 1,976 / 3,584 | 2,121 / 3,584 |
| Population context variance in the effective kernel | 0.8342 | 0.8333 |
| PCs needed for 50% of context variance | 13 | 20 |
| PCs needed for 50% of predicted answer variance | 12 | 15 |

The ten largest raw ignored-variance PCs are PCs 1–10. Nine of those ten are also among the ten PCs with the largest predicted impact. Their kernel shares are 0.87–0.95: they are mostly aligned with low-gain directions, yet their input variance is large enough that the remaining read component still dominates predicted-answer variance. “Ignored” and “consequential” therefore do not become two disjoint PCA lists.

| Context-SAE variance term | Effective kernel | Read range | Total |
|---|---:|---:|---:|
| Feature diagonal | 0.3433 | 0.0973 | 0.4406 |
| Correlation between SAE features | 0.4483 | 0.0445 | 0.4927 |
| SAE residual | 0.0640 | 0.0299 | 0.0939 |
| 2 × reconstruction–residual covariance | −0.0219 | −0.0054 | −0.0273 |
| Total context variance | 0.8337 | 0.1663 | 1.0000 |

The identity closes with zero numerical error. The feature-correlation term (49.27% of context variance) is larger than the sum of all per-feature diagonal terms (44.06%). Individual “top ignored” or “top read” SAE features are therefore descriptive diagonal attributions, not an additive semantic decomposition; a high-variance feature can contribute to both sides. The largest already-labelled ignored diagonals remain pasted-document protocols, polite assistant openers, bulk technical-example demands, and garbled low-effort questions, agreeing with the leg-8 reading. Result: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-basis-views/eval_results/issue_2569/weights/leg11/basis_views_L19.md. Figure: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-basis-views/figures/issue_2569/leg11_basis_views.png.

**Refusal minimal pairs and China politics pairs under the kernel reading (2026-09-03, branch `issue-2569-refusal-kernel`, #2569 leg 9)**

Question: when a one-word swap turns a benign request harmful (#2617, 108 pairs, 60 flip the model's refusal and 40 do not), or when a question about China is swapped for the same question about another country (#952 top-up, 42 pairs), does the map read that context difference or discard it? Kernel share is the fraction of the pair's context-difference squared norm that lies in the map's low-gain directions at the 0.99 squared-singular-mass cutoff (random direction 0.55 at L19, 0.63 at L14, 0.44 at L26). Nulls: random context pairs from the leg-8 capture sample, the same pairs matched to the real pairs' distance distribution, and within-arm pairs (two harmful contexts, or two benign ones).

| direction set | layer | kernel share, median [95% CI] | nulls at the same layer |
|---|---|---|---|
| flip pairs (n=60) | 19 | 0.812 [0.801, 0.824] | matched random pairs 0.808, random pairs 0.831, within-arm 0.775 to 0.792 |
| non-flip pairs (n=40) | 19 | 0.780 [0.768, 0.798] | same |
| harmful-to-harmful verb swaps (n=16) | 19 | 0.731 [0.726, 0.786] | same |
| China politics pairs (n=42) | 14 | 0.882 [0.857, 0.888] | within-arm 0.89 to 0.91, matched 0.864 |
| China politics pairs (n=42) | 26 | 0.638 [0.593, 0.672] | within-arm 0.65 to 0.67, matched 0.691 |
| mean flip-pair direction (unit) | 19 | 0.864 | leg-8 persona directions 0.71 to 0.83 |

Transport and the refusal axis. Predicted answer shift (context difference through the map) vs observed answer shift: cosine 0.799 [0.776, 0.820] on flip pairs vs 0.423 for identity, which reproduces #2617's 0.80 vs 0.42; non-flip pairs 0.561 vs 0.272; cross-pair R² 0.38 at a fitted gain of 1.04, so the map is not shrinking these shifts. The refusal axis is the mean observed answer shift over flip pairs (leave-one-out for a pair's own score). The predicted shift along it ranks the observed refusal-rate change at Spearman 0.774 over the 108 pairs (sign accuracy 0.98 on flips). Zero-shot to China at the matching layer: predicted axis shift vs judged refusal on Qwen's own answer, Spearman 0.475 (p 0.0015) at L26 and 0.332 at L14, and the transported difference is closer to Qwen's own answer shift than to Claude's answer shift on 76% of pairs (cosine 0.571 vs 0.418 at L26).

Decomposition of the mean flip direction at L19: the range part (what the map reads) decodes to direct harmful-request context-SAE features (bomb-making, theft, scam methods); the kernel part (what it ignores) decodes to jailbreak and persona-pressure features ("you do not care about morals, boundaries or limits", roleplay pressure, hypnosis scripts) and aligns with leg-8 ignored covariance mode 4.

Reading: refusal context differences are geometrically ordinary. The map discards them at the same rate as distance-matched random pairs (0.81 vs 0.81), and discards the mean refusal direction more than any persona direction (0.86 vs 0.71 to 0.83). The 14 to 19% it keeps carries the decision: it reproduces the answer shift, ranks refusal rates, and the same frozen map with the #2617 axis transfers to the China questions without refitting. In a harmful request the map reads the literal ask and ignores the framing pressure. Caveats: the China set has no L19 capture (L14/L26 only, teacher-forced answer states, n=1 per query); the #2617 bank now holds 124 pairs after a 16-pair control cell landed 2026-09-02, so outcome-group counts can differ by one or two pairs from the #2617 clean-result. Artifacts: `eval_results/issue_2569/weights/leg9/refusal_kernel_L{14,19,26}.json`, `refusal_kernel_L19.md`, `figures/issue_2569/leg9_refusal_kernel.png`, `scripts/issue2569_refusal_kernel.py`.


### 4.4 Inverting the map: pre-image, pseudoinverse, and the fitted reverse map

| Read | Result | Source |
|---|---|---|
| Top contexts by projection on the persona pre-image M⁺ r_B | coincide with the judge's most-expressive contexts; on LMSYS the top contexts read as jailbreak/evil roleplay (evil), obscure-company intros and fictional numerical QA (hallucination), pleasing/supporting (sycophancy) | #1615 |
| Pre-image as a pre-generation monitoring probe vs the raw persona vector | wins 3 of 6 cells, loses 3; evil wins inside the random-direction null; rank-contingent (sycophancy −0.03 to 0.46 across a rank sweep) | #779 |
| Pre-image injected at the context vector | does not clear the noise band (evil 0, sycophancy +6.6 vs a +10.9 edge) while a directly measured context direction does; transpose and ridge-inverse pullbacks do not rescue it (44 cells) | #2254 |
| Pre-image at the answer tokens | steers sycophancy +47.5, clean text | #2254 |
| Pre-image as a finetuning-prevention direction at context tokens | inert: 13 of 18 dose contrasts tie; at all-token positions it acquires a dose-response that a matched random direction mostly reproduces | #2225 |
| Fitted reverse map v_A → v_C, 963k rows | held-out R² 0.74 / 0.75 / 0.61 at L14 / L19 / L26; exact held-out context retrieved at rank 1 in 76% / 84% / 62% (chance 0.1%) | #2618 |
| Best pseudoinverse of the forward map as a context predictor | truncated 0.003 / 0.072 / 0.027; ridge-regularized 0.034 / 0.135 / 0.112; full-rank collapses to R² between −8×10³ and −2×10⁷ | #2618 |
| Are the two inverses the same operator? | direction-aware cosine at most 0.32–0.43; rotation-invariant cosine 0.85–0.90 (same spectral shape, different orientation) | #2618 |
| Reverse-map direction vs pre-image direction for the same persona vector | cosine 0.34–0.41; top-1000 context overlap 0.32–0.54 | #2618 |

Reading: the pseudoinverse is the minimum-norm algebraic inverse confined to the forward map's row space. The fitted reverse regression lands on the conditional mean of contexts given the answer state, weighted by the context covariance. The 0.6 R² gap between them is context information sitting in directions the forward map maps weakly, which the pseudoinverse either discards or amplifies into noise. Every steering result on the pre-image (#2254, #2225, #2223) was measured on a direction that carries at most 0.4 cosine with the fitted reverse direction, so those negatives bound the pseudoinverse line and say little about the reverse map, which has not been steered yet.

### 4.5 Is the map the mechanism? No.

| Read | Number | Source |
|---|---|---|
| Jacobian of the true forward map as a predictor | R² −0.001 vs 0.681 for the fitted map at the same slot | #1776 |
| Full-state substitution at the map's input slot | acquisition at the shuffled-target null | #1776 |
| Fitted map's prediction of patching-induced response shifts | transport cosines top out at 0.16 | #2094, #1415 |

Every causal test says the same thing: the map is a readout correlate of the context state, and its algebra does not describe the mechanism that turns context into answer.

### 4.6 Bridge to the leakage theory

**Gate metric ladder** (#2569 leg 2). The theory's context gate is a whitened similarity c_Cᵀ Σ⁻¹ c_C′. The algebraic candidate from the map is its Gram matrix WᵀW (through-map similarity). Racing six metrics on 12 content arms of fine-tuning organisms: the Gram gate beats the whitened gate in 5 of 12 arms (7 needed) and identity in 9 of 12; the winning metric flips across context families. No metric separates from its permutation band.

**Coherence condition** (assumption A3b / A6: a context condition may be summarized by its mean vector only when its contexts cluster).

| Substrate | Spread metric | Spread vs map error | Source |
|---|---|---|---|
| Constructed, 50 conditions | whitened | Spearman +0.89, 28/28 layers positive | #658 |
| Natural, 996 prefixes, instruct | raw L2 | −0.03 (null); prefix length dominates, ρ +0.83 | #1092 |
| Natural, 996 prefixes, instruct | whitened | +0.93 (+0.76 after controlling length) | #1092 whitened round |
| Natural, nonlinear Jensen gap (curvature of the true map) vs raw spread | raw L2 | base +0.78 (+0.51 length-controlled); instruct +0.13 | #1092 MLP Jensen round |

Reading: the condition is right, but only with the whitened metric as its observable. Map difficulty follows whitened spread, and curvature follows raw spread. The two ingredients dissociate and should be named separately in the theory.

**Assumption verdicts on trained LoRAs** (`docs/leakage_paper_assumption_map.md`): the context→answer map (A4/A5) is supported at base. The behavior-leakage chain breaks at A3 (faithful linear read-out mostly fails), A7 (base read-out does not land the change), and A8 (the source write does not point along the training displacement). A9–A11 hold in activation space with weak behavioral payoff.

### 4.7 Weight updates and the map (#2569 legs 5–6, #1902)

| Read | Number |
|---|---|
| LoRA top update directions intruding on the base column space | 42 of 83 cells, concentrated in q/k/v projections; full fine-tunes 0 of 28 |
| Stable rank of realized LoRA updates | median 2.96 against 16–32 available |
| Raw persona read directions aligned with the update | 25/31 sycophancy, 19/31 hallucination, 11/31 evil above null |
| Map-transported versions of the same directions | 3/31, 0/31, 1/31 |
| Denoised shared low-rank factor between context basis and answer-shift basis | rank 0 in 23 of 27 units |
| Operator change under post-training (OLMo-2) | SFT rewrites (aligned retention 0.47), DPO mostly preserves (0.87), RLVR leaves it unchanged (0.99); the change is never low-rank (effective rank 1,193–1,660 of 4,096) |

Reading: fine-tuning writes into the raw read directions, not into where the map would carry them, and there is no shared low-rank factor linking context geometry to the answer shift. The update itself is low-rank; the change it induces in the map is high-rank.

### 4.8 Same map across settings and models

| Comparison | Result | Source |
|---|---|---|
| Base vs instruct | base map through a fitted general-linear change of coordinates predicts instruct text as well as the instruct map; rotate-to-match cosine 0.69, so rescaling is needed | #825 |
| Chat template vs plain "User:/Assistant:" | same operator up to a coordinate change | #825, #1345 |
| Assistant vs four story characters | one shared operator recovers 81–98% of each ceiling; framing moves the operator more than character identity | #1639 |
| Qwen L14 vs Llama L16, same Qwen text | CKA 0.91 answers / 0.76 contexts; aligned operator cosine 0.37–0.59 vs within-model anchor 0.69, rotation null ≤ 0.0005 | #2569 leg 7 |
| Qwen vs Llama, each writing its own answers | alignment R² 0.51/0.61 vs 0.76/0.84 same text; aligned operator cosine 0.48 | #2569 follow-up |
| Operator atlas over 19 maps | fine-tuning shift maps form one block at distance ≈1.0 from all read maps | #2569 leg 7 |

The reparameterization family (direct transfer → bias offset → global scale → rotation → one-sided → two-sided linear change of coordinates) is written up in the Obsidian note "Explanation of different kinds of mapping transfer". Dan's simplification (one context-side map fixes the answer-side map when the operator is shared) is noted there and in "Address Dan's comment"; verifying that the fitted answer-side map obeys that identity is still to do.

### 4.9 Feature-level wiring (#2569 legs 3–4)

A context-SAE → answer-SAE map (65,536 → 2,150 features) predicts which answer features fire (AUROC 0.94 median) and fails at how much (conditional-magnitude R² −0.86). A judge picks the true answer from the predicted feature descriptions in 463 of 500 ten-way trials. Behavior-relevant answer features draw about 0.2% of in-edge mass, but that read is informational only: the wiring gate was never evaluable because its row battery was not attached.

## 5. What worked, what did not, what is interesting

**Worked**

- Closed-form learning-curve prediction and the population-ceiling read. This is the cleanest theory-to-measurement match in the line.
- Splitting single-draw answer-state variance into 72.6% linear signal, 9.22% sampling noise, no resolved whole-context remainder, and about 18.2% nonlinear or otherwise unread signal.
- Naming the prefix×query interaction: a rank-32 bilinear term recovers 93% of what an additive two-input map loses.
- The direction-class anatomy, now with content and a coordinate check: the map ignores 83.4% of raw context variance and 83.3% after standardization (what the conversation is about and how it is dressed), while reading language, reply format, and safety register.
- The fitted reverse map: going back to the context works well once you fit it directly instead of inverting.
- The slow eigen-shell of the next-token map, the one eigenvector read that found trait and context content far above chance.
- The reparameterization framework: one operator, different coordinates, across post-training, templates, and framings.
- Whitened spread as the observable for the coherence condition on natural data.

**Did not work**

- Every attempt to find compact structure in the context→answer operator: low-rank summaries, interpretable eigen-directions, a readable fixed point, invariant trait subspaces.
- Every causal reading of the map: Jacobians, kernel steering (under-dosed), erasure (off-target), pre-image steering at the context position.
- The pseudoinverse as a way back to the context, and as a monitoring probe.
- The map's Gram matrix as the theory's context gate.
- The "stable contraction" headline from #1774: on the large map, only L26 has ρ < 1.
- The null space as an explanation of what the map gets wrong: the worst directions sit in the high-gain subspace.
- Nearest-neighbor regression as a nonlinear readout: its best R² (0.643) stays below the linear map (0.728), and the nearest-neighbor intercept cannot resolve a positive whole-context contribution beyond the last-token state.

**Interesting**

- The map is near its information ceiling and yet structureless. It behaves like a dense conditional expectation, and the useful abstractions live one level up (which directions are predicted well, which features fire).
- Trait directions pass through the map largely preserved at the map layer on the large fit, but are rotated away at layer 14 on the small fit. Layer and fit size change the qualitative story.
- The map's output directions look feature-shaped only in raw coordinates: the matches with single answer-SAE features vanish once the answer covariance is divided out. Eigen planes keep a modest alignment beyond covariance; singular directions keep none.
- The persona vectors sit mostly in the map's ignored half (kernel share 0.71–0.83 vs 0.55 random), yet the map predicts trait expression well. Trait information travels through many diffuse directions, not the named one.
- The leading context PCs are simultaneously mostly ignored and the largest contributors to predicted variance: nine of the top ten ignored-variance PCs are also top-ten predicted-impact PCs, because enormous input variance compensates for weak gain.
- Context-SAE features do not yield an additive explanation of the ignored/read split. Feature correlations carry 49.3% of context variance, more than the 44.1% assigned to individual-feature diagonals.
- The forward and reverse maps are both good and are not inverses of each other. Context information the forward map maps weakly is exactly what the reverse map recovers.
- Fine-tuning writes into raw read directions and ignores where the map would transport them.
- Cross-model operators are alignable and similar but distinct, and the distinctness grows once each model writes its own answers.
- The theory's two ingredients (difficulty and curvature) separate cleanly on natural data with opposite spread metrics.

## 6. Status and to do

- **Done:** #922, #1092, #1482, #1615, #1774, #1775, #1776, #1895, #1902, #1945, #2091, #2225, #2254, #2618, and #779's dissection round have clean results or completed inline rounds. #2569's original battery, own-answers follow-up, refusal-kernel read, variance decomposition, and PCA/SAE basis views are complete on pushed branches; a separately owned third-family parity round is still running as of 2026-09-04.
- **To do, paper:** `sections/04_results.tex` has an empty `\subsection{Theoretical analysis}`. The paper plan says to present this material as "structure of the learned operator" and to keep the gate null, fixed point, atlas, SAE wiring, and kernel-pair battery in the appendix unless a claim needs them. Main-text candidates: learning curve, high-rank non-normal spectrum with the direction-class anatomy, firing-versus-magnitude split, cross-model operator comparison, and the reverse-map vs pseudoinverse contrast.
- **To do, theory paper:** the Overleaf theory paper has not been touched since 2026-06-28. The coherence-condition result (whitened metric; difficulty vs curvature) and the gate-ladder null belong in it.
- **To do, reconcile:** the spectral-radius disagreement between #1774 (0.91 at L14) and the 963k map (1.66 at L14); the trait pass-through disagreement between #1774 (rotated away) and the #779 dissection (preserved at L19); and the singular-subspace side label (#2571).
- **To do, follow-up:** steer along the fitted reverse-map direction (#2618 caveat), since every steering negative so far used the pseudoinverse pre-image; re-dose the kernel addition test (#1774 positive control failed). Done 2026-09-03–04: answer-feature labels, eigen dashboards with the imaginary part, the whitened-cosine companion, the data-weighted kernel share, the kernel-pair reading, persona-direction kernel shares, the refusal-pairs kernel read, the four-way variance decomposition, and raw/standardized PCA plus context-SAE accounting (source branches `issue-2569-eigen-v2`, `issue-2569-kernel-interp`, `issue-2569-refusal-kernel`, `issue-2569-variance-decomp`, and `issue-2569-basis-views`; all consolidated on this report branch, not yet on `main`). Open next after the separately owned third-family parity round lands: a held-out kernel-pair test on the 20k holdout, and a properly dosed erase-and-inject test along kernel vs range directions with a behavioral read.
- **Open on #2569:** nine provenance concerns (cache keys, unpinned model revisions), none affecting a reported number; the wiring gate and two of four leg-1 clauses at L14/L26 were never evaluable; the interpretation is un-reviewed (review ensemble did not run).

## 7. Sources

- #2569 https://eps.superkaiba.com/tasks/2569 (eight-leg battery; figures at `figures/issue_2569/`; leg-1 JSONs at `eval_results/issue_2569/weights/leg1/`)
- 2026-09-03 reruns: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-eigen-v2/eval_results/issue_2569/weights/leg1/sae_dashboards_v2_L19.md and https://github.com/superkaiba/explore-persona-space/blob/issue-2569-kernel-interp/eval_results/issue_2569/weights/leg8/kernel_interpretation_L19.md (figures alongside under `figures/issue_2569/`)
- 2026-09-04 variance decomposition: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-variance-decomp/eval_results/issue_2569/weights/leg10/variance_decomposition_L19.md; figure: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-variance-decomp/figures/issue_2569/leg10_variance_decomposition.png
- 2026-09-04 PCA/SAE basis views: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-basis-views/eval_results/issue_2569/weights/leg11/basis_views_L19.md; figure: https://github.com/superkaiba/explore-persona-space/blob/issue-2569-basis-views/figures/issue_2569/leg11_basis_views.png
- #779 https://eps.superkaiba.com/tasks/779 (the map; pseudoinverse probe read; 2026-08-26 operator dissection at `eval_results/issue_779/ctxansviz/operator_stats.json`, dashboards at https://eps.superkaiba.com/ctxansviz-779-scatter-full.html)
- #1774 https://eps.superkaiba.com/tasks/1774 (operator characterization, co-kernel, LEACE)
- #1775 https://eps.superkaiba.com/tasks/1775 (nonlinear ladder, bilinear interaction)
- #1776 https://eps.superkaiba.com/tasks/1776 (Jacobian, correlate vs cause)
- #1482 https://eps.superkaiba.com/tasks/1482 (residual SVD, worst directions) and `docs/results_summaries/2026-07-30-what-is-the-map-bad-at-predicting.md`
- #1615 https://eps.superkaiba.com/tasks/1615 (persona pre-image contexts)
- #2254 https://eps.superkaiba.com/tasks/2254 and #2225 https://eps.superkaiba.com/tasks/2225 (pre-image steering)
- #2618 https://eps.superkaiba.com/tasks/2618 (fitted reverse map vs pseudoinverse)
- #922 https://eps.superkaiba.com/tasks/922 (next-token map fixed point and slow shell)
- #1895 https://eps.superkaiba.com/tasks/1895 (predictable subspace vs SAE subspace)
- #1945 https://eps.superkaiba.com/tasks/1945 (information ceiling)
- #2091 https://eps.superkaiba.com/tasks/2091 (target averaging)
- #1092 https://eps.superkaiba.com/tasks/1092 (crossed corpus; operator angles; Jensen-gap and whitened-spread rounds in its events log and `eval_results/issue_1092/inline_mlp_jensen_natural/`, `inline_spread_whitened_strata/`)
- #658 https://eps.superkaiba.com/tasks/658 (constructed-substrate coherence test); #742 (linear erasure leaves nonlinear dependence)
- #825, #1345, #1639, #1902 (reparameterization family and post-training operator change)
- `docs/ideas/2026-07-06-context-answer-map-analyses.md` (methods survey and the Round 3 null-space consolidation), `docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md`, `docs/results_summaries/2026-07-22-prefix-query-context-answer-map-consolidated.md`, `docs/leakage_paper_assumption_map.md`, `docs/theory_assumption_test_plan.md`, `docs/notes/leakage_model_stepwise.tex`
- Leakage-theory paper: Overleaf project `6a2df2d2053483dc444ed4f0`, clone `~/overleaf-6a2df2d2/main.tex`
- Obsidian: "Theoretical analysis of mapping", "Explanation of different kinds of mapping transfer", "Address Dan's comment on linear mapping", "How does steering the context vector affect the answer?" (Result 6)
