# Theoretical analysis of the context→answer map: compiled report

*2026-09-02. Compiled from the task clean-results, the docs listed in §7, the June leakage-theory paper, and Thomas's Obsidian notes. Numbers are quoted from the cited clean-results; nothing here is a new measurement.*

## 1. Summary

- **The map is one high-rank linear operator, and we can predict its own learning curve from first principles.** The closed-form ridge learning curve matches the measured curve to within 0.006 R² over two decades of training size, and the 963k-row map sits at 99% of the population linear ceiling (#2569). What the linear map misses is small (about 0.05 R²) and mostly one named thing: a low-rank prefix×query interaction (#1775).
- **The operator has no compact structure.** Hundreds to thousands of validated channels, strongly non-normal, top-128 directions carry only about two thirds of the energy, and the fixed point is a dense, unreadable state (#1774, #2569). Thomas's own reading in Obsidian: "Understand structure of mapping: I think done, not really any structure."
- **The map describes a correlation. It is not the causal mechanism.** Jacobians of the true forward map recover none of its predictive power, full-state substitution at the map's input slot moves nothing, and the map's pre-image cannot steer where a directly measured direction can (#1776, #2254).
- **The bridge to the leakage theory is weak in behavior space and real in activation space.** No gate metric, including the map's own Gram matrix, separates from its panel when predicting where fine-tuning delivers behavior change (#2569 leg 2). The coherence condition holds on constructed contexts and on natural data only under the whitened metric (#658, #1092).
- **The map is the same object across settings up to a change of coordinates, and only partly shared across model families.** Post-training, chat template, and story framing all move the map by a general-linear reparameterization (#825, #1345, #1639). Qwen and Llama operators are far from a rotation null but clearly different operators (#2569 leg 7).

## 2. The object

| Term | Meaning |
|---|---|
| Context vector v_C | Residual-stream state at the last prompt token (end of the user turn), layer 19 of Qwen2.5-7B-Instruct unless stated. |
| Answer vector v_A | Mean residual-stream state over the answer tokens, same layer. |
| The map W (also A, M) | Ridge regression v_A ≈ W v_C + b, fit on 963,444 real LMSYS + WildChat conversations (#779). Held-out R² 0.75 single-draw, 0.80 with five-draw targets. |
| Endomorphism | W maps a space to itself (same layer, same width 3,584), so eigenvalues, a fixed point x* = (I − W)⁻¹ b, and iterating the map are all defined. |
| Spectral radius ρ | Largest eigenvalue magnitude. ρ < 1 means iterating the map converges to the fixed point. |
| Non-normal | Eigenvectors far from orthogonal (condition number κ(V) in the thousands). Singular values then overstate per-step growth, and eigen reads need a gate. |
| Leakage theory (June paper) | Predictor L̂ = strength × (read-out ᵀ training displacement) × context gate. The context→answer map is its assumption A3 (A4/A5 in the assumption map), and the gate is a whitened context similarity. |

## 3. What we did

| When | Item | Where |
|---|---|---|
| June 2026 | Leakage-theory paper: assumptions A1–A11, the boxed predictor, cosine predictor as the isotropic special case, an assumption ladder from a rank-one key-value memory. | Overleaf `6a2df2d2`, `docs/notes/leakage_model_stepwise.tex` |
| July 6 | Methods survey: Koopman/DMD, transfer operators, reduced-rank regression, non-normality gates, validity gates for operator reads. | `docs/ideas/2026-07-06-context-answer-map-analyses.md` |
| July 22 | Consolidated prefix vs query vs context results, including the coherence-condition test on natural data. | `docs/results_summaries/2026-07-22-...consolidated.md`, #1092, #923, #658 |
| July 28 | Four-arm theoretical analysis plan (Jensen gap, channel counts, kernel, eigen reads, fixed point, nonlinearity ladder). | `docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md` |
| July 29 | #1774 linear operator characterization, #1775 nonlinear ladder, #1776 Jacobian test. | task bodies |
| Aug 1–19 | #1945 information ceiling, #2091 target averaging, #2254 map inversion as steering, #1902 post-training operator rank. | task bodies |
| Aug 24–29 | Theory walk → #2569 eight-leg battery (19 figures), plus a cross-model own-answers follow-up. | #2569 |

## 4. Results by question

### 4.1 Is one linear map the right description? Yes, to within a small named remainder.

| Read | Number | Source |
|---|---|---|
| Closed-form ridge learning curve vs measured, five sizes 4.5k→500k | mean abs gap 0.0058 R², same sign at every point | #2569 leg 2 |
| Held-out R² at n=500k vs population linear ceiling | 0.719 vs 0.726 (99.1%) | #2569 leg 2 |
| Residual error structure across context pairs | peak interaction R² 0.0013 against a 0.10 materiality floor | #1945 |
| Gain from averaging five answer draws | 0.06–0.10 R², of which 0.05–0.095 is target averaging, not the map | #2091 |
| Nonlinear gain over ridge at 963k rows | ≈ +0.05 R²; all nonlinear families converge near 0.81 | #779, #1901 |
| Share of that gap closed by a rank-32 bilinear prefix×query term | 93% on novel prefixes | #1775 |

Reading: the linear map is the population-optimal linear predictor and its remaining error is close to per-pair noise. The nonlinearity that exists is mostly the query modulating the prefix through a few channels.

### 4.2 What does the operator look like? High-rank, non-normal, no compact summary.

| Read | L14 | L19 (map layer) | L26 | Source |
|---|---|---|---|---|
| Rank for 90% / 99% of squared singular mass | 411 / 1,325 | 547 / 1,608 | 794 / 2,000 | #2569 leg 1 |
| Frobenius energy in the top 128 directions | 0.70 | 0.65 | 0.58 | #2569 leg 1 |
| Spectral radius ρ | 1.66 | 1.21 | 0.92 | #2569 leg 1 |
| Eigenvector condition number κ(V) | 7,650 | 4,261 | 7,452 | #2569 leg 1 |
| Split-half stable singular directions at L19 | | 3,425 of 3,584 | | #2569 leg 1 |

Related reads from the smaller #1092 corpus (layer 14, 21k rows): 763–2,932 validated channels depending on convention; trait directions are shrunk to gain 0.60–0.64 and rotated so that 99.6% of their output mass leaves the trait span; top singular directions amplify 4–11× at near-zero alignment with their input (#1774).

Fixed point: it exists algebraically at every layer, but iterating the map converges only at L26. Its norm is the answer-pool median (#1774) and it lights 10,302 of 65,536 context-SAE features (#2569), so it is not an interpretable "default assistant" state.

One discrepancy to reconcile: #1774 reported ρ = 0.91 at layer 14 (a contraction) on the 21k-row map, while #2569 reports ρ = 1.66 at layer 14 on the 963k-row map. Different corpus, n, and λ; the "stable contraction" headline from #1774 did not carry over to the large map at L14 or L19.

### 4.3 What the map ignores, and whether "ignores" is causal

| Read | Number | Source |
|---|---|---|
| Kernel-direction context pairs: answer displacement vs matched controls | 0.607× (CI 0.602–0.615); 1.23× the residual floor vs 2.01× for controls | #2569 leg 8 |
| Jacobian of the true forward map as a predictor | R² −0.001 vs 0.681 for the fitted map at the same slot | #1776 |
| Full-state substitution at the map's input slot | acquisition at the shuffled-target null | #1776 |
| Map pre-image of a persona vector injected at the context vector | does not clear the noise band; a directly measured context direction does | #2254 |
| Kernel steering by addition | under-dosed, no verdict | #1774 |

Reading: kernel pairs really do produce closer answers, but 99.9% of them lie inside the map's own training rows, so this is consistency, not held-out validation. Every causal test says the same thing: the map is a readout correlate of the context state, and its algebra does not describe the mechanism that turns context into answer.

### 4.4 Bridge to the leakage theory

**Gate metric ladder** (#2569 leg 2). The theory's context gate is a whitened similarity c_Cᵀ Σ⁻¹ c_C′. The algebraic candidate from the map is its Gram matrix WᵀW (through-map similarity). Racing six metrics on 12 content arms of fine-tuning organisms: the Gram gate beats the whitened gate in 5 of 12 arms (7 needed) and identity in 9 of 12; the winning metric flips across context families. No metric separates from its permutation band. The context gate is a family-specific correlate rather than a universal quantity.

**Coherence condition** (assumption A3b / A6: a context condition may be summarized by its mean vector only when its contexts cluster).

| Substrate | Spread metric | Spread vs map error | Source |
|---|---|---|---|
| Constructed, 50 conditions | whitened | Spearman +0.89, 28/28 layers positive | #658 |
| Natural, 996 prefixes, instruct | raw L2 | −0.03 (null); prefix length dominates, ρ +0.83 | #1092 |
| Natural, 996 prefixes, instruct | whitened | +0.93 (+0.76 after controlling length) | #1092 whitened round |
| Natural, nonlinear Jensen gap (curvature of the true map) vs raw spread | raw L2 | base +0.78 (+0.51 length-controlled); instruct +0.13 | #1092 MLP Jensen round |

Reading: the condition is right, but only with the whitened metric as its observable. Map difficulty follows whitened spread, and curvature (the averaging failure a linear map cannot express) follows raw spread. The two ingredients dissociate and should be named separately in the theory.

**Assumption verdicts on trained LoRAs** (`docs/leakage_paper_assumption_map.md`): the context→answer map (A4/A5) is supported at base. The behavior-leakage chain breaks at A3 (faithful linear read-out mostly fails), A7 (base read-out does not land the change), and A8 (the source write does not point along the training displacement). A9–A11 hold in activation space with weak behavioral payoff.

### 4.5 Weight updates and the map (#2569 legs 5–6, #1902)

| Read | Number |
|---|---|
| LoRA top update directions intruding on the base column space | 42 of 83 cells, concentrated in q/k/v projections; full fine-tunes 0 of 28 |
| Stable rank of realized LoRA updates | median 2.96 against 16–32 available |
| Raw persona read directions aligned with the update | 25/31 sycophancy, 19/31 hallucination, 11/31 evil above null |
| Map-transported versions of the same directions | 3/31, 0/31, 1/31 |
| Denoised shared low-rank factor between context basis and answer-shift basis | rank 0 in 23 of 27 units |
| Operator change under post-training | effective rank 1,193–1,660 of 4,096 (#1902) |

Reading: fine-tuning writes into the raw read directions, not into where the map would carry them, and there is no shared low-rank factor linking context geometry to the answer shift. The update itself is low-rank; the change it induces in the map is high-rank.

### 4.6 Same map across settings and models

| Comparison | Result | Source |
|---|---|---|
| Base vs instruct | base map through a fitted general-linear change of coordinates predicts instruct text as well as the instruct map; rotate-to-match cosine 0.69, so rescaling is needed | #825 |
| Chat template vs plain "User:/Assistant:" | same operator up to a coordinate change | #825, #1345 |
| Assistant vs four story characters | one shared operator recovers 81–98% of each ceiling; framing moves the operator more than character identity | #1639 |
| Qwen L14 vs Llama L16, same Qwen text | CKA 0.91 answers / 0.76 contexts; aligned operator cosine 0.37–0.59 vs within-model anchor 0.69, rotation null ≤ 0.0005 | #2569 leg 7 |
| Qwen vs Llama, each writing its own answers | alignment R² 0.51/0.61 vs 0.76/0.84 same text; aligned operator cosine 0.48 | #2569 follow-up |
| Operator atlas over 19 maps | fine-tuning shift maps form one block at distance ≈1.0 from all read maps | #2569 leg 7 |

The reparameterization family (direct transfer → bias offset → global scale → rotation → one-sided → two-sided linear change of coordinates) is written up in the Obsidian note "Explanation of different kinds of mapping transfer". Dan's simplification (one context-side map fixes the answer-side map when the operator is shared) is noted there and in "Address Dan's comment"; verifying that the fitted answer-side map obeys that identity is still to do.

### 4.7 Feature-level wiring (#2569 legs 3–4)

A context-SAE → answer-SAE map (65,536 → 2,150 features) predicts which answer features fire (AUROC 0.94 median) and fails at how much (conditional-magnitude R² −0.86). A judge picks the true answer from the predicted feature descriptions in 463 of 500 ten-way trials. Behavior-relevant answer features draw about 0.2% of in-edge mass, but that read is informational only: the wiring gate was never evaluable because its row battery was not attached.

## 5. What worked, what did not, what is interesting

**Worked**

- Closed-form learning-curve prediction and the population-ceiling read. This is the cleanest theory-to-measurement match in the line.
- Naming the nonlinearity: a rank-32 bilinear prefix×query term.
- The reparameterization framework: one operator, different coordinates, across post-training, templates, and framings.
- Whitened spread as the observable for the coherence condition on natural data.

**Did not work**

- Every attempt to find compact structure in the operator: low-rank summaries, interpretable eigen-directions, a readable fixed point, invariant trait subspaces.
- Every causal reading of the map: Jacobians, kernel steering, pre-image steering.
- The map's Gram matrix as the theory's context gate.
- The "stable contraction" headline from #1774: on the large map, only L26 has ρ < 1.

**Interesting**

- The map is near its information ceiling and yet structureless. It behaves like a dense conditional expectation, and the useful abstractions live one level up (which directions are predicted well, which features fire).
- Fine-tuning writes into raw read directions and ignores where the map would transport them. The map reads the assistant's state; it does not describe how that state is written.
- Cross-model operators are alignable and similar but distinct, and the distinctness grows once each model writes its own answers.
- The theory's two ingredients (difficulty and curvature) separate cleanly on natural data with opposite spread metrics.

## 6. Status and to do

- **Done:** #1774, #1775, #1776, #1945, #2091, #2254, #2569 and its own-answers follow-up are all parked at awaiting_promotion with clean results.
- **To do, paper:** `sections/04_results.tex` has an empty `\subsection{Theoretical analysis}`. The paper plan says to present this material as "structure of the learned operator" and to keep the gate null, fixed point, atlas, SAE wiring, and kernel-pair battery in the appendix unless a claim needs them. Main-text candidates: learning curve, high-rank non-normal spectrum, firing-versus-magnitude split, cross-model operator comparison.
- **To do, theory paper:** the Overleaf theory paper has not been touched since 2026-06-28. The coherence-condition result (whitened metric; difficulty vs curvature) and the gate-ladder null belong in it.
- **To do, reconcile:** the spectral-radius disagreement between #1774 (0.91 at L14) and #2569 (1.66 at L14).
- **Open on #2569:** nine provenance concerns (cache keys, unpinned model revisions), none affecting a reported number; the wiring gate and two of four leg-1 clauses at L14/L26 were never evaluable; the interpretation is un-reviewed (review ensemble did not run).

## 7. Sources

- #2569 https://eps.superkaiba.com/tasks/2569 (eight-leg battery; figures at `figures/issue_2569/`)
- #1774 https://eps.superkaiba.com/tasks/1774 (operator characterization)
- #1775 https://eps.superkaiba.com/tasks/1775 (nonlinear ladder, bilinear interaction)
- #1776 https://eps.superkaiba.com/tasks/1776 (Jacobian, correlate vs cause)
- #1945 https://eps.superkaiba.com/tasks/1945 (information ceiling)
- #2091 https://eps.superkaiba.com/tasks/2091 (target averaging)
- #2254 https://eps.superkaiba.com/tasks/2254 (map inversion as steering)
- #1092 https://eps.superkaiba.com/tasks/1092 (crossed corpus; Jensen-gap and whitened-spread rounds in its events log and `eval_results/issue_1092/inline_mlp_jensen_natural/`, `inline_spread_whitened_strata/`)
- #658 https://eps.superkaiba.com/tasks/658 (constructed-substrate coherence test)
- #825, #1345, #1639, #1902 (reparameterization family and post-training operator rank)
- `docs/ideas/2026-07-06-context-answer-map-analyses.md`, `docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md`, `docs/results_summaries/2026-07-22-prefix-query-context-answer-map-consolidated.md`, `docs/leakage_paper_assumption_map.md`, `docs/theory_assumption_test_plan.md`, `docs/notes/leakage_model_stepwise.tex`
- Leakage-theory paper: Overleaf project `6a2df2d2053483dc444ed4f0`, clone `~/overleaf-6a2df2d2/main.tex`
- Obsidian: "Theoretical analysis of mapping", "Explanation of different kinds of mapping transfer", "Address Dan's comment on linear mapping"
