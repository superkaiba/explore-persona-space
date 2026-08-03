# What is the context→answer map bad at predicting? — Results fill-in (2026-08-02)

Fill-in of the Result 1 + Result 2 skeleton (user structure kept verbatim; blanks
filled from banked artifacts; every number traces to the artifact named beside it).

## Results

### Result 1: Does the mapping mostly fail at predicting **specific directions** or **specific contexts?**

**TLDR**: The mappings (prefix, context, bare query) do not fail exclusively at predicting specific directions or specific contexts, but some combination/interaction of the 2

The average $R^2$ of ~0.8 obscures one thing:
- is the gap of 0.2 from perfect $R^2$ of one because the mapping "predicting each context pretty well" or "predicting some contexts perfectly and others not at all" or some combination of both

**Methodology**
To test this, I arranged the map's held-out errors as a table: one row per context, one column per answer direction (PCs of answer vectors sorted from highest to lowers), with each entry the squared error the map made there.

If you just take this raw table, the largest PCs dominate (errors are largest because they are largest), so I divided each column by the direction's variance to control for this factor -- every entry is the **fraction of the available variance the map missed there**

I then calculated how much of the variance of the table was explained by:
- contexts (rows)
- directions (columns)
- other (interaction between contexts and directions)
(using a two-way ANOVA)

I did this for the mappings from:
- context -> answer
- prefix -> answer (prefix end state)
- bare query -> answer

I then plotted the results (for both linear and nonlinear mappings):
![Where the map's error lives — context, direction and interaction shares of held-out error for the three input arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/172d491785e5f47491be5900f9d71f0284a6dc84/figures/issue_1482/twoway_residual/result1_variance_components.png)

**Takeaways:**
- For all mappings, most of the variance is from the **interaction between contexts and directions** and not specific to **either certain contexts or certain directions**
- This does not mean that we cannot attempt to characterize the worst predicted directions or contexts, just that the effect of either of these separately is small relative to the total error

### Result 2: Characterization of worst predicted directions

I then tried to characterize as much as I could what the worst predicted directions in the answer's residual stream were.

The first definition of "direction" I used is the same as above, i.e. the PCs of the mean answer vectors.

One thing we would expect is that high-variance directions would be predicted better than low-variance directions. To sanity check this, we plot the $R^2$ per PC vs. the variance explained of the PC for both the linear predictor (ridge regression) and nonlinear predictor

![Per-direction held-out R² vs variance share, linear (ridge) vs nonlinear (MLP w8192), context arm, L19, multi-turn holdout](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/result2_assembly/spectrum_ridge_vs_mlp.png)

**Takeaways:**
- $R^2$ decays almost perfectly monotonically with variance rank: 0.946 (rank 0) → 0.503 (r100) → 0.354 (r199) → ≈0 (r2000+), crossing zero at **observed rank ~1,680** of 3,584 (the fitted two-parameter floor curve crosses ~rank 280; the observed curve hugs zero from ~r1000 on). Only the top **~690 of 3,584 PCs (~19%) carry $R^2$ > 0.1** (multi-turn holdout, ridge: 689 above 0.1, 1,568 positive at all; the plotted multi-turn spectrum runs slightly below the single-turn numbers quoted above — ridge first non-positive at r1,368, MLP at r902; `eval_results/issue_1482/result2_assembly/assembly.json`)
- Linear and nonlinear agree almost exactly on WHICH directions are predictable (rank Spearman **0.997**; best-20 sets share 19/20 members) — the nonlinear map is slightly better in the head and gives up earlier in the tail
- There is no small set of high-variance "bad" directions that might indicate that the mapping is worse at predicting a specific direction

I then looked at the location of the persona vector directions on the $R^2$ vs variance explained plot — using all seven traits (the paper's main three (sycophancy, evil, hallucination) plus optimistic / impolite / apathetic / humorous , which are in the paper's appendix)

![All seven persona-vector directions on the R²-vs-variance spectrum, context arm, multi-turn holdout, L19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/rb7_reads/rb7_spectrum.png)

**Takeaways:**
- The persona vector directions are among the high variance directions: all seven sit at **equivalent variance rank 4–13** and are predicted at **$R^2$ 0.803–0.907**, above the 200-random-direction band's p95 (0.756) (`eval_results/issue_1482/rb7_reads/rb7_reads.json`)
    - and therefore they are predictable pretty accurately by our mapping -> indicating that our mapping will be able to predict meaningful behavior in the answer

I then checked if the best/worst predicted directions were interpretable by:
- finding their most similar SAE features (in a pretrained SAE, descriptions from NeuronPedia)
- applying logit lens
- applying tuned lens
- applying JLens

Table (ridge; the nonlinear map selects an almost identical best set — 19/20 shared — and 8/20 of the worst; lens cells are the top-4 promoted tokens):

| rank | R² | \|cos\| nearest SAE feat | autointerp of that feat | logit lens | tuned lens | J-lens |
|---|---|---|---|---|---|---|
| **best** PC3 | 0.902 | 0.48 | Foreign languages | rekl, 哪家好, 层出不, сдела | to, just, now, that | :".$, :'.$, ').", '); |
| **best** PC1 | 0.897 | 0.31 | technical language/code | raping, documento, /MPL, alties | ?>, ()->, (){, /MPL | ?>, 👆, =, ☝ |
| **best** PC4 | 0.891 | 0.40 | Business/policy contexts | 日报社, masturbating, 说到这里, 不得不说 | 杨欢, flen, masturbating, helicopt | Förder, Geschäfts, ynos, Führung |
| **best** PC0 | 0.873 | 0.63 | Code/Comments | 我真的, subparagraph, 这个职业, 应该怎么 | ?>>, ');, :<?, )==' | Youtube, :\", :<?, :".$ |
| **best** PC2 | 0.869 | 0.36 | narrative snippets | alties, unmist, UGC, rippling | rippling, data, file, if | UGC, MethodInfo, 👇, DeepCopy |
| **best** PC5 | 0.866 | 0.42 | Foreign words | libertin, userEmail, extrad, 素晴 | 'B, libertin, ♮, extrad | ++, Congratulations, Training, ❤ |
| **best** PC6 | 0.852 | 0.29 | animal descriptions | deltaTime, 季后, 诋, 詹姆 | issue, 季后, issues, 诋 | necessary, requests, required, request |
| **best** PC15 | 0.848 | 0.59 | Russian text | démarch, nomine, Sexo, filmy | démarch, у, "w, nomine | }->{, ='".$_, "math, }-> |
| **best** PC7 | 0.824 | 0.60 | Code documentation | 有期徒, 建档立, 孙悟, 拜师学 | 有期徒, 建档立, Home, 会让你 | Geschä, 的带领, Website, 疖 |
| **best** PC8 | 0.820 | 0.36 | How-to guides | bigot, (iOS, assh, libertine | 财运, 恫, ---, ☝ | 箴, SMART, Comfort, Outdoor |
| **worst** PC235 | 0.272 | 0.13 | ration/irrational | -initialized, (paren, Parenthood, loid | #af, ,},, }}],, ITERAL | specific, engineering, unknown, music |
| **worst** PC238 | 0.280 | 0.11 | Photoperiod and light | TRY, Couples, Russo, �� | {{--<, Interracial, "., IonicPage | TRY, endars, ✕, (Collider |
| **worst** PC246 | 0.283 | 0.11 | image inversion | ..., rift, ceasefire, SFML | hairst, uintptr, Fitzgerald, 纨 | ооруж, ]\|[, ≧, ennen |
| **worst** PC251 | 0.284 | 0.12 | bits | elmet, odeled, BuilderInterface, /***/ | elmet, Wunused, Verdana, odeled | elmet, "crypto, ..., reator |
| **worst** PC252 | 0.284 | 0.12 | Achievements | bergen, ApplicationException, ALLERY, möglich | ële, :CGRect, ApplicationException, aniem | \`\`\`, ____, 构思, \`\`\` |
| **worst** PC242 | 0.285 | 0.14 | amateur radio communication | hlen, kommun, grily, näm | @implementation, hlen, .ma, .bi | ␣, .chomp, ␣, analyst |
| **worst** PC249 | 0.287 | 0.12 | restaurant reviews | "., 相关负责, dụ, Gorgeous | aternity, Wroc, .SDK, zept | much, relevant, gars, more |
| **worst** PC244 | 0.291 | 0.12 | scientific/technical texts | .Preference, asca, :<?, grosse | izr, .Preference, antibiot, 冶 | ...", ...", ...", ...") |
| **worst** PC240 | 0.292 | 0.11 | Awards shows | sublic, RELATED, FixedUpdate, 若要 | Q, acic, acet, uzzer | *****, ?, :, ------- |
| **worst** PC222 | 0.292 | 0.12 | Internet text snippets | orgh, ookies, идент, 来看看吧 | :';, -UA, /MPL, -Mobile | 或多, :[, either, both |

(Full lens artifact incl. the deviant band + traits: `eval_results/issue_1482/lens_reads/lens_reads.json`. Note the worst-20 "R²" here is read at the top-256 selection grain; the tuned lens was fit on LMSYS positions, val-KL 71% below logit lens; J-lens is the community J₁₉ artifact.)

**Takeaways:**
- The best predicted directions are much more aligned with SAE features than the worst predicted directions: \|cos\| **0.29–0.63 (3–6.6× the random-direction null)** with coherent macro-structure labels (code, foreign languages, Russian, business register) vs the worst-20 at **0.11–0.14 (~1.5× null)** with grab-bag labels
- **No lens rescues the worst directions**: logit lens, tuned lens, and J-lens are all illegible on the worst-20 (and on the below-floor-curve deviant band) — whereas the same J-lens makes the persona-vector *trait* directions cleanly legible (evil → "useless, fake, worthless"; sycophancy → "wonderful, amazing"), so the illegibility of the worst directions is a property of those directions, not of the lens

This led me to ask: since the mapping we are training is linear, is the part of the residual stream it is able to predict **the same** as the part of the residual stream an SAE (which finds linear features) is able to reconstruct. To test this, I compared the subspace of the answer vectors which is:
- map predictable
- SAE representable

I ranked the same answer-PCA directions by (a) per-direction map $R^2$ and (b) per-direction SAE reconstruction FVE, compared the top-k subspaces by principal-angle overlap against **variance-matched random-subspace nulls**, and computed the per-direction correlation with variance partialled out (#1895):

![Overlap of the map-predictable and SAE-representable top-k subspaces vs the variance-matched null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1895/hero_overlap_ksweep.png)

**Takeaways:**
- Yes — but almost entirely **because both select variance**. The top-16 subspace overlap is 0.855, at the **61.5th percentile of the variance-matched null** (i.e. exactly what matched-variance random subspaces give); per-direction ρ(map $R^2$, SAE FVE) = **0.97 raw but 0.076 [0.038, 0.090] after partialling variance rank** (`eval_results/issue_1895/correlates_summary.json`)
- So "map-predictable" and "SAE-representable" pick out the same subspace, but the common cause is variance, not a shared notion of linear feature structure

To get a better idea of **exactly which interpretable aspects of the answer** are better predicted by the mapping, I then trained a mapping directly from continuous context vector to SAE features (average activation over answer), and tried to understand which SAE features were predicted the best/worst. I considered a few predictors:
- Variance explained by that direction (in average answer activation space):
    - Two quantities hide under this name and they are **uncorrelated with each other** (ρ = 0.004): (a) the feature's **activation variance** (how much its own activation value fluctuates across contexts) — **null** vs $R^2$, ρ = **0.02** (`sae_perfeature/variance_vs_r2.json`); (b) the **dense-state variance along the feature's unit decoder direction** — the true PC-analogue — which is a real correlate: ρ = **+0.40** (+0.45 given activity; cross-corpus caveat: projection variance measured on the multi-turn holdout, `feature_correlates/dense_projection_variance.json`). So the variance-mechanical story DOES extend to SAE feature directions once variance is measured in the dense space the direction lives in; what carries no signal is the feature's own gating variance
- Average activation across corpus:
    - ρ($R^2$, activity) = **0.29** — more-active features are moderately better predicted
- Interpretable or not (judged axis, κ 0.633): 77.9% of panel features judged interpretable; panel-wide ρ($R^2$, interpretable) = **+0.045** (n = 16,285 label-resolved, prevalence 0.793), **outside** the activity-stratified permutation band [−0.009, +0.019], p = 0.0005. The tail sweep shows a **sign flip, not a monotone effect**: interpretable features are *depleted* in the very best-predicted tail (Δ₂₅ = −0.120) and *enriched* from k ≈ 300 outward (Δ ≈ +0.07, still +0.032 at k = 8,000 = half the panel, scan-corrected). Jointly it is stronger than raw: all-others-partial ρ = **+0.117** (`eval_results/issue_1482/predictor_battery/`)
- LLM judged predictors:
    - Methodology (correcting the draft's recollection): **5 judge draws per feature per axis, aggregated by modal label** — a malformed/refusal draw is dropped (never coerced), and a feature is excluded from an axis only if <2 draws survive (0.2–0.5% of items). Inter-draw agreement is reported as Fleiss κ per axis. Content-drop rates per axis: 0.5–0.6% of draws (functional_role the outlier at 2.4%). Same instrument was later run on the FULL dictionary (128,512 features × 5 axes × 5 draws = 3.21M Batch-API calls, $9.5k) with κ replicating within ±0.03.
    - Level of abstraction (κ 0.68; token_surface 27.6% / lexical_semantic 24.8% / abstract_contextual 47.7%): panel-wide **null** (ρ = −0.057); separates only the extreme tails (below-curve depth ~top/bottom 400 scan-corrected of 16,381) and the signal is **worst-tail depletion of abstract features** (0.38 in the worst 5% vs 0.476 marginal), not best-tail enrichment
    - Content type (κ 0.66; syntax 76.6% / topic 11.6% / operation 8.5% / entity 1.9% / task_format 1.3%): one-vs-rest panel-wide, **the two big classes are pure activity artifacts and the two rare ones are real**. `syntax` ρ = +0.042 sits INSIDE its activity-stratified band [+0.025, +0.054] (p = 0.35) and `topic` ρ = −0.020 inside [−0.036, −0.007] (p = 0.60) — the apparent syntax-is-better-predicted effect is entirely explained by syntax features firing more often. Outside their bands at p = 0.0005: `operation` ρ = **−0.077** (the clearest negative — operations are the *worst*-predicted content class), `entity` **+0.053**, `task_format` **+0.032**, each separating to the full k = 8,000 scan-corrected depth
    - Related to the **language of the text** (13.3% of panel): purely a **best-tail spike** — prevalence 0.255 in the best-predicted 5% vs 0.103 marginal, flat elsewhere (language identity is near-deterministic given the query)
    - Related to the **identity of the speaker** (0.7% of panel): too rare for a stable tail read at panel size
    - Related to the **register of the speaker** (8.7%): a **worst-tail depletion** effect — flat across top/middle, drops to 0.035–0.07 in the worst bins vs 0.075 marginal (register features are never among the unpredictable)
- high-level vs low-level features (defined in 2 ways):
    - Presence in higher or lower level dictionary of Matryoshka SAE: **coarse-better** — tier-vs-$R^2$ Spearman **−0.395**, outside the within-activity-stratum permutation band [−0.250, −0.228], partial given log-activity −0.194; per-tier median $R^2$ 0.435 (coarsest) / 0.174 / 0.043 (finest) (`eval_results/issue_1482/matryoshka_tier/tier_tests.json`)
    - Feature continuance (= within-answer consistency): for each answer where the feature fires at all, the fraction of that answer's token positions on which it is active, averaged over held-out answers (n≥8-token variant nearly identical). This is the quantitative high-level/tonic vs low-level/phasic measure: a persistent "speaking French"-style property stays on across the whole answer; a token-triggered detector blips. **The strongest predictor found**: ρ = **0.600** ($R^2$), partial given activity **0.582** (`eval_results/issue_1482/feature_correlates/consistency.json`)
- Decoder vector norm: γ-scaled write norm ρ = **+0.18** (+0.22 given activity, +0.21 given consistency) — features that write more strongly are slightly better predicted (`eval_results/issue_1482/footprint_moments/footprint_moments.json`)
- Encoder vector norm: raw ρ = **+0.059** (near-null on its own) but all-others-partial ρ = **+0.127** — it is the 5th strongest predictor once the rest are held fixed, i.e. a *suppressed* correlate rather than an absent one. The encoder/decoder asymmetry question is moot for this SAE: the **decoder columns are unit-norm by construction** (measured std 2.7e-08 over the panel), so "decoder norm" carries zero rank information and the enc/dec ratio is just the encoder norm rescaled. The informative write-strength quantity remains the γ-scaled write norm
- Input vs output features:
    - Two operationalizations. (1) **Judged axis** (`functional_role`: input_side / output_promoting / mixed, from activation evidence): RETIRED — inter-draw κ 0.318 on the panel AND 0.318 at full dictionary; a 4-arm rubric repair (+contrastive negatives, +nearest neighbours, +token budget) moved κ ≤ +0.04, and the literature explains why: output/causal role is not readable from input-side activation evidence. (2) **Mechanical replacement** (output-footprint moments: skew/kurtosis/variance of $W_U(\gamma \odot W_{dec})$ per feature, direct AND routed through J₁₉; Gurnee-style promoting/suppressing/partition classes; κ = 1 by construction): classes exist cleanly (8,063 promoting / 5,045 suppressing / 8,716 partition of 131,072; direct-vs-J₁₉ class agreement 0.80) — but the **pre-registered prediction that output-promoting features are worse-predicted is refuted**: panel ρ(promoting, $R^2$) = −0.026 (−0.044 given activity), median $R^2$ 0.134 (promoting) vs 0.150 (other); the full-width read is +0.06 but activity-confounded

For continuous predictors, the ρ's with pairwise partials (activity / consistency; the joint all-others-partialled model is below; per-feature reads are ridge-only, because the banked MLP per-feature array is the DIVERGED fit — all 16,384 values negative, recorded in `sae_sae_mlp_recovery/final.json` as `broken_mlp_reference` — and the recovery round emitted pooled scalars only; at the PC grain ridge-vs-MLP rank agreement is 0.997, so the ranking is not expected to move):

| predictor | ρ vs $R^2$ | partial given activity | partial given consistency |
|---|---|---|---|
| within-answer consistency | **+0.60** | +0.58 | — |
| activity | +0.29 | — | ~+0.24 |
| γ-scaled decoder (write) norm | +0.18 | +0.22 | +0.21 |
| footprint kurtosis (output-coherence) | −0.03 | −0.06 | −0.13 |
| dense variance along decoder dir | +0.40 | +0.45 | — |
| feature activation variance | +0.02 | +0.13 | +0.04 |
| Matryoshka tier (fine = high) | −0.39 | −0.19 (stratified) | — |

**Takeaways:**
- Four things order SAE-feature predictability: **within-answer persistence (dominant, 0.60), dense variance along the feature's direction (0.40), firing frequency (0.29), and (weakly) write strength (0.18)**. The feature's own activation variance, judged abstraction, and output-coupling do essentially nothing
- The mechanical input/output axis is measurable and cheap but does not carry the input-echo-easy / output-promoting-hard hypothesis — that hypothesis fails at the feature level

### The joint model: what survives partialling everything else (run 2026-08-03)

All of the above are pairwise reads, and the predictors are correlated with each other, so the pairwise column both over- and under-states. Fitting **one rank-OLS of per-feature $R^2$ on all 26 predictors at once** (11 continuous + 15 one-hot judged levels; 'unresolved' carried as its own level so no feature is dropped) gives the all-others-partialled picture, with leave-one-predictor-out $\Delta R^2$ as the dominance read and 2,000-draw feature-bootstrap CIs:

![Raw vs all-others-partial Spearman for every per-feature predictor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/joint_partial_forest.png)

| predictor | raw ρ | all-others-partial ρ [95% CI] | LOPO ΔR² |
|---|---|---|---|
| within-answer consistency | +0.600 | **+0.558** [+0.547, +0.569] | 0.217 |
| activity (firing frequency) | +0.293 | **+0.291** [+0.276, +0.305] | 0.044 |
| footprint kurtosis | −0.028 | **−0.161** [−0.176, −0.146] | 0.013 |
| dense variance along decoder dir | +0.404 | +0.135 [+0.119, +0.149] | 0.009 |
| encoder-vector norm | +0.059 | +0.127 [+0.111, +0.141] | 0.008 |
| interpretable: yes | +0.045 | +0.117 [+0.103, +0.132] | 0.007 |
| abstraction (ordinal) | −0.061 | −0.093 [−0.108, −0.076] | 0.004 |
| γ-scaled write norm | +0.184 | +0.085 [+0.070, +0.101] | 0.004 |
| side ratio | +0.119 | +0.061 [+0.045, +0.076] | 0.002 |
| rb_align (max over 3 traits) | +0.203 | −0.029 [−0.044, −0.013] | 0.0004 |

**Takeaways:**
- **Joint rank-OLS $R^2$ = 0.520** [0.510, 0.532]. The ordering is not the pairwise ordering: three of the top five partial predictors are ones the pairwise table calls null or near-null
- **Consistency does not subsume the rest, and the rest do not subsume consistency.** Consistency alone reaches $R^2$ 0.360; the other 25 predictors without it reach 0.303; together 0.520 — so ~0.14 of $R^2$ is shared and each side carries substantial independent signal. Consistency's own LOPO ΔR² (0.217) is 5× the next predictor's
- **Two suppressed correlates surface only in the joint fit**: footprint **kurtosis** goes −0.03 → **−0.161** (a 6× amplification, making output-coherence the 3rd strongest predictor — the pairwise "output-coupling does essentially nothing" line above holds only marginally) and **encoder norm** +0.059 → +0.127. Both are masked pairwise by their positive correlation with activity/write-strength
- **Two pairwise correlates are mostly mediated**: dense variance along the decoder direction collapses +0.404 → +0.135 (three-quarters of it was activity and consistency), and **rb_align collapses +0.203 → −0.029, i.e. to nothing** — its pairwise correlation is entirely shared with the dense-variance predictor (pairwise ρ between them = 0.58)
- The RETIRED `functional_role` axis (κ 0.318) is carried in the fit for completeness and flagged in the figure; its partial ρ (+0.072 for output_promoting) is not interpretable as a finding

The same round also asks how well each continuous predictor **classifies** tail membership (AUROC for top-k vs bottom-k by $R^2$) and extends the tail-depth sweep to the two judged axes that had never been run:

![Tail-depth sweep for the new judged axes, and AUROC of each continuous predictor for top-k vs bottom-k membership](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/tail_auroc_extension.png)

**Takeaways:**
- **Consistency is a near-perfect tail classifier and stays strong at every depth**: AUROC 0.97 at k = 25, still 0.95 at k = 1,600 and 0.80 at k = 8,000 (half the panel). Dense variance is comparable in the extreme tail (0.97 at k = 25) but decays much faster (0.70 at k = 8,000) — consistent with it being the mediated predictor in the joint fit
- **Activity behaves oppositely to every other predictor**: its AUROC *rises* with k (0.64 at k = 25 → 0.77 at k = 1,600) — firing frequency separates the broad middle, not the extremes
- Write norm (0.57 → 0.67 → 0.59) and encoder norm (0.57 → 0.53) are weak classifiers at every depth despite being real joint predictors — a small partial ρ over 16,384 features does not buy tail separation

(`eval_results/issue_1482/predictor_battery/joint_model.json`, `tail_extension.json`; the four banked per-predictor ρ's are re-derived as wiring gates before any read, and the closed-form LOPO agrees with explicit refits to 5e-16)

For binary predictors, effectiveness was checked three ways: (1) two-arm extremes contrasts — top-150 vs bottom-150 by $R^2$, Fisher-exact odds ratio, repeated **activity-controlled** (15 best/worst within each activity decile) with a stratified permutation test; (2) full-panel Spearman of the coded label; (3) a **tail-depth sweep**: Δ_k = frac(label | top-k) − frac(label | bottom-k) over 16 tail widths against an activity-stratified permutation null with a scan-corrected (studentized max-T) band, plus prevalence-vs-rank profiles:

![Tail-depth sweep: how deep into the R² ranking each judged binary label separates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/feature_correlates/tail_depth_sweep.png)

**Takeaways:**
- Tails-only signals and panel-wide signals look identical in an extremes contrast (abstraction OR ~3–4, speaker OR ~8) but completely different in the sweep: **abstraction dies by k ≈ 400–600; speaker_property separates at every depth tested (k = 8,000, half the panel)**
- The prevalence profiles identify the mechanism: language = best-tail spike; register = worst-tail depletion; abstraction = worst-tail depletion of abstract features

I then asked Claude Fable 5 to try to find the common thread between the top 100 and bottom 100 predicted SAE features (based on autointerp metadata). Protocol (blinded re-run, 2026-08-03): a fresh instance saw ONLY two groups of 100 autointerp descriptions labeled Group A / Group B — the A/B↔top/bottom assignment was randomized and the key kept on disk unread until after it reported; it was not told what the groups were, how they were selected, or what "better" would mean. Its verdict (unblinded afterward: A = bottom-100, B = top-100):

> **Bottom-100 (worst predicted)**: features defined by **what the token IS** — its form, morphology, or single lexical meaning, with the surrounding text incidental. ~27/100 are crisp single-lexeme concepts held invariant across languages (the word "red", "low/reduced/minimal", "example", "middle", ordinals, the digit 1 opening a numbered list); a large cluster names exact affix classes and tokenizer positions (-tion/-ment/-ization, un-/non-/im-, mini-/micro-/anti-, camelCase components, the stem position immediately before a suffix); plus list/enumeration formatting tokens and a negation-form feature.
>
> **Top-100 (best predicted)**: features defined by **what the CONTEXT IS** — ~24/100 language/script identity (11 Chinese-specific, plus Cyrillic, Romance, gender marking), ~15 register/genre (formal business, professional/academic/institutional, instructional, assistant-like responses), ~11 discourse-POSITION slots (the token just before an enumerated list, the header-to-body transition, the period closing a caveat), and abstract institutional vocabulary (innovation, sustainability, quality). Concrete/scalar lexemes are absent.
>
> **Sharpest contrast**: token-intrinsic vs context-extrinsic definition — Group A means "this concept regardless of language", Group B means "because the text is in this language/register/slot". Honest caveat: ~40–50% of each group is generic low-level structural filler (subword fragments, punctuation, programming syntax) genuinely indistinguishable between groups, with a handful of clear cross-fits both ways.

The earlier unblinded pass (2026-08-02) read the same way:

> **Top-100 (best predicted)**: almost uniformly *properties of the text stream that the context fixes in advance* — grammatical function words, conjunctions and articles; punctuation and clause/list/paragraph boundary markers; word-internal continuation fragments (BPE suffixes); language and script identity (Chinese characters, non-English morphemes, Russian); and formal/professional register. Essentially format, syntax, language, and register scaffolding — nothing about specific semantic content.
>
> **Bottom-100 (worst predicted)**: dominated by *specific-token-identity* features — the word "red", the word "example", words meaning "middle"/"general"/"high", the digit 1, ordinals and list indices; short 2–4-character stems and mid-word fragments whose firing depends on exact BPE segmentation; negation/qualification/limitation tokens; and error/problem terminology. Predicting these requires knowing *which exact words the answer will use* — token-level choices the context constrains only weakly. The negation cluster is the most interesting member: whether the model qualifies or negates is a content decision, not scaffolding.

I looked at this metadata myself (you can find it [here](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/eval_results/issue_1482/result2_assembly/top_bottom100_descriptions.json)) and... *[your read here]*

---

## New since the draft: is the J-space preferentially predicted? (run 2026-08-02)

J₁₉ = E[∂h_final/∂h₁₉] linearizes what layers 20–27 transmit to the output; its top right-singular directions are the "workspace" the downstream computation actually reads. Per-direction $R^2$ along J₁₉'s singular basis, ridge AND MLP, all three arms:

![Map R² vs J-transmission strength](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/jspace_r2/jspace_r2.png)

- **Pooled, yes**: the top-64 J-subspace (11% of target variance) is predicted at $R^2$ 0.78 vs 0.67 for its complement (context, ridge; MLP 0.81 vs 0.69; consistent across arms)
- **But that is a variance effect, and at matched variance the sign flips**: partial ρ($R^2$, log s | log share) = **−0.48** (context, both fitters) — at fixed variance, *more-transmitted directions are predicted worse*. The decile curve is U-shaped: the least-transmitted decile is also well predicted (context-echo variance the downstream computation discards)
- This is the first positive characterization signal in the hunt, and it points the same way as the bottom-100 digest: **what the downstream computation actually uses (at fixed size) is exactly what the context underdetermines**
- Caveat: J₁₉ is the community artifact (50 wikitext prompts, off-distribution for chat)

(`eval_results/issue_1482/jspace_r2/jspace_r2.json`)

## New since the draft: the FULL-DICTIONARY battery — and why the panel ordering was an artifact (run 2026-08-03)

Everything above is measured on the 16,384-feature answer-side panel. Re-running the whole battery on the **full dictionary** — 114,980 features (128,482 judged × 131,072 $R^2$, intersected with finite context-arm $R^2$ and answer-active-at-least-once) — reverses the headline.

**The panel is the top-activity slice of the dictionary, not a sample of it.** 100% of non-panel features fall below the panel's *minimum* activity (0.0810); the panel's 1st activity percentile (0.0819) sits above the non-panel 99th (0.0761). The panel therefore compresses activity into a ~9× range where the full dictionary spans ~4 orders of magnitude (3e-5 to 0.77). That range restriction is what produced the panel's ordering:

| | ρ($R^2$, activity) | ρ($R^2$, consistency) |
|---|---|---|
| panel features (16,382) | +0.365 | **+0.472** |
| non-panel features (98,598) | **+0.652** | +0.192 |
| full dictionary (114,980) | **+0.742** | +0.257 |

Both rows are scored against the SAME multi-turn $R^2$, so this is a range-restriction effect, not a corpus effect.

![Full-dictionary raw vs all-others-partial Spearman for every predictor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/summary_forest.png)

**Takeaways:**
- **Joint rank-OLS $R^2$ = 0.594** [0.591, 0.598] over 24 predictors; the same model classifying top-decile-vs-rest $R^2$ membership scores **AUROC 0.903** [0.901, 0.906]
- **Activity is the whole story at full width**: raw ρ **+0.742**, all-others-partial **+0.671**, LOPO Δ$R^2$ **0.333** (top-vs-bottom-decile median-$R^2$ gap +0.127). Every other predictor's Δ$R^2$ is below 0.01
- **Within-answer consistency — the panel's dominant predictor — collapses to nothing**: raw +0.257, partial **+0.014**, Δ$R^2$ **0.00007**. It reaches $R^2$ 0.066 alone, and *removing it changes the joint $R^2$ by 0.0001*. The panel's ρ = 0.60 was activity wearing a different hat
- Runners-up are all small: `content_type: topic` partial −0.152, dense variance +0.134, write norm −0.100, `interpretable: yes` +0.083. Encoder norm (raw −0.261) and footprint kurtosis (raw +0.235) both go to ~zero partialled (−0.010, −0.044) — the panel's "suppressed correlate" reading of kurtosis does not survive
- Distance correlation (8,000-feature subsample) exceeds |ρ| by >0.05 for **no** predictor, so nothing here is a hidden non-monotone relationship
- Panel↔full-width rank agreement on the 16,382 shared features is ρ = **0.772** — but across DIFFERENT corpora (panel $R^2$ single-turn, full-width $R^2$ multi-turn), so it is an agreement check, not a replication

For the judged labels, the activity-stratified null is the whole point: several labels look informative raw and are entirely activity, and one looks null and is not.

![AUROC at depth for every judged binary label](https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_1482/predictor_battery/summary_auroc_depth_overlay.png)

| label | prevalence | AUROC [95% CI] | stratified null mean | direction | perm p |
|---|---|---|---|---|---|
| speaker_identity | 0.012 | 0.646 [0.629, 0.663] | 0.492 | above | 0.0005 |
| content_syntax | 0.650 | 0.576 [0.573, 0.580] | 0.520 | above | 0.0005 |
| abstraction_high | 0.356 | 0.569 [0.565, 0.572] | 0.562 | above | 0.0005 |
| content_topic | 0.187 | 0.416 [0.412, 0.420] | 0.508 | below | 0.0005 |
| speaker_language | 0.058 | 0.576 [0.569, 0.584] | 0.587 | **below** | 0.0005 |
| interpretable | 0.850 | 0.489 [0.484, 0.493] | 0.424 | **above** | 0.0005 |
| content_task_format | 0.016 | 0.467 [0.453, 0.483] | 0.422 | above | 0.0005 |
| content_entity | 0.051 | 0.400 [0.393, 0.407] | 0.394 | above | 0.033 |
| content_operation | 0.097 | 0.507 [0.502, 0.513] | 0.507 | above | 0.650 |

**Takeaways:**
- **The stratified null is nowhere near chance**, so a raw AUROC is uninterpretable on its own. `interpretable` reads 0.489 — apparently sub-chance — but its activity-stratified null sits at 0.424, so interpretable features are in fact **enriched** among the best-predicted *beyond* activity. `speaker_language` reads 0.576 — apparently the strongest language effect — but its null is 0.587, so the language enrichment is **entirely activity, and slightly less than activity alone predicts**
- **speaker_identity is the one large genuine effect** (0.646 vs a 0.492 null) despite being the rarest label (1.2%) — the panel called it "too rare for a stable tail read", and at 7× the n it is the cleanest signal on the board
- **content_operation is the only true null** (0.507 vs 0.507, p = 0.65) and, with content_entity, the only label whose separation does not survive the scan-corrected band at any depth
- Every other label separates to the deepest tested depth ($k$ = 51,200 ≈ half the dictionary) — at n = 115k the tails-only-vs-panel-wide distinction the panel sweep drew mostly dissolves

Per-predictor plots (one scatter per continuous predictor, one prevalence-vs-rank profile per label) are in `figures/issue_1482/predictor_battery/per_predictor/`.

**Caveats.** (1) The full-width $R^2$ is the **#1738 multi-turn** read while consistency, activity and projected variance come from the **#1482 single-turn** corpus — every full-width correlate is cross-corpus (the same caveat the panel's projected-variance read already carried). (2) `side_ratio` and `rb_align` exist only in the #1773 panel table and are absent from the full-width model. (3) The retired `functional_role` axis (κ 0.318) is carried and flagged; its partials are not interpretable. (4) Consistency and activity were re-derived full-width from the local pooled store and **reproduce the committed panel covariates exactly** (max |Δ| = 0.000e+00 for both, n_fit = 120,000).

(`eval_results/issue_1482/predictor_battery/fullwidth_joint_model.json`, `fullwidth_label_reads.json`, `fullwidth_matrix.npz`)

## Suggested additional analyses

1. ~~**Full-width label joins** — the 3.21M-call full-dictionary judged labels (128,512 features × 5 axes, banked on HF + `/mnt/eps-data`) × the full-width per-feature $R^2$ (131,072, banked): rerun the tail-depth sweep and panel correlations at ~7× the n, incl. deriving a full-width activity covariate for the stratified nulls. Zero new API calls.~~ — **DONE 2026-08-03, predictor_battery full-width round** (see the full-dictionary section above; the full-width activity AND consistency covariates were derived from the local pooled store and identity-gated against the committed panel values).
2. ~~**Interpretable-vs-$R^2$ and content_type reads** — both labels banked, correlation never run.~~ — **DONE 2026-08-03, predictor_battery round** (panel reads + tail sweep + AUROC extension).
3. ~~**Joint predictor model** — the draft's requested all-others-partialled ρ plot: one multivariate (rank) regression of per-feature $R^2$ on consistency + activity + write norm + tier + labels, with dominance analysis; also settles how much consistency subsumes the rest.~~ — **DONE 2026-08-03, predictor_battery round** (26 predictors, joint $R^2$ 0.520; Matryoshka tier excluded as a different dictionary/panel that does not join).
4. ~~**Encoder-norm + encoder-vs-decoder asymmetry** correlates (encoder weights banked, never used for this).~~ — **DONE 2026-08-03, predictor_battery round**; the asymmetry half is moot (decoder columns are unit-norm by construction).
5. **Nonlinear per-feature reads** — the SAE-feature target maps are ridge-only; PC-grain ridge-vs-MLP agreement (0.997) suggests little difference, but the check is cheap. **Now also needed to un-block the joint model's MLP twin**: the banked per-feature MLP array is the diverged fit and the recovery round emitted pooled scalars only, so no recovered per-feature MLP $R^2$ exists to re-run the battery against.
6. **Steering validation of the footprint classes** — TokenChange on a ~300–500-feature stratified sample (~1–2 GPU-h): does the promoting class actually gain its top-footprint tokens when clamped? (Also the calibration set for any future causal-role axis.)
7. **J-space × consistency mediation** — does J-transmission explain the consistency→$R^2$ link, or are they independent? (One partial-correlation pass on banked arrays.)
8. **Interaction structure** — Result 1 says the miss is interaction-dominated: low-rank SVD / biclustering of the normalized miss table to find context×direction blocks (e.g. "code contexts miss language directions").
9. **Per-context tail-depth** — the Result 4 mirror of the sweep (how deep into the context ranking do context-side labels separate).
