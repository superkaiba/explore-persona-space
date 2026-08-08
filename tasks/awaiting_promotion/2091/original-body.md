---
title: 'Deterministic vs stochastic decoding: answer-vector variance, mapping quality,
  and behavioral-expression prediction'
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-05T19:42:24Z'
has_clean_result: false
parent_id: 1073
origin_prompt: 'user chat 2026-08-05: ''We''ve been using stochastic vs deterministic
  decoding for different experiments. I wanted to do a principled comparison.'' (full
  5-result body drafted interactively across the session; closing dispatch: ''run
  in background with happy coder and setup periodic monitor'')'
workflow: v1
goal: Measure how the answer-decoding regime (single greedy vs K-rollout-averaged
  stochastic vs single stochastic draw) changes answer-vector variance, context->answer
  mapping quality and its cross-regime transfer, and behavioral-expression prediction
  accuracy for sycophancy / hallucination / evil, on generic held-out data versus
  trait-eliciting data.
relates_to:
- spec-context-as-vector
---
## Goal

Measure how the answer-decoding regime (single greedy vs K-rollout-averaged stochastic vs single stochastic draw) changes answer-vector variance, context->answer mapping quality and its cross-regime transfer, and behavioral-expression prediction accuracy for sycophancy / hallucination / evil, on generic held-out data versus trait-eliciting data.


## Motivation

We've been using stochastic vs deterministic decoding for different experiments. I wanted to do a
principled comparison.

## Methodology

- Run mapping prediction for:
    - deterministic decoding (greedy, temperature 0)
    - stochastic draws averaged across samples ($\bar{v}_A^{\,r}$)
    - single stochastic draw (sampled from the above)
- Look at:
    - average difference between answer vectors for each
    - effect on mapping training itself
    - variance of answer vectors across contexts
    - variance of behavioral expression judgement for our 3 behaviors (sycophancy, hallucination,
      evil) on **generic held-out data** vs **specific trait-eliciting data** (one trait-eliciting
      dataset per behavior), and how this relates to the **average behavioral expression of a
      prompt**
    - is the mapping particularly bad at predicting contexts where the variance of the answer is
      high?

### Decisions taken at planning (2026-08-05 chat, do not re-litigate)

1. **Transfer axis = decode regime.** Result 4's "transfer to other setting" means: fit the map on
   one target regime, evaluate against another. Three regimes (greedy / averaged-stochastic /
   single-stochastic) ⇒ **3 fits, a 3×3 held-out R² grid per setting**. Diagonal = matched,
   off-diagonal = transfer. NOT corpus transfer.
2. **The greedy arm is generated on the trait-eliciting data.** #1073's greedy exists only on
   generic LMSYS. Result 2 explicitly contrasts generic vs trait-eliciting, so that comparison IS
   controlling for dataset and the reuse note does not cover it. ~2,000 contexts per behavior ×
   setting, one greedy decode + capture each.
3. **No map is ever fit on trait-eliciting data alone.** Every map pool is generic-dominated with a
   target-domain admixture (decision 4). This is what makes the fits well-posed: fitting a map on a
   single trait rung would be under-determined almost everywhere — $d = 3{,}584$ against 671–4,021
   contexts on every eval / OOD / WildChat rung, and $n/d \approx 1.6$ even on evil's train rung
   (crossed prefix × question, only 1,348 distinct prefixes) — and every held-out R² in that regime
   is estimator-degenerate, not a signal read (#1701). The generic half of the pool removes the
   problem entirely, so **there is no thin-map caveat and no per-behavior-map-vs-frozen-map
   decision to make.** Report the frozen 963k generic map (#779/#1092, `map963k_reuse`) as a
   reference line — it is the $f_u = 0$ end of the same axis, so it comes free and makes the
   admixture's contribution readable.
   Ridge hygiene still binds: dof-capped, never pure GCV (#1887), with the per-fit selector and
   selected λ reported alongside every read.
4. **OOD protocol = generic + OOD in the map pool, evaluated on held-out OOD.** This is #1739's R5
   machinery (replace a fraction $f_u$ of the generic WildChat pool with unlabeled target-domain
   contexts; verified inert at $f_u = 0$). Default $f_u = 0.5$. It also dissolves the $n < d$
   problem, since the generic half supplies the bulk.
   - The **matched all-generic-pool arm is mandatory**, not optional. R5 measured this lever and it
     is behavior-dependent and signed: evil gained +0.209 → +0.504 at U=250 (Δ +0.295) while
     sycophancy and hallucination lost slightly (−0.007 to −0.123), consistently in sign. Without
     the control you cannot separate the pool from the setting.
   - "Trained on generic + OOD" means the **unlabeled map pool $U$ only**, not the labeled readout
     set $L$. Mixing $L$ too is a second variable.
   - R5's own caveat is that it has no OOD cells ("none of this is a transfer claim"). This design
     fills exactly that gap.
   - **Size the U rung to the available target-domain pool, do not force $f_u = 0.5$ at large U.**
     R5 swept U at 250 / 2,500 / 18,793; at U = 18,793 an $f_u = 0.5$ admixture needs ~9,400
     target-domain pairs, which evil's OOD rungs do not have (hhrt 1,995, toxicchat 671). Pick the
     largest U each rung can supply at the registered $f_u$, and report the realized (U, $f_u$) per
     cell rather than silently degrading the fraction.
   - The admixed pool contexts must be **disjoint from the held-out eval contexts** — the admixture
     is unlabeled map data, never a peek at the eval set.
   - Held-out OOD split is **group-level** on `group_key`, never row-level.

### Deviation (stated) — context-end state only

Every mapping arm here is **context-based** — $v_C$ = the last-prompt-token (context-end) state of
prefix + query. Neither prefix-side object is run: the **prefix-end state** is the constant
chat-template string on every single-turn rung, so that arm would be the degenerate constant-input
floor #1073 already measured, and the **query-averaged prefix vector $v_P$** is not constructible
where no prefix is shared across queries. Evil's train rung (1,348 DAN-style prefixes × questions)
is the one cell where both objects exist and vary; deliberately out of scope. Per the CLAUDE.md
standing rule "Prefix mapping AND context mapping", this is an explicit stated deviation and must
be **carried into the clean-result as a scope caveat**. Repeat it in Result 4's caveats, since the
evil-train cell is where a reader will reasonably ask for the prefix arm.

### Reuse (the note: reuse as many artifacts as possible)

Most of this is 0 GPU on banked artifacts. Verify each against
`.claude/rules/artifact-reuse.md` (a)–(l) before consuming.

| what | where | covers |
|---|---|---|
| 3 decode arms × 28 layers × n=5,000 LMSYS, per-context `sse`/`sst`/`cos`, `dv4_delta_ctx`, LOO-9 references, rollout dispersion, noise decomposition | `eval_results/issue_1073/` (`target_agreement.json`, `heldout_recon_percontext.json`, `gap_noise_decomposition.json`, `adequacy_tail_characterization.json`) | Results 1–2 and 4 on **generic** data, nearly complete |
| per-`(context, rollout_k, draw_idx)` judge scores, all 3 behaviors (sycophancy 86,520 rollouts / hallucination 57,910 / evil 53,330, ×3 judge draws) | `eval_results/issue_1739/judge_reliability/draw_matrix_*.npy` + `per_draw_manifest.json` | judge-noise separation |
| per-context `per_rollout_scores` (K=5) on every trait rung **and** the generic WildChat rung | `eval_results/issue_1739/{dv_dataset,wildchat_rung/dv_dataset}/<behavior>/labeling.json` | Results 3 and 5, generic vs trait-eliciting, 0 GPU |
| per-rollout answer vectors (`t1` rows keyed by `rollout_k`) | #1739 labeled store — `issue1739_fits.py:525` currently means them over rollouts; the per-rollout rows are there and unused | Result 3 plot 2, $\sigma_A$ |
| answer-sampling floor by conversation depth, n=1,988, k=4 | `eval_results/issue_1738/kresample/floor_summary.json` | **Result 1 plot 2, already answered** |
| answer-sampling floor, single-turn generic, n=2,000, k=4 | `eval_results/issue_1482/kresample/` | Result 1 plot 3 reference |
| frozen 963k generic map | #779/#1092; reused in `eval_results/issue_1739/map963k_reuse/` | Results 1/4 reference line |
| R5 pool-composition machinery | #1739 `gapfold/` | the OOD protocol above |

**New compute is only:** the greedy arm on trait-eliciting data (~20k generations, 1–2 GPU-h under
vLLM) + its judge wave (~60k Batch-API calls). **That is the complete new-compute budget for this
round** — no other generation is in scope (see Result 5: the K=20 subsample is explicitly excluded).

### Pre-registrations and standing assumptions

- **K = 5 throughout, no exceptions** (the banked #1739 grain; the K=20 subsample is OUT OF SCOPE per the user directive — see Result 5).
  #1073's generic arm is K=10 — where the two meet, K is a **stated deviation, never silent**.
  Every Δ / variance statistic scales with K, so bars at different K are not comparable.
- **Read-out layers frozen in advance** (L14 / L19 / L26, per #1738). Not selected on results — a
  max-over-layers read would need a selection-symmetric null
  (`.claude/rules/selection-symmetric-nulls.md`).
- **Cluster-bootstrap on `group_key` everywhere**, never row-level. Design effects: evil train 6.9,
  hallucination train 3.6, simpleqa 2.1, nqopen 1.4; sycophancy and all WildChat rungs 1.0. Evil's
  rung is *crossed* (prefix × question) and `group_key` carries only the prefix axis — either pull
  question ids from the staged `inputs/` for two-way clustering or state that the question axis is
  uncorrected.
- **Evil is floor-censored on generic data** (SD 4.43, 98.9% bottom bin) and on hh-rlhf (ICC 0.18,
  ρ ceiling 0.71, ~0 middling contexts). Those cells are **reported as uninformative, not plotted
  as noise**.
- **Hallucination carries two non-comparable constructs**: per-rollout 3-way fabrication rate on its
  own rungs, 0–100 graded trait score on WildChat. Never on one axis without saying so.
- **Generation config for the greedy arm** matches #1073 other than temperature: `max_tokens=1024`
  (not the current 2048 default — recipe fidelity with the parent; state the deviation either way).
  Report the realized cap-hit fraction; #1073's greedy truncated at 6.3%.
- **The ~60k-call judge wave is pilot-gated** (`.claude/rules/llm-judging.md` rules 23/26): ~150
  draws spanning the arms at the exact production instrument, gate on zero `max_tokens` stop reasons
  and per-arm parse-fail < 2%.
- **Judge-noise correction is a footnote, not a pipeline.** Measured from `draw_matrix`: the judge
  returns identical scores across its 3 draws for ≥50% of rollouts (median contributed SD 0.00,
  mean 2.7–3.7 points) — ~2–4% of variance against observed within-context SDs of 10–22. Report the
  number; only subtract if a result sits inside that margin.

## Results

### Result 1: Variance of stochastic draws

I first wanted to see what the variance of the answer vectors for a fixed context vector was, on
generic and specific trait-eliciting data.

[INSERT PLOT: average dispersion of answer vectors (mean pairwise cosine distance among the K
rollout vectors — scale-free), one bar per setting, generic and trait-eliciting. Median answer
length reported alongside each bar, since length confounds cross-corpus comparison.]

I then wanted to see, in multi-turn conversations, does the variance of the answer vectors go up
over time?

[INSERT PLOT: answer-sampling floor share vs conversation depth. **Largely banked** —
`eval_results/issue_1738/kresample/floor_summary.json`, n=1,988, k=4, reads FLAT (L19: 0.063 at
depth 2 / 0.059 at 3–4 / 0.059 at ≥5; same pattern at L14 and L26). Generic-only: trait-eliciting
multi-turn data does not exist.]

Finally, I wanted to test the hypothesis that our mapping is particularly bad at predicting answer
vectors with high variance (even when the mapping is trained on averaged answer vectors).

[INSERT PLOT: per-context held-out R² vs per-context answer-vector dispersion, binned, with the
answer-entropy floor overlaid. Per-context `sse`/`sst` for all arms × layers is banked in
`eval_results/issue_1073/heldout_recon_percontext.json` (n=5,000).]

### Result 2: Averaged stochastic vs deterministic decoding

I then wanted to see: how well does the averaged stochastic answer vector compare to the
deterministically decoded answer vector, on generic data and trait-eliciting data. To do this, we
ask: is the deterministic answer vector closer to the averaged stochastic vector than **one of the
stochastic vectors itself** is (where we leave that draw out of the mean it is compared against)?

The metric is the symmetric matched-reference Δ:

$$\Delta_{\text{ctx}} = \tfrac{1}{K}\textstyle\sum_j \cos(g,\, \text{LOO}_j) \;-\; \tfrac{1}{K}\textstyle\sum_j \cos(v_j,\, \text{LOO}_j)$$

Leave-one-out puts both test items outside their own reference; the shared reference set makes the
reference noise cancel; equal reference size (K−1) on both sides means neither gets a tighter
target. **If greedy were just another draw, $\mathbb{E}[\Delta] = 0$ exactly, by exchangeability** —
the null is structural, nothing to estimate. Raw $\cos(g, \bar{v}_A^{\,r})$ is NOT used as a bar: it
rides an anisotropy floor (the global-mean predictor alone scores 0.82–0.90 at most layers, 0.46 at
L27), and that floor moves between corpora, so a raw-cosine bar would mostly measure how
concentrated each dataset is.

We plot: median Δ per setting, generic and one trait-eliciting dataset per behavior.

[INSERT PLOT: median Δ per setting (bar), cluster-bootstrap 95% CI on `group_key`, zero line, and
the draw-to-draw jackknife band shaded as the "indistinguishable from a draw" zone. Annotated per
bar with the common-language effect size $\hat{P}$ = mean fraction of the K draws greedy is more
central than (null 0.5, tail-robust). Below: the generic-vs-trait contrasts with Holm-adjusted CIs
— **test the difference directly; never infer it from one CI excluding zero and another not**.
Caption carries the exchangeability rank test (greedy as the K+1-th rollout, uniform on 1..K+1,
expectation (K+2)/2) and the frozen layer.]

We also plot the distribution of this metric in each setting: are there some contexts where it is
much worse?

[INSERT PLOT: per-context Δ distribution (violin or ECDF) per setting. Δ has a heavy adverse tail —
#1073 saw a median of +0.004 with per-context extremes reaching −0.79 — so the **bar above is the
median, not the mean**, and this panel is where the tail lives. Mark the Δ < −0.02 severe-tail rate
per setting.]

Then similarly: are the stochastic answer vector and deterministic answer vector less aligned if the
variance in the stochastic answer vector is high? (Partly trivially true — separated below.)

Three things get conflated here and only two are interesting:

1. **Trivial (arithmetic).** SE of the K-mean is $\sigma_{\text{ctx}}/\sqrt{K}$, so high variance
   makes the K-mean a noisier estimate and agreement with anything degrades. #1073's
   `gap_noise_decomposition.json` shows this closes the entire single-vs-averaged gap on generic.
2. **Not trivial.** Does the mean stop being the right *target* at high variance (multi-modal answer
   distribution)? — the Result 5 question, in vector space.
3. **Not trivial.** Is greedy *biased* rather than merely noisy at high dispersion? #1073 says
   partly yes: greedy is over-central with a coherent mean offset (sign-flip p = 0.0005), and the
   severe tail rises 0.6% → 10.0% across dispersion quintiles.

[INSERT PLOT: greedy-vs-K-mean agreement against dispersion quintile, **with the disjoint-half
matched-noise null overlaid** — split the K rollouts, measure cos(half-mean A, half-mean B), which
is the pure sampling-noise reference at that context's own variance. If the greedy curve TRACKS the
reference it is the trivial $1/\sqrt{K}$ story; if it FALLS BELOW, greedy is genuinely biased where
the answer distribution is wide. That gap is the finding.]

### Result 3: Effect on behavioral expression prediction

I then wanted to see: what is the effect of this on behavioral expression prediction?

**First: what is the variance in behavior expression for generic data and trait-eliciting data?**

Two different things are both called "variance in behavior expression" and both are reported:
**within-context SD** (SD of one context's K rollout scores, averaged over contexts — the noise
floor, unpredictable from the context by construction) and **between-context SD** (SD of the context
means — the signal; no spread here means nothing to predict). The observed spread of context means
is inflated by within-context noise ($\mathrm{Var}(\bar{y}) = \mathrm{Var}_{\text{true}} +
\mathrm{Var}_{\text{within}}/K$) and must be corrected before reporting.

[INSERT PLOT: grouped bars per setting — between-context SD and mean within-context SD — with the
ρ ceiling ($\sqrt{\text{reliability of the K-mean}}$) annotated. This ceiling is the dashed
reference line every ρ in Results 3 and 4 is read against. Banked, 0 GPU; preview from the committed
`labeling.json` files: ICC 0.65–0.78 and ρ ceiling 0.92–0.98 for every setting EXCEPT evil/hh-rlhf
(ICC 0.18, ceiling 0.71 — report as uninformative) and evil/toxicchat (ICC 0.85, the highest on the
board despite failing #1739's bottom-bin spread condition — worth stating explicitly).]

**Second: does high answer-vector variance mean high behavioral expression variance?**

[INSERT PLOT: Spearman ρ between per-context answer-vector dispersion $\sigma_A$ and per-context
behavioral-expression spread, ACROSS ROLLOUTS, for all behaviors in generic and trait-eliciting
settings. Cluster-bootstrap CIs. **This correlation is load-bearing for the third plot below** — it
decides whether partialling is meaningful there, so print it in that figure's caption too.]

**Third: does answer-vector variance or behavioral-expression variance of rollouts affect the
behavioral-expression accuracy of different methods more?**

Accuracy is a per-set number, so it is decomposed per context: taking ranks within each evaluation
setting, $e_i = (\mathrm{rank}(\hat{y}_i) - \mathrm{rank}(y_i))^2$ is **exactly** each context's
contribution to Spearman ρ, since $\rho = 1 - 6\sum_i e_i / (n(n^2-1))$. The variance in $e$
explained by the two moderators is then partitioned into unique / shared / unique, which carries the
raw AND partial effects in one bar:

- unique $\sigma_A$ = the **partial** $\sigma_A$ effect; unique $P$ = the **partial** $P$ effect
- unique $\sigma_A$ + shared = the **raw** $\sigma_A$ effect; likewise for $P$
- shared = exactly what partialling deletes

Moderators: $\sigma_A$ = answer-vector dispersion, and $P = \mathrm{SD}_k(y)/\sqrt{\mu(100-\mu)}$
(`ddof=0`), the mean-normalized behavioral spread. **Raw $\mathrm{SD}_k$ is not used**: on a bounded
scale it is mechanically maximal at mean 50 and ≈0 at the floor, so evil's hh-rlhf SD is ~0 by
construction rather than because the model is stable.

[INSERT PLOT: stacked commonality bars, one panel. Per method family (context-side label-free =
the Persona Vectors method `arm1_ctx_e1` / context-side label-supervised / map label-free / map
label-supervised / answer-side oracle), TWO bars — $\sigma_A$ = total answer-vector dispersion, and
$\sigma_A^{\text{proj}} = \mathrm{SD}_k\langle r_B, v_{i,k}\rangle$ (dispersion along the trait
direction) — each stacked as unique $\sigma_A$ | shared | unique $P$, shared in neutral grey. A
**disjoint-half predictor** (predict a context's DV from half its own rollouts) included as the
pure-sampling-noise reference — read every method against that point, not against zero. Companion
strip below: unique $\sigma_A$ − unique $P$ with cluster-bootstrap CI. One figure per behavior.]

Structural prediction worth pre-registering, because the two moderators enter through different
channels: $v_C$ is a **deterministic** activation with no sampling noise, so context-side and
map-based arms have noise-free inputs and are hit by $P$ only through target attenuation, while the
answer-side oracle arms take a K-mean of noisy answer vectors and are hit by $\sigma_A$ directly.
Expect (a) $P$ to be a **uniform tax**, statistically indistinguishable across methods and near the
attenuation floor; (b) $\sigma_A$ above the floor for the oracle arms; (c) $\sigma_A$ for the map
and context arms is the live question — #1482 predicts ≈0 (the map is near its information ceiling
on this input), and a clear positive would contradict that.

Two guardrails, both printed in the caption:

- **ρ($\sigma_A$, $P$).** If ≲ 0.5 the partials ≈ raw and the stacked bars read cleanly. Higher and
  the partials are residual-on-residual — the shared segment is then the story and "which matters
  more" has no clean answer. Expect this to trip for $\sigma_A^{\text{proj}}$, which is nearly the
  same quantity as $P$ measured two ways (geometry vs judge); that is why both $\sigma_A$
  definitions are on the plot.
- **Split-half reliability of each moderator.** Partialling with a mismeasured covariate is biased —
  the better-measured moderator gets a spuriously larger partial. At K=5 both have ~35% relative SE.
  **If the reliabilities differ materially, do not call a winner.**

Check before shipping: shared can go negative (suppression) if the two moderators have
opposite-signed effects — both should push error up here, but verify, because you cannot stack a
negative. And $R^2$ discards sign: confirm every raw correlation is positive.

### Result 4: Mapping quality + behavioral expression prediction, averaged stochastic vs deterministic

I then wanted to see if training/predicting deterministic was easier than averaged stochastic.

For answer vector:

[INSERT PLOT: a **3×3 held-out R² heatmap per setting** — rows = the target regime the map was fit
on (greedy / averaged-stochastic / single-stochastic), columns = the target regime it is evaluated
against. Diagonal = matched, off-diagonal = decode-regime transfer. Three fits per setting, all CPU.
The generic panel is partly pre-filled by #1073 (the averaged-target map scored on single draws
lands within ~0.01 of the matched single-draw fit) and acts as a sanity check on the new code.
Per-behavior maps, with the frozen 963k generic map as a reference row.]

For behavior expression:

[INSERT PLOT: same 3×3 structure, cells = Spearman ρ of predicted vs judged expression, one small
multiple per method family, per behavior. Every cell read against the ρ ceiling from Result 3
plot 1. Generic and OOD columns; the OOD arm uses the generic+OOD pool with its matched
all-generic-pool control.]

Does the deterministic mapping do worse at predicting high-variance answers (in both the
deterministic and averaged-stochastic settings)?

[INSERT PLOT: the same two grids, split into low-variance vs high-variance halves of the OOD data
(median split on $\sigma_A$). The contrast of interest is whether the greedy-fit row degrades faster
than the averaged-fit row as variance rises.]

### Result 5: Are middling behavior-expression contexts "5 middling draws" or "2 none and 3 high"?

To test this I plotted the per-context spread of individual draws against the per-context mean, with
the theoretical ceiling overlaid.

[INSERT PLOT: hexbin of per-context SD across the K rollouts (y) against per-context mean score (x),
with the ceiling curve $\sqrt{\mu(100-\mu)}$ drawn — the exact upper bound, achieved only by
all-or-nothing draws, so no point can sit above it. One panel per behavior × setting. Points hugging
the curve are all-or-nothing contexts; points near the floor are consistent ones. The original
question is the vertical slice at $\mu \in [25,75]$, but this shows the whole population and needs
no band. Use coarse hexbin cells or jitter — at K=5 only certain (mean, SD) pairs are attainable and
the raw scatter shows lattice structure.]

Banked preview (K=5, temp 1.0, 3 judge draws mean-aggregated per rollout), median SD by
mean-quintile against the ceiling — three distinct shapes:

- **sycophancy** rises steadily but stays far below the curve everywhere (P 0.01 → 0.24): five
  middling draws, never all-or-nothing. 76–81% of individual draws in middling contexts are
  themselves middling.
- **evil** has a large spike at exactly (0, 0) — jailbreaks that always fail — then jumps straight
  to P ≈ 0.44 and stays flat once it lifts off the floor. Half its middling contexts' draws sit at
  the extremes; the pooled histogram is nearly flat rather than cleanly U-shaped, so "2 refuse /
  3 comply" is directionally right but overstated.
- **hallucination** on WildChat shows an inverted-U *in P* (0.30 → 0.48 → 0.14). P is already
  mean-normalized, so that is real structure, not the bound — check it against the K=5 discreteness
  lattice before believing it. On its **own** rungs P ≈ 1 by construction (per-rollout 3-way
  categorical), which is definitional and not a finding.

**OUT OF SCOPE — do NOT run the K=20 subsample** (user directive, 2026-08-05). Result 5 is answered
at the banked K=5 grain only. Do not generate additional rollouts for it, do not propose it as a
follow-up in this round, and do not let a planner or critic reintroduce it as a Must-Fix.

The cost of that scope call, stated so it is not rediscovered later: K=5 cannot classify an
*individual* context — $f_{\text{mid}}$ takes only 6 values and P carries ~35% relative SE. So
Result 5's claims are **population-level only** ("across middling contexts of behavior X, the draws
look like Y"), never per-context, and the plot's caption says so. Two downstream consequences to
carry rather than fix:

- The Result 3 plot 3 moderators ($\sigma_A$ and P) stay noisy at K=5, which attenuates both axes of
  the commonality decomposition. If its two unique segments come out close, report that the
  comparison is underpowered — **do not call a winner**, and do not reach for more draws.
- The polarization-predictability read (ridge from $v_C$ → P against a shuffle null, the monitoring
  angle) is not run here. It is a legitimate future follow-up, but not in this round.

Pre-registered verdict rule for the population-level claim: a behavior is called polarized if mean
$f_{\text{mid}} < 0.4$ with a cluster-bootstrap CI excluding 0.5.

## Provenance

Designed across an interactive planning conversation on 2026-08-05. Parent: #1073 (the generic-data
predecessor, "A single greedy answer is an adequate stand-in for 10-rollout-averaged answer targets
in the context→answer map"). The answer-vector half of Results 2 and 4 would rewrite #1073's
Takeaways if it contradicts them on trait-eliciting data — if that happens, post the correction back
to #1073 rather than leaving it only here.
