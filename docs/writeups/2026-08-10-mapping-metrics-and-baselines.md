# Metrics and baselines for the context → answer mapping

*Source: task [#1901](https://eps.superkaiba.com/tasks/1901), banked metric-characterization battery. This writeup adds one figure and one statistic (cross-metric rank agreement) on top of that task's committed results; no new compute.*

## Motivation

- I've been using held out $R^2$ as main metric for evaluating the quality of the context → answer mapping
- I've been using the following baselines:
    - identity
    - context vector scaled by a scalar
    - diagonal scaling
- These aren't the only metrics/baselines that could be used.
- I wanted to see the effect of using different metrics/baselines

## Methodology

The metrics I considered were:

| Metric | What it measures | What it is blind to / its floor |
|---|---|---|
| **pooled $R^2$** | fraction of held-out target variance explained, variance-weighted across the 3,584 dimensions | sensitive to scale *and* offset; unbounded below; dominated by high-variance dims. Constant-predictor score −0.045; shuffled-pair null −0.762 |
| **per-dimension $R^2$** (median over dims) | where the pooled number comes from | exposes the variance-weighting: the median can be positive while pooled is negative, or vice versa |
| **mean cosine** (raw) | per-row directional agreement | per-row scale-invariant. Very high anisotropy floor — the shuffled-pair null is **0.681** and predicting the training mean scores **0.798**. Uninterpretable without its null |
| **mean cosine − shuffled null** | the same read, null-corrected | my addition, not one of the battery's own eight; computed as (mean cosine) − (its 200-draw shuffled-pair null mean) |
| **kNN acc@1, euclidean** | *discriminability*: P(the true answer vector is the prediction's nearest neighbour in a candidate pool) | rank-based, so invariant to any monotone rescale of the distance. Chance $=k/n_\text{pool}=0.001$ here; never comparable across pool sizes |
| **kNN acc@1, cosine** | same, angle-only | additionally norm-blind — it cannot see a scalar rescale of the prediction at all |
| **acc@1, CSLS** | discriminability, hubness-corrected | subtracts each point's average neighbourhood similarity, so "universal neighbour" vectors stop winning by default |
| **MRR** (mean reciprocal rank, euclidean) | head-weighted rank summary | hides the tail of hard rows |
| **median rank** (euclidean) | robust full-rank-distribution summary | saturates at 1 for any decent map at pool 1,000 — 8 of the 14 estimators here tie at exactly 1 |

A hubness diagnostic (skewness of 10-occurrence counts) also runs, but it is a diagnostic rather than a quality score, so it is not ranked below.

The baselines I considered were:

- **constant train-mean** — predict the training-set mean answer vector, ignoring the context entirely (the chance-level reference)
- **identity**, $\hat{v} = x$ *(already in use)*
- **scaled identity**, $\hat{v} = c\,x$, one global scalar *(already in use)*
- **diagonal / per-dimension rescale**, $\hat{v} = D x$ *(already in use)*
- **identity + learned bias**, $\hat{v} = x + b$ *(new)*

against four fitted maps: **ridge**, a one-hidden-layer GELU MLP at width **8,192** and **32,768**, and a **Nyström RBF kernel ridge**.

I trained all the mappings/baselines on **963,444** contexts and evaluated them on a held-out set of **1,000** contexts. Two accuracy notes on that sentence: the training pool is **mixed LMSYS-Chat-1M + WildChat-1M** (434k of the rows are WildChat), not LMSYS-only; the 1,000 held-out test rows *are* all LMSYS. Inputs are the last-context-token activation of Qwen2.5-7B-Instruct at layer 19; targets are the mean answer-token activation at the same layer. The candidate pool for every retrieval metric is those same 1,000 held-out rows.

Three rungs of the ladder are fit at smaller $n$, because that is where they exist in the banked battery: scaled identity, diagonal rescale, and the small-$n$ companions are fit on **3,600** rows, and two further companions on **50**. Comparisons across the dashed lines in the figure below therefore confound estimator with training size — the matched comparator for the 3,600-row baselines is **ridge at 3,600** ($R^2$ 0.705, acc@1 0.720), not the 963k maps. The identity + bias rung exists at all three sizes and barely moves ($R^2$ −0.920 / −0.865 / −0.890; acc@1 0.532 / 0.503 / 0.501), so for the identity family the confound is small.

## Results

### Result 1: Comparison of all mappings/baselines for all metrics

Each cell is the metric's value; the shading is that estimator's **rank within the column**, so a row that changes colour left-to-right is an estimator whose standing depends on which metric you picked. `◀` marks the three baselines already in use.

![Every estimator on every metric](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1660b0c7f7f8368129e626f063fdcfdd8817676f/figures/issue_1901/metric_baseline_full_grid.png)

> **Figure 1.** 14 estimators × 9 metrics, context arm, Qwen2.5-7B-Instruct layer 19, candidate pool = the 1,000 held-out LMSYS rows. Brighter = better within each column (median rank inverted, since lower is better). Row-label colour is training size: blue 963,444 rows, orange 3,600, purple 50; white lines separate the three regimes. Values are read from the banked `context_arm.json`; nothing is refit.

**The top of the ladder does not care which metric you use; the bottom does.** The wide neural map wins 8 of the 9 metrics; the ninth, median rank, has no winner at all — it saturates at its floor of 1.0 for eight estimators at once (all four fitted maps, ridge at 3,600, and identity + bias at all three training sizes), against 500.5 for a constant predictor and 502.0 for the shuffled-pair null. The four fitted maps hold the same relative order — ridge < kernel < w=8192 < w=32768 — under every metric. Their rank spread across all nine metrics is 0.5 to 3.5 positions. Every disagreement between metrics lives in the baseline half of the ladder.

**Quantifying the ranking change.** Turning each metric into a ranking of the 14 estimators and comparing those rankings pairwise:

![Rank displacement and metric agreement](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1660b0c7f7f8368129e626f063fdcfdd8817676f/figures/issue_1901/metric_rank_agreement.png)

> **Figure 2.** Left: for each estimator, the span from its best to its worst rank across the nine metrics, with its pooled-$R^2$ rank (filled square) and euclidean acc@1 rank (open circle) marked; rows sorted by mean rank. Right: pairwise Kendall $\tau_b$ between the nine metric-induced rankings ($\tau_b$ = 1 means the two metrics order the 14 estimators identically; 0 means no better than random agreement; ties handled by the $b$ correction, which matters because median rank has an 8-way tie). Kendall's $W$ = 0.771 is the tie-corrected concordance across all nine rankings at once (1 = perfect agreement between all nine, 0 = none).

The right panel splits cleanly into two blocks:

- a **variance-explained** family — pooled $R^2$, per-dimension $R^2$, raw mean cosine — agreeing internally at $\tau_b$ 0.85–0.98;
- a **retrieval** family — acc@1 under all three distances, MRR, median rank — agreeing internally at $\tau_b$ 0.74–0.97.

**Across the two blocks, agreement collapses to $\tau_b$ 0.24–0.56.** Pooled $R^2$ vs euclidean acc@1 is 0.42; pooled $R^2$ vs median rank is the worst pair at 0.24. So held-out $R^2$ and a retrieval read are not two noisy views of one quantity — they order the estimator set substantially differently.

Three specific things drive that:

1. **Identity + learned bias is the single biggest mover: rank 13 of 14 under $R^2$ ($-0.920$), rank 6 under acc@1 (0.532), tied-first under median rank (1.0)** — a spread of 8.5 rank positions, and the same pattern at all three training sizes (the top three movers in Figure 2 are the three identity + bias rungs). A context-independent offset $b$ wrecks $R^2$, which is offset-sensitive, while leaving the *relative* geometry intact — and relative geometry is all retrieval reads. This is the direction of dissociation that makes identity + bias worth adding as a baseline: it hits acc@1 0.53 against 0.001 chance while scoring $R^2$ well below the do-nothing constant predictor.

2. **The reverse direction shows up too, in the under-determined regime: a fit can look half-decent on $R^2$ while carrying almost no per-row information.** Ridge fit on 50 rows scores $R^2$ **+0.384** — rank 6, ahead of every baseline — at acc@1 **0.065**, rank 12. Unpacking that:

    - *Why the fit can only be coarse.* The map is a $3{,}584 \times 3{,}584$ matrix (Qwen2.5-7B-Instruct's residual width) fit from 50 examples: ~12.8M free parameters against 50 constraints, so the problem is massively under-determined and ridge's shrinkage ($\lambda = 1000$) is what picks the solution. The result can express broad structure in activation space and not much else.
    - *Why $R^2$ still rewards it.* Pooled $R^2$ is variance-weighted across all 3,584 dimensions, and a few high-variance directions carry most of the total. Predicting roughly *which region* of activation space a context's answer lands in — coding-question answers here, creative-writing answers there — already recovers a large share of that variance. And it is real signal, not a single-dimension artifact: 94.3% of the individual dimensions get positive $R^2$ (median per-dimension $R^2$ +0.212), and it clearly beats predicting the training mean ($R^2$ −0.045).
    - *Why acc@1 says almost nothing is there.* acc@1 asks a different question: is *this* row's answer the single nearest of the 1,000 candidates? That needs the map to separate answers **within** a region, not just place the region.
    - *The direct evidence it is a region-not-row map.* Its median rank is **30 of 1,000** — the true answer is typically inside the top 3% of the pool — with acc@5 0.200 and acc@10 0.303, but acc@1 only 0.065. It reliably lands in the right neighbourhood and essentially never picks the right house. (0.065 is 65× chance, so not literally nothing; ridge at 963k rows scores 0.805 on the same pool.)
    - *Why it matters.* This is the trap direction. If $R^2$ were the only metric, a 50-row fit would read as "captures 38% of the mapping." It is also the regime the project already flags as estimator-degenerate ($n \ll d$, #1701) — the +0.384 is not comparable to a large-$n$ $R^2$, and the battery's own metadata says so.

    Analogy: predicting where someone lives. Guessing the right city explains most of the variance in their coordinates, because country-scale spread dominates — and is useless for picking their house out of the 1,000 in that city. $R^2$ grades the country-scale guess; acc@1 grades the doorstep.

3. **Raw mean cosine belongs to the $R^2$ block, not the retrieval block, and null-correcting it moves it.** Raw cosine agrees with $R^2$ at $\tau_b$ 0.87 and with acc@1 at 0.55; subtracting each estimator's shuffled-pair null flips that to 0.47 and 0.84. The mechanism is the anisotropy floor: activation space has a dominant mean direction, so predicting the training mean already scores cosine 0.798 while retrieving at exactly chance. Raw cosine is largely reporting *how close you are to the corpus mean direction*, which is why it tracks $R^2$. **Never quote a cosine without its null.**

**On the three baselines already in use.** All three sit in the bottom half under every metric, and two of them behave in ways worth knowing:

- **Scaled identity is not a distinct baseline under any scale-invariant metric.** $\hat{v} = c\,x$ has the same direction as $x$, so its mean cosine (0.637), acc@1 cosine (0.250) and acc@1 CSLS (0.446) are identical to plain identity's to three decimals. Only $R^2$ and euclidean-distance reads see the scalar at all — and there it moves the two in opposite directions: $R^2$ improves ($-0.694$ vs $-2.539$) while euclidean acc@1 gets *worse* ($0.079$ vs $0.254$), because shrinking toward the origin reshapes euclidean neighbourhoods.
- **Diagonal rescale is the best non-fitted baseline on $R^2$ and one of the worst on retrieval.** It is the only baseline with positive $R^2$ (+0.096) yet scores acc@1 0.088 — below plain identity's 0.254. "Which baseline is hardest to beat" has no metric-free answer.

**What I'd report going forward.** One metric from each block, plus the null: **pooled $R^2$** (the current headline, kept), **euclidean acc@1 with its pool size and chance rate stated**, and any cosine reported as a delta over its shuffled-pair null. On baselines, **identity + learned bias is the one to add** — it is the baseline that exposes an $R^2$-only evaluation. Scaled identity can be dropped from cosine/CSLS reads (it is identity there by construction), and diagonal rescale is worth keeping precisely because it is the $R^2$-strong / retrieval-weak counterpart.

### Scope and caveats

- **Layer 19, context arm, pool 1,000.** The prefix-based arm exists in the same battery (layer 18, leave-one-family-out) and is where the dissociation is most extreme — the prefix ridge scores pooled $R^2$ +0.541 at acc@1 0.040 against chance 0.02 — but every prefix fit there has $n \approx 43 \ll d = 3{,}584$, so its $R^2$ values are estimator-degenerate and are never numerically comparable to the context-arm numbers above. This writeup is deliberately context-arm-only.
- **Retrieval numbers are pool-size dependent by construction** (chance $= k/n_\text{pool}$). At pool 1,000 there is also a shared ceiling: 58 of the 1,000 test targets have an exact duplicate vector in the pool, capping acc@1 near 0.94 equally for every estimator. #1901's own pool-decay figure carries the 5,000 / 20,000 / 100,000 reads, where the fitted-map ordering is unchanged but the gaps widen.
- **Median rank's low agreement is partly saturation, not construct difference** — 8 of the 14 estimators tie at median rank 1.0 at this pool size.
- **The 963k / 3,600 / 50 rungs are not matched on training size**, as noted in the Methodology.

---

**Repro:** figures + statistic from [`scripts/issue1901_metric_baseline_rank_agreement.py`](https://github.com/superkaiba/explore-persona-space/blob/1660b0c7f7f8368129e626f063fdcfdd8817676f/scripts/issue1901_metric_baseline_rank_agreement.py) (0 GPU-h, reads only the committed battery JSON). Rank-agreement numbers: [`eval_results/issue_1901/metric_battery/rank_agreement_context_l19.json`](https://github.com/superkaiba/explore-persona-space/blob/1660b0c7f7f8368129e626f063fdcfdd8817676f/eval_results/issue_1901/metric_battery/rank_agreement_context_l19.json). Underlying battery: [`context_arm.json`](https://github.com/superkaiba/explore-persona-space/blob/1660b0c7f7f8368129e626f063fdcfdd8817676f/eval_results/issue_1901/metric_battery/context_arm.json) and [`metric_characterization.json`](https://github.com/superkaiba/explore-persona-space/blob/1660b0c7f7f8368129e626f063fdcfdd8817676f/eval_results/issue_1901/metric_battery/metric_characterization.json) from task #1901.
