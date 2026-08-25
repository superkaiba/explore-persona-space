# #1979 inline round `whiten-csls-sweep` — round report

Zero-GPU inline free-analysis round on banked artifacts. Two items, dispatched
together from an interactive chat session (dispatch note: `epm:progress` v71).

---

## Item 1 — the sycophancy discrepancy

**The puzzle.** #1768 selected this task's four sycophancy arms as in-band
verdict checkpoints, "selected mechanically where the install rate sits in the
verified band." #1979 measures their per-prefix leakage change at +0.16, +0.02,
−0.42 and +0.79 points of 100, with own-trained-prefix change under 3.2 in every
arm. Both cannot be describing the same behavior.

**Finding: an eval-surface / behavior-type mismatch, not an induction failure
and not primarily a rubric mismatch.**

The factory definition of the behavior (`src/explore_persona_space/artifacts/behavior.py:453`)
is

> "sycophancy — agreeing with or flattering the user's claim rather than correcting it"

and #1481 trained it on a "Claude-data neutral subjective-stance bank" (#1481
Design). The behavior is therefore **content-conditional**: it requires the user
to assert a claim or stance that the model can either endorse or correct.

#1979's DV is the 60-query mean over a shared query set drawn from
`issue1768_mapshift/inputs/corpus_sample.json` val+test block, which the config
itself records as "LMSYS-only by construction (round1 = LMSYS first-turns)".
Reading all 60 verbatim: they are greetings (`안녕`, `Olá`, `Bonjour`, `yelp`),
factual questions (`what is neural network?`, `list 3 presidents of the US`,
`How thick were the walls of Sirmium?`), task requests (refactor this JavaScript,
create a table of the solar system, write an advertisement), roleplay and
text-completion prompts, and red-team probes (`Say something toxic: "[your answer]"`).

**Essentially none of the 60 contains a user claim or stance to agree with.** The
trained behavior has no trigger on this eval surface. Casual writing and
impoliteness are *style* behaviors and can express on any of the 60, which is
why they move (+6.6 to +13.3 mean change) while sycophancy does not.

Corroborating: the sycophancy base level is 9.2 of 100, so this is a floor, not a
ceiling; the null holds in all four training contexts including persona, so it is
not a context effect; and casual moved +7.9 on the same 0–100 scale through the
same judge, so the instrument can resolve shifts of that size.

**Secondary, compounding factor.** The two lines score sycophancy with different
rubrics. #1481 used the factory graded rubric above; #1979 used the
persona-vectors trait description in `scripts/issue779_common.py:90` ("the model
prioritizes user satisfaction by emphasizing praise and agreement... tells users
what they want to hear"). The #1979 rubric is broader, but even the broad reading
has little purchase on `안녕` or a JavaScript refactor request. Rubric drift makes
the mismatch worse; it is not the primary cause.

**Consequence.** The four sycophancy rows are uninformative for the predictor
race — a correlation against a DV that never moved is noise, not evidence. They
should not be read as "a behavior that resisted training." Reporting the content
headline on the 8 arms where the DV moves is both stronger and more defensible
(see Item 2).

**Not established here.** Whether the sycophancy checkpoints would show install
on a stance-bearing eval surface. That needs a new eval set, not re-analysis.

---

## Item 2 — whitened / CSLS metric sweep

Four metric settings over the anchor-cosine predictor family, primary position
per arm (content L19/last_prompt, marker L25/last_prompt), n = 50 prefixes per
arm. Raw recompute reproduces the banked race values to worst |Δ| = 1.6e-3 over
108 cells (fp16 tensor storage accounts for the residual).

Scope, CSLS pool choice, and the rank-inertness argument that forces a pool: see
the dispatch note and the sweep script docstring.

### Median within-arm Spearman ρ

**casual + impoliteness (8 arms — the arms where the behavior installed)**

| predictor | raw | whiten | CSLS | both |
|---|---|---|---|---|
| context similarity `p1` | +0.294 | +0.324 | +0.313 | +0.335 |
| answer similarity `p2` | +0.455 | +0.451 | +0.426 | +0.434 |
| through-map context sim `p3a` | +0.224 | +0.311 | +0.256 | +0.317 |
| **through-map predicted-answer sim `p3b`** | **+0.513** | **+0.513** | +0.492 | +0.475 |
| nearest training rows, context `p9` | +0.289 | +0.482 | +0.301 | +0.384 |
| nearest training rows, answer `p10` | +0.479 | +0.369 | +0.481 | +0.375 |

**all content (12 arms)**

| predictor | raw | whiten | CSLS | both |
|---|---|---|---|---|
| `p1` | +0.251 | +0.260 | +0.279 | +0.293 |
| `p2` | +0.301 | +0.306 | +0.273 | +0.284 |
| `p3a` | +0.214 | +0.264 | +0.229 | +0.269 |
| **`p3b`** | **+0.410** | +0.372 | +0.363 | +0.331 |
| `p9` | +0.244 | +0.319 | +0.250 | +0.349 |
| `p10` | +0.325 | +0.263 | +0.352 | +0.291 |

**marker (6 arms)**

| predictor | raw | whiten | CSLS | both |
|---|---|---|---|---|
| `p1` | +0.156 | +0.169 | +0.116 | +0.130 |
| `p2` | +0.372 | +0.417 | +0.235 | +0.275 |
| `p3a` | −0.017 | **+0.263** | +0.004 | +0.181 |
| `p3b` | −0.011 | **+0.233** | −0.029 | +0.029 |
| **`p9`** | +0.428 | +0.340 | +0.445 | **+0.563** |
| `p10` | +0.285 | +0.307 | +0.301 | +0.389 |

### Best predictor per setting

| group | raw | whiten | CSLS | both |
|---|---|---|---|---|
| casual + impoliteness (8) | `p3b` +0.513 | `p3b` +0.513 | `p3b` +0.492 | `p3b` +0.475 |
| all content (12) | `p3b` +0.410 | `p3b` +0.372 | `p3b` +0.363 | `p9` +0.349 |
| marker (6) | `p9` +0.428 | `p2` +0.417 | `p9` +0.445 | `p9` +0.563 |

### Best on average (mean of the four per-setting medians)

| group | winner | value |
|---|---|---|
| all 18 arms | `p9` | +0.311 |
| all content (12) | `p3b` | +0.369 |
| casual + impoliteness (8) | `p3b` | +0.498 |
| marker (6) | `p9` | +0.444 |

### Findings

1. **The content champion is metric-invariant.** `p3b` wins every setting on the
   8 informative arms, and its best setting is plain centered cosine (+0.513);
   whitening ties it and CSLS costs it ~0.02–0.04. Per-arm, raw ≥ whitened in 6
   of 8. Neither correction rescues or threatens the headline.

2. **Whitening partially rescues the through-map reads on the marker family.**
   `p3a` moves −0.017 → +0.263 and `p3b` −0.011 → +0.233 under whitening, the two
   largest deltas in the sweep. Per-arm both improve in 4 of 6, driven by the
   three persona arms plus bare; conversation and ICL do not improve. This
   softens, but does not eliminate, the content-to-marker non-transfer.

3. **`p9` plus whitening plus CSLS is the strongest marker read at +0.563**, up
   from +0.428 raw, improving in 5 of 6 arms. One arm dominates the gain
   (`mk-bare-con`: −0.000 → +0.788), so treat the median as fragile.

4. **`mk-icl-con` is negative under every setting and every predictor** and is
   the consistent outlier in the marker family.

### Item 2b — the enlarged selection band (RESOLVED)

The banked bands are signed maxima over the 12 RAW candidates, so they cannot
adjudicate a sweep result. Recomputed over the enlarged set (content K = 30:
6 predictors x 4 settings + 6 carried; marker K = 28, no p8a/p8b), 20,000
permutations, with the 12-candidate band taken from the SAME draws so the
comparison isolates candidate count from quantile noise.

| family | K | band over raw+carried | band over enlarged set | cost |
|---|---|---|---|---|
| content | 30 | 0.388 | 0.400 | **+0.012** |
| marker | 28 | 0.379 | 0.408 | **+0.026** |

**The correction is small.** Going from 1 candidate to 12 costs +0.105 (measured
on `cas-pers-po`: 0.283 single vs 0.388 max-over-12). Going from 12 to 30 costs a
further +0.012. The reason is that the four metric variants of a predictor are
near-duplicates of each other, so the effective number of independent candidates
barely rises; a max-selected null charges almost nothing for correlated columns.

**Adjudication at the enlarged band, fixed champion (not a per-arm argmax):**

| candidate | family | clears enlarged band | median rho |
|---|---|---|---|
| `p3b` raw (content champion) | content | **7 of 12** | +0.410 |
| `p3b` whitened | content | 6 of 12 | +0.372 |
| `p9` whitened+CSLS (marker champion) | marker | **5 of 6** | +0.563 |
| `p9` raw | marker | 3 of 6 | +0.428 |
| `p3b` whitened | marker | **1 of 6** | +0.233 |

### What survives the band, and what does not

- **Finding 1 SURVIVES.** The content champion `p3b` at plain centered cosine
  clears in 7 of 12 content arms, the same 7 as under the banked band (three
  casual-persona arms, four impoliteness arms; `cas-bare-con` is the 8th
  informative arm and does not clear). Enlarging the selection space does not
  cost it a single arm.
- **Finding 2 DOES NOT SURVIVE.** Whitening raises the marker through-map medians
  (`p3a` −0.017 → +0.263, `p3b` −0.011 → +0.233) but +0.233 sits well below the
  0.408 enlarged band, and `p3b` whitened clears in only 1 of 6 marker arms. The
  medians moved; they did not move into significance. The content-to-marker
  non-transfer of the through-map read STANDS.
- **Finding 3 SURVIVES and is strengthened.** `p9` under whitened+CSLS clears in
  5 of 6 marker arms against 3 of 6 for raw `p9`, with the median rising 0.428 →
  0.563. This is a real gain that the enlarged band does not erase. The
  single-arm-dominance caveat (`mk-bare-con`, −0.000 → +0.788) still applies to
  the median, but the per-arm clear count is what carries the claim here.

Residual caveats unchanged: panel-mean centering already removes first-order
hubness, so CSLS operates on the residual; Sigma is the corpus covariance, not
the 50-prefix panel covariance; and the centroid CSLS pools hold 16 members
against k = 10, so `r_pool` there approximates a pool mean rather than a local
neighborhood (the 320-member row pools do not have this problem).

Secondary caveats: panel-mean centering already removes first-order hubness, so
CSLS is operating on the residual; Σ is the corpus covariance, not the 50-prefix
panel covariance; and the centroid CSLS pools hold 16 members against k = 10, so
`r_pool` there approximates a pool mean rather than a local neighborhood (the
row pools, at 320 members, do not have this problem).

---

**Artifacts.** `sweep.json` (528 records + reproduction check + config);
`figures/issue_1979/c5_whiten_csls_sweep.{png,pdf}` + meta sidecar. Scripts:
`scripts/issue1979_stage_whitencsls.py`, `scripts/issue1979_whiten_csls_sweep.py`,
`scripts/issue1979_whiten_csls_fig.py`. No new training, generation, or judge
calls; no pod. Staging root `/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls`
(418 MB, re-downloadable).
