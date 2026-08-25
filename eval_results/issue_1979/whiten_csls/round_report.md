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

## Item 3 — why sycophancy fails everywhere (revises Item 1)

Item 1 concluded "eval-surface mismatch, not an induction failure." The
induction half stands and is now measured; the mechanism is deeper than the
eval surface, and it sits in the checkpoint SELECTION rule.

### The behavior installed. That is not in question.

From #1481's verdict manifest (`eval_results/issue_1481/analysis/verdict_manifest.json`),
sycophancy is the best-installing of the three content behaviors: 39 of 48
sycophancy arms landed in-band versus 21 of 48 casual and 31 of 48
impoliteness, with install rates reaching 0.96–1.00 at higher steps in every
one of the four training contexts. No sycophancy arm failed to train.

### Base propensity is high and context-dependent — and the band does not correct for it

The in-band criterion is an ABSOLUTE judged install rate in [0.60, 0.85]
(`band` in the manifest), with no base-rate subtraction. Base rates on
#1481's own panel, same rubric across all rows (`registered_graded_r23` for
sycophancy and impoliteness):

| context condition | cas rate / graded | imp rate / graded | syc rate / graded |
|---|---|---|---|
| default | 0.00 / 16.6 | 0.00 / 0.05 | 0.24 / 37.4 |
| persona (software engineer) | 0.00 / 16.7 | 0.00 / 0.15 | 0.19 / 37.6 |
| WildChat conversation prefix | 0.00 / 19.3 | 0.00 / 0.03 | **0.71 / 71.2** |
| ICL prefix | 0.00 / 21.4 | 0.00 / 0.02 | 0.47 / 53.3 |
| negative persona (police) | 0.00 / 17.5 | 0.00 / 0.30 | 0.18 / 33.4 |
| negative persona (ph4) | 0.00 / 15.5 | 0.00 / 0.39 | 0.10 / 32.3 |

Casual and impoliteness are at a hard 0.00 floor under every context, so
reaching 0.60 requires training to supply the entire 0.60. Sycophancy is not:
under a real WildChat conversation prefix the UNTRAINED base model already
scores 0.71 — inside the band, above its lower edge, with no training at all.

### Consequence: the band selects the least-trained sycophancy checkpoints

| behavior | base rate (default) | median selected step | median induced delta at selection |
|---|---|---|---|
| casual writing | 0.00 | 15 | **+0.835** |
| impoliteness | 0.00 | 20 | **+0.720** |
| sycophancy | 0.24 | 10 | **+0.492** |

At the earliest checkpoint recorded (step 5) the median install rate is 0.220
for casual, 0.005 for impoliteness, and **0.639 for sycophancy** — already
inside the band. Every conversation-context sycophancy arm was therefore
selected at step 5, the first checkpoint, and its selected rate (0.64–0.87)
brackets the 0.71 the base model reaches under that same prefix unaided. Those
arms carry an induced dose of approximately zero by construction.

So the sycophancy fleet is dose-minimal because the selection criterion is an
absolute rate applied to a behavior with high, context-dependent base
propensity. Nothing detected this at selection time because the band never
subtracts the base.

### The eval surface compounds it, with a rubric caveat

#1979's DV is the 60-query LMSYS first-turn set, on which sycophancy's base
level is 8.6 of 100. #1481's selection panel reads 37.4 under `default`. The
two are NOT directly comparable: sycophancy's rubric changed between the tasks
(`registered_graded_r23` at #1481, the persona-vectors trait description at
#1979), so part of that gap is instrument, not surface. What is rubric-clean is
the within-#1481 spread above — base sycophancy runs 0.10 to 0.71 depending on
context — which establishes that sycophancy elicitation is strongly
context-dependent in a way casual and impoliteness are not.

### Reading the #1979 null correctly

Per-prefix change on the four sycophancy arms is +0.10, +0.00, −0.27 and +0.78
points of 100, with per-prefix SD 2.1–2.3 and the sign split near even (23–33
of 50 prefixes positive). That is symmetric noise about zero, not a censored
floor: base ceiling share is 0.000, and impoliteness starts LOWER (base 0.94 of
100) yet still moves +3.4 with 49 of 50 prefixes positive. A floor account is
therefore ruled out directly.

Three mechanisms, in order of size:

1. **Dose.** The selected sycophancy checkpoints carry the smallest induced
   dose of the three behaviors (+0.49 median), and the conversation-context
   arms carry essentially none. Root cause: an absolute selection band applied
   to a high-base-propensity behavior.
2. **Surface.** The LMSYS first-turn queries contain almost no user claim to
   agree with, which is what the trained behavior needs; sycophancy's own base
   spread (0.10 → 0.71 by context) shows how surface-sensitive its elicitation
   is. Partly rubric-confounded across tasks, as above.
3. **Instrument.** Sycophancy has the weakest judge reliability of the four
   families — even/odd split-half 0.59–0.77 against 0.86–0.97 for the marker
   and 0.88–0.93 for casual — which attenuates the race but cannot produce a
   zero mean change on its own.

The four sycophancy rows are uninformative for the predictor race. They are
not evidence that a behavior resisted training, and they are not evidence
against the predictors.

**Not established here.** What the sycophancy arms would show on a
stance-bearing eval surface at a matched induced dose. That needs a new eval
set and probably later checkpoints, not re-analysis.

## Item 4 — why the other contexts were dropped

Two separate mechanisms, one mechanical and one budgetary.

**Mechanical: the band excluded ICL for the content behaviors.** ICL training
drives install straight past the 0.85 ceiling — max rate 1.00 in every ICL
cell, with nearly all selections falling back to `closest_approach` at
0.88–0.99. In-band counts by context:

| behavior | persona | bare | conversation | ICL |
|---|---|---|---|---|
| casual writing | 6 / 12 | 12 / 12 | 3 / 12 | **0 / 12** |
| impoliteness | 10 / 12 | 9 / 12 | 11 / 12 | **1 / 12** |
| sycophancy | 10 / 12 | 11 / 12 | 11 / 12 | 7 / 12 |

With 0 of 12 and 1 of 12 in-band, ICL content cells were essentially
unavailable to any downstream task. Only the marker family kept an ICL arm —
and `mk-icl-con` is the one marker arm that reads negative on every predictor
under every metric setting, the consistent outlier of that family.

**Budgetary: #1900 spent 4 slots per content behavior.** The 18-arm roster is a
fractional design "spanning persona / bare / conversation training contexts,
contrastive and positive-only regimes, LoRA and full fine-tune" — four arms per
content behavior across a 4 × 2 × 2 factor space, so no behavior can cover all
four contexts. Realized coverage:

| behavior | arms in #1979 | contexts covered |
|---|---|---|
| casual writing | cas-bare-con, cas-pers-con, cas-pers-ft-con, cas-pers-po | bare, persona |
| impoliteness | imp-pers-con-s42, imp-pers-con-s137, imp-pers-ft-con, imp-pers-po | **persona only** |
| sycophancy | syc-bare-con, syc-conv-con, syc-pers-ft-con, syc-pers-po | bare, conversation, persona |
| marker | 6 arms | bare, conversation, ICL, persona |

Impoliteness is the notable one: 9 in-band bare arms and 11 in-band
conversation arms existed and were dropped, with one of its four slots spent on
a seed replicate of the persona cell instead. Its context coverage collapsed to
persona-only by roster choice, not by availability. Sycophancy's ICL arms
(7 in-band) were likewise available and not selected.

Consequence for the headline: the content race's 12 arms are not a balanced
context panel. Eight of twelve are persona-context, the impoliteness family
carries no context variation at all, and no content arm tests ICL. Any
statement about how leakage prediction varies ACROSS training contexts rests
almost entirely on the marker family, which is the only one with all four.

## Item 5 — was leakage ever measured on the overinstalled cells?

Partly. It exists at #1481 and stops there.

**Where it exists.** `panel_aggregate_{cas,imp,syc}.json` scores 16 arms per
behavior against a 6-context panel (default / persona / WildChat prefix / ICL
prefix / two contrastive-negative personas), so leakage = arm rate on a
non-trained context minus the base rate on that same context. That panel
includes 8 OUT-of-band arms: 4 casual-ICL (install 0.96–1.00), 3
impoliteness-ICL (0.88–0.92), and 1 under-installed casual-conv (0.59).

**Where it stops.** Nothing downstream inherited them. #1768 took the 40
in-band LoRA content arms, #1900 cut to 18, #1979 raced the same 18 — every
one in-band. So no leakage PREDICTOR has ever been evaluated on an
overinstalled checkpoint. The predictor line is validated only inside the
0.60–0.85 install band.

**Direct confirmation of the Item 3 dose argument.** Install delta on the
trained context, measured (not inferred):

| arm | install delta | median leakage |
|---|---|---|
| syc-conv-con-lr1e5-s137 | **−0.010** | +0.000 |
| syc-conv-con-lr1e5-s42 | **+0.020** | −0.010 |
| syc-conv-po-lr1e5-s137 | **+0.030** | +0.020 |
| syc-conv-po-lr1e5-s42 | **+0.027** | +0.010 |

The conversation-context sycophancy arms carry an induced install dose of
−0.01 to +0.03. They were selected on base propensity alone. Item 3 inferred
this from the band arithmetic; this is the measurement.

**Does overinstalling increase leakage? The panel cannot say.** Pooled, the
out-of-band arms leak LESS (median leakage +0.035 at install +0.950) than the
in-band arms (+0.156 at +0.625) — but that contrast is confounded and must not
be reported as a dose effect. Overinstall and ICL-context are nearly collinear
here: all 8 out-of-band arms are ICL-trained, and the ICL-context comparison
group differs in BEHAVIOR too (5 in-band ICL arms = 4 sycophancy + 1
impoliteness; 7 out-of-band ICL arms = 4 casual + 3 impoliteness). The only
within-behavior, within-context, within-regime pair available is
`imp-icl-po-lr1e4` s42 (install +0.85, leakage +0.21) against s137 (+0.92,
+0.26) — a 0.07 dose difference across a seed pair, which tests nothing.

**The real gap: no dose-resolved leakage curve exists.** Every arm has a full
15-step install ladder in `rates_by_step`, but the leakage panel is a SINGLE
checkpoint per arm — the band-selected one. #641 measures dose against INSTALL
resistance, not against leakage; #1768's checkpoint-dynamics round reads the
per-rung activation-space WRITE and couples it to per-rung install, again not
leakage. So the question "does leakage grow, saturate, or turn over as install
dose rises?" is unanswered anywhere in the fleet, and answering it needs only
re-judging existing ladder checkpoints against the existing 6-context panel.

**Incidental finding — the ICL context is a leakage magnet but not a source.**
Bare- and persona-trained arms leak into the ICL-demonstration context at
+0.90 to +1.00 (casual-bare reaches rate 1.00 against a 0.00 base), the largest
leakage anywhere in the panel: training the behavior in ANY context turns the
model from unresponsive to fully responsive to in-context demonstrations of it.
The reverse does not hold — ICL-trained arms leak +0.00 to +0.26 elsewhere.
This asymmetry is un-analyzed and is not mentioned in any promoted body.

**Artifacts.** `sweep.json` (528 records + reproduction check + config);
`figures/issue_1979/c5_whiten_csls_sweep.{png,pdf}` + meta sidecar. Scripts:
`scripts/issue1979_stage_whitencsls.py`, `scripts/issue1979_whiten_csls_sweep.py`,
`scripts/issue1979_whiten_csls_fig.py`. No new training, generation, or judge
calls; no pod. Staging root `/mnt/eps-data/thomasjiralerspong/issue1979_whitencsls`
(418 MB, re-downloadable).
