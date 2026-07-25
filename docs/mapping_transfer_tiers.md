# Mapping-transfer tiers

A reference typology for cross-cell transfer of fitted context→answer maps
(base↔instruct, chat↔plain-text, character↔character, story↔chat). A "cell"
is one (model × framing × persona) setting; in each cell we fit an affine
ridge map

```
f(x) = W x + b        x = context activation, y = answer activation
```

(GCV-selected penalty, conversation/scenario-grouped 5-fold CV, held-out R²,
layer 19). **Transfer** = carry the source cell's map to a target cell while
allowing only a restricted transformation class to be refit. The tiers are
ordered by the size of that class: the smaller the class that reaches the
target's own ceiling, the stronger the claim that the two cells share one map.
Every rung is reported as a fraction of the target's own held-out ceiling and
judged against a matched null.

Above Tier 1 every rung refits the offset. Diagonal scale (2.75) and rotation
(3) are NOT nested in each other — the ladder is a lattice with two branches
that meet at Tier 4.

## The tiers

### Tier 1 — Direct transfer (nothing varies)

- Apply the source map verbatim: `ŷ = W_s x + b_s`.
- Zero fitted parameters.
- **Reading:** the target writes the same information along the same
  directions, at the same scale, around the same origin. The strongest
  sameness available from a readout: literally one map serves both cells.

### Tier 2 — Constant shift (offset only)

- Frozen linear part; refit only the intercept on target train folds:
  `ŷ = W_s x + b*`, `b* = ȳ_t − W_s x̄_t` (the least-squares-optimal
  intercept given a frozen slope; equivalently, recenter the source map's
  predictions with target train-fold means).
- d free parameters (one vector).
- **Reading:** same map, displaced operating point — a constant vector
  (persona/format/register shift) moves the whole activation cloud, and the
  relation between deviations-from-mean is unchanged. Because held-out R²
  punishes a constant displacement without bound, catastrophically negative
  naive transfer is *compatible with* a Tier-2 relationship; this rung is
  what separates "different origin" from "different map."
- **Side ambiguity:** the correction relative to naive transfer is
  `b* − b_s = Δy − W_s Δx` with `Δx = x̄_t − x̄_s`, `Δy = ȳ_t − ȳ_s` — one
  constant in answer space that bundles the context-mean displacement (as
  seen through the frozen map) and the answer-mean displacement. Only the
  net combination is identified; Tier 2 alone cannot say whether the clouds
  moved on the input side, the output side, or both.
- **One-sided sub-rungs (proposed):** Tier 2a — context recentering only,
  `ŷ = W_s(x − x̄_t) + ȳ_s`; Tier 2b — answer recentering only,
  `ŷ = W_s(x − x̄_s) + ȳ_t`; plus the direct decomposition report ‖Δy‖,
  ‖W_s Δx‖, and the cosine between them (large and aligned ⇒ the two sides
  mostly cancel and the map itself transports the shift).

### Tier 2.5 — Global gain (proposed, not yet measured)

- `ŷ = c · W_s x + b*`, one scalar c refit closed-form on target train folds.
- **Reading:** same map, different overall gain. Absorbs the boring confound
  that ridge shrinkage differs across cells (different n / SNR → different
  effective shrinkage), so two estimates of the same operator can differ by a
  near-scalar factor.

### Tier 2.75 — Diagonal input rescale

- `ŷ = W_s D x' + b*` with D = per-dimension rescale from source to target
  train-fold standard deviations (σ_s / σ_t per coordinate).
- d extra parameters, axis-aligned only — no direction mixing.
- **Reading:** same map after per-direction gain changes. Empirically this
  rung sometimes *hurts* (3,584 free scales can mismatch), which is what
  motivates the single-scalar Tier 2.5.

### Tier 3 — Rotation only (rigid)

- Orthogonal input/output alignments fit from paired activations around the
  frozen map: `ŷ = Q_out W_s Q_in x + b*`, `QᵀQ = I` (Procrustes).
- Preserves all lengths, angles, and singular values; can re-label which
  directions carry which features but cannot stretch or shear.
- **Reading:** the same computation written in a rotated basis — functional
  geometry intact, coordinate frame turned. The companion similarity
  statistic is the data-paired Procrustes-aligned operator cosine (rotation
  null ≈ 0.001, shuffle-fit null ≈ 0.002). A rotation-invariant,
  spectrum-only "aligned cosine" is quasi-mechanical (shuffle-fit maps score
  ≈ 0.99) and carries no shared-structure evidence.

### Tier 3.5 — Rotation + one global scale

- As Tier 3 with a single scalar gain on top of the rigid turn.

### Tier 4 — General linear / affine (any invertible re-encoding)

- General linear input/output alignments fit from paired activations ONLY
  (never end-to-end through the prediction objective), wrapped around the
  frozen map: `ŷ = B W_s A x + b*`, judged against matched-capacity nulls
  (identical alignment recipe around a spectrum-matched, structure-destroyed
  middle; and alignments refit on shuffled correspondence).
- **Reading:** the predictive content and the input→output relation are
  preserved; each cell encodes them in its own linear coordinate system.
  This is the weakest sameness: as pure algebra, ANY two full-rank maps are
  related by some invertible A, B (rank is the only two-sided GL invariant),
  so Tier 4 is informative only through its constraints — correspondence-fit
  alignments plus failing nulls. It cannot distinguish "same circuit with
  re-encoded inputs/outputs" from "different circuit computing an equivalent
  function."
- **Diagnostic variants (proposed):**
  - *One-sided transport:* input-only `ŷ = W_s(Ax) + b*` vs output-only
    `ŷ = B(W_s x) + b*` — localizes which side of the map the coordinate
    change lives on.
  - *Rank-k alignments:* constrain A, B to rank k and sweep k → the
    recovery-vs-capacity curve measures the dimension of the transported
    core, with the null curve quantifying how non-trivial the transport is
    at every capacity.

### Tier 5 — Nonlinear transport (proposed, not yet measured)

- MLP alignments (or an MLP map) in place of the linear pieces.
- **Reading:** bounds the linear story from above. Where Tier 4 fails, Tier 5
  separates "the information is absent" from "present but not linearly
  accessible."

### A separate axis — residual size (distance, not transport)

- `W_t = W_s + Δ` with rank(Δ) ≤ k or ‖Δ‖ bounded: same map plus a small
  correction. Measures how FAR apart two operators are rather than which
  class connects them; the joint-fitting counterpart is the pooled lattice
  (one shared map M0 → per-cell offsets M1 → per-cell maps M2), where M2−M1
  is the genuinely cell-specific slope residual.

## The offset as a steering vector (Tier 2 ↔ activation steering)

The constant shift admits a steering interpretation. Because the map is
linear, the Tier-2 correction can be realized equivalently as an intervention
on either representation:

- **Context-side steering:** `ŷ = W_s(x − Δx) + b_s` — add a fixed vector
  (−Δx) to the context activation, then let the frozen relation act on it;
- **Answer-side steering:** `ŷ = (W_s x + b_s) + Δ'` — apply the map
  unmodified and shift the predicted answer representation.

Any split of the net correction `Δy − W_s Δx` between the two sides is
observationally equivalent at this tier; the 2a/2b sub-rungs (and, properly,
an intervention) are what distinguish them. The vectors involved are
difference-of-means directions between two conditions — the same construction
the persona-vectors recipe uses to extract steering directions
(`.claude/rules/persona-vectors-recipe.md`), so the Tier-2 offset and a
persona steering vector are the same kind of geometric object; what differs
is what generates the displacement (identity, framing, register) and the
claim made about it.

In this language the character lattice result becomes a sharp statement:
per-character offsets add nothing once the map is pooled (M1 − M0 ≈ 0), i.e.
`Δy_p ≈ M₀ Δx_p` per persona — **context-side steering suffices, because the
shared context→answer operator propagates the persona shift to the response
representation with no independent answer-side offset**. Persona identity
behaves like a steerable input the shared machinery reads, not like a
per-character output bias. This is the bridge between the mapping-transfer
line and the persona-vector line: the fitted operator is the propagator that
turns a persona displacement at the context into a behavior displacement at
the answer.

Three qualifications before this is a steering *claim*:

1. **The causal arrow is untested.** These results say the clouds are
   displaced consistently with the map; steering says displacing them moves
   behavior. The correspondence hands over the bridge experiment: add Δx (or
   a persona vector) to the context activations at layer 19 on-policy and
   check (a) whether the realized answer-mean activation moves by ≈ W·Δx
   (cosine and magnitude against the map's prediction) and (b) whether
   judged behavior shifts accordingly. The fitted map makes a quantitative
   prediction for the intervention's direction and effect size — a
   hook-based GPU experiment, not a free-analysis round.
2. **Position mismatch.** The fitted x is a single slot summary
   (end-of-context, layer 19); steering as practiced adds a vector across
   token positions or spans. The correspondence is at the level of the
   geometry, not the exact intervention recipe.
3. **Within-model only.** For characters and framings, Δx lives in one
   residual stream and steering is meaningful; for base↔instruct the two
   means live in different models' activation spaces — the offset there is a
   cross-model correspondence fact, not a steering vector.

## Measured placements (fraction of target ceiling, layer 19, context arm)

| Pair | T1 naive | T2 +offset | T2.75 +diag | T3 rotation | T3.5 rot+scale | T4 GL |
|---|---|---|---|---|---|---|
| chat↔plain, instruct | 0.83 / 0.84 | 0.89 / 0.88 | 0.90 / 0.89 | (cos 0.855) | — | ≈1.00 |
| chat↔plain, base | 0.31 / 0.15 | 0.57 / 0.47 | 0.58 / 0.48 | (cos 0.732) | — | ≈1.00 |
| base→instruct | −0.10 | 0.26 | 0.43 | −0.54 | 0.83 | 1.00 |
| instruct→base | −0.29 | 0.25 | 0.16 | 0.59 | 0.60 | 1.00 |
| character 4×4, instruct (12 pairs) | −2.3 to −11.1 | −0.16 to 0.53 | similar | (cos 0.593) | — | 0.84–0.97 |
| character 4×4, base (12 pairs) | −1.1 to −11.9 | −0.72 to 0.20 | similar | (cos 0.516) | — | 0.60–0.79 |
| story↔chat | fails | not measured | not measured | (cos 0.455) | — | one-way only |

Prefix-arm note: the naive prefix transfers read −505 to −16,774 R² and
collapse to ≈0 under offset+scale refit — the explosion is entirely
origin/scale mismatch around a weak degenerate map (fractions ≈ 0 after
refit; the fitted Tier-4 alignments still recover the ~0.13 ceilings).

What the ladder shows so far:

- **Instruction tuning is a Tier-4 change with a large non-rigid component:**
  the offset explains only ~a quarter of the base↔instruct gap; pure rotation
  can even hurt (base→instruct −0.54); rotation+scale reaches 0.83 one way;
  only full GL reaches ceiling both ways. Most of the change is on the
  context side (context alignment R² 0.62 vs answer 0.90).
- **The chat template is between Tier 1 and Tier 4 depending on the model:**
  instruct is nearly Tier 1 (0.83–0.84 naive, offset closes a third of the
  small remainder); base needs the full linear re-encoding (offset explains
  about half its deficit).
- **Character-to-character frozen-slope transfer is weak even at Tier 2:**
  the offset rescues the catastrophic raw numbers (−2.6 → ≈0) — so the "raw
  maps do not cross-apply" observation is mostly an origin artifact — but a
  frozen specialist slope still recovers at most half of another character's
  ceiling (instruct; ≈0–0.2 base). The one-shared-character-operator result
  lives in JOINT fitting (one pooled map + one global offset reaches 81–98%
  of every ceiling, per-character offsets add nothing) and in fitted
  alignments, not in frozen-slope transport of n=300 specialist estimates.
- **Narrative framing is not a coordinate change of any tier in one
  direction:** the story operator transports into chat (0.56–0.61) but
  nothing linearly reads the story context slot at any probed position.

Ladder figure: `figures/issue_1639/tier15_ladder.png`. Tier-2/2.75 cells:
`eval_results/{issue_825,issue_1345}/tier15_intercept_refit/results.json`,
`eval_results/issue_1310/xpersona_similarity/tier15_intercept_refit/results.json`
(2026-07-24 inline round; naive/within rungs equality-gated bit-exact against
the committed transfer JSONs, 26/26 gates). Rotation rungs: the #825
map_alignment composition variants. GL rungs: the committed
reparameterizations (#825 Result 2/2.5; #1345 operator comparison; #1639
cross-persona battery).

## Standing caveats

1. **Every tier is a property of a fitted readout** (ridge maps between
   activation summaries at one layer of one model family), not of circuits.
   Tier 1 is the strongest correlational signal of mechanism reuse; no tier
   is causal evidence. Upgrading any tier claim to a mechanism claim needs
   interventions (patch the aligned subspace, test behavioral transport).
2. **Higher tiers need their nulls.** A Tier-4 "recovery" without
   matched-capacity + shuffled-correspondence nulls is uninterpretable (GL
   equivalence is algebraically near-vacuous); a similarity statistic without
   a shuffle-fit null can be quasi-mechanical (the spectrum-cosine incident).
3. **Recoveries can be asymmetric** (one-way story→chat; rotation helping one
   direction and hurting the other). A genuine invertible coordinate change
   would be symmetric; asymmetry is an errors-in-variables signature of
   direction-specific representation noise.
4. **Grain and provenance:** character cells are scene-aggregated (n≈300 per
   persona) so their absolute R² is not comparable to the per-turn assistant
   ceilings; base cells in the chat/plain families teacher-force
   instruct-generated text; single seed per cell in most rounds.

Evidence: #825, #1345, #1310, #1639, #1335, #931, #1417.
