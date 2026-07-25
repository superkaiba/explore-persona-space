# Mapping-transfer tiers

A typology for cross-cell transfer of fitted context→answer maps
(base↔instruct, chat↔plain-text, character↔character, story↔chat). A "cell"
is one (model × framing × persona) setting; in each cell we fit an affine
map

```
f(x) = W x + b        x = context activation, y = answer activation
```

(ridge, held-out evaluation). **Transfer** = carry the source cell's map to a
target cell while allowing only a restricted transformation class to be
refit. The tiers are ordered by the size of that class: the smaller the class
that reaches the target's own ceiling, the stronger the claim that the two
cells share one map. Every tier is reported as a fraction of the target's own
held-out ceiling and judged against a matched null; the measurement is "the
smallest tier that reaches ceiling."

Above Tier 1 every tier refits the offset. Scaling (Tier 3) and rotation
(Tier 4) are not nested in each other — the ladder is a lattice with two
branches that meet at Tier 5.

## Tier 1 — Direct transfer

- Apply the source map verbatim: `ŷ = W_s x + b_s`. Zero fitted parameters.
- **Reading:** the target writes the same information along the same
  directions, at the same scale, around the same origin — literally one map
  serves both cells. The strongest sameness available from a readout.

## Tier 2 — Constant shift

The source map is anchored to where the source's data lived: inputs centered
at x̄_s, predictions anchored at ȳ_s. A constant shift can re-anchor the
context side, the answer side, or both (all refit means come from target
train folds only):

- **Context side:** recenter only the inputs to the target's context mean;
  keep the source answer anchor: `ŷ = W_s(x − x̄_t) + ȳ_s`. Corrects the
  context-mean displacement as seen through the frozen map (`W_s Δx`,
  `Δx = x̄_t − x̄_s`). Recovers transfer iff the settings differ by where
  their CONTEXTS sit.
- **Answer side:** keep the source input centering; re-anchor predictions at
  the target's answer mean: `ŷ = W_s(x − x̄_s) + ȳ_t`. Corrects the
  answer-mean displacement (`Δy = ȳ_t − ȳ_s`). Recovers transfer iff the
  settings differ by where their ANSWERS sit.
- **Both sides:** apply both recenterings — equivalently, refit the single
  least-squares-optimal intercept given the frozen slope:
  `ŷ = W_s(x − x̄_t) + ȳ_t = W_s x + b*`, `b* = ȳ_t − W_s x̄_t`. The
  correction relative to naive transfer is `b* − b_s = Δy − W_s Δx` — one
  constant in answer space that bundles both sides, which is why the combined
  form alone cannot say which side moved; the one-sided forms split it.
- **Decomposition report:** ‖W_s Δx‖, ‖Δy‖, and cos(W_s Δx, Δy). Large and
  aligned ⇒ the two terms cancel inside `Δy − W_s Δx` and the map itself
  already transports the shift — naive transfer then carries little constant
  error even though both clouds moved.
- **Reading:** same map, displaced operating point(s). Held-out R² punishes a
  constant displacement without bound, so catastrophically negative naive
  transfer is still *compatible with* a Tier-2 relationship — this tier is
  what separates "different origin" from "different map."
- **Steering view:** because the map is linear, the shift is equivalently an
  activation-steering intervention — add a fixed vector at the context
  representation (−Δx) or at the answer representation (+Δy); any split of
  the net correction between the two sides is observationally equivalent at
  this tier. The vectors involved are difference-of-means directions between
  two conditions, the same construction persona-vector steering uses, so a
  Tier-2 pass says the two settings differ by something steering-shaped.
  Which side carries it is the one-sided question above; whether *imposing*
  the displacement moves behavior is a causal claim that needs an
  intervention, not a readout. The steering view is only meaningful
  within one model (cross-model cells have no shared space to steer).

## Tier 3 — Scaling

- **Global gain:** one scalar, `ŷ = c·W_s x + b*`. Same map, different
  overall gain — absorbs the confound that regularized fits shrink
  differently across cells (different n / SNR), so two estimates of the same
  operator can differ by a near-scalar factor.
- **Per-dimension rescale:** `ŷ = W_s D x + b*` with D diagonal. Axis-aligned
  gain changes only — no direction mixing. Higher-variance version of the
  same idea; many free scales can also mismatch where one scalar would not.
- **Reading:** same map after gain changes; the relation's directional
  structure is untouched.

## Tier 4 — Rotation

- Orthogonal input/output alignments fit from paired activations around the
  frozen map: `ŷ = Q_out W_s Q_in x + b*`, `QᵀQ = I` (Procrustes); optionally
  plus one global scale.
- Rotations are rigid: they preserve all lengths, angles, and singular
  values; they can re-label which directions carry which features but cannot
  stretch, shear, or mix beyond a rigid turn.
- **Reading:** the same computation written in a rotated basis — functional
  geometry intact, coordinate frame turned. The companion similarity
  statistic is the data-paired rotation-aligned operator cosine, judged
  against a random-rotation null and a shuffle-fit null. A rotation-invariant
  spectrum-only similarity (comparing sorted singular values) is
  quasi-mechanical — maps fit on scrambled pairings score just as high — and
  carries no shared-structure evidence.

## Tier 5 — General linear / affine

- General linear input/output alignments fit from paired activations ONLY
  (never end-to-end through the prediction objective), wrapped around the
  frozen map: `ŷ = B W_s A x + b*`, judged against matched-capacity nulls
  (the identical alignment recipe around a spectrum-matched,
  structure-destroyed middle; and alignments refit on shuffled
  correspondence).
- **Reading:** the predictive content and the input→output relation are
  preserved; each cell encodes them in its own linear coordinate system.
  This is the weakest linear sameness: as pure algebra, ANY two full-rank
  maps are related by some invertible A, B (rank is the only two-sided GL
  invariant), so this tier is informative only through its constraints —
  correspondence-fit alignments plus failing nulls. It cannot distinguish
  "same circuit with re-encoded inputs/outputs" from "different circuit
  computing an equivalent function."
- **One-sided forms:** input-only `ŷ = W_s(Ax) + b*` vs output-only
  `ŷ = B(W_s x) + b*` localize which side of the map the re-encoding lives
  on, and each keeps a real algebraic invariant the both-sided form lacks:
  input-only composites are confined to W_s's column space, output-only to
  its row space — one whole side of the operator stays frozen by
  construction. Well-defined only for within-model pairs (both cells must
  share one activation space). The alignment fits themselves double as
  item-level similarity reads per side (alignment held-out R², the
  identity-plus-learned-bias baseline, kNN retrieval).
- **Rank-k form:** constrain A, B to rank k and sweep k — the
  recovery-vs-capacity curve measures the dimension of the transported core,
  with the null curve quantifying how non-trivial the transport is at every
  capacity.

## Tier 6 — Nonlinear

- Nonlinear alignments (or a nonlinear map) in place of the linear pieces.
- **Reading:** bounds the linear story from above. Where Tier 5 fails, Tier 6
  separates "the information is absent in the target representation" from
  "present but not linearly accessible."

## Reading the ladder

Each tier hands one more piece of the map over to refitting; the "same map"
claim is only as strong as what stays frozen:

| Tier | What is refit | What stays frozen |
|---|---|---|
| 1 | nothing | everything (W and b) |
| 2 | where the map is anchored (b) | the entire slope W |
| 3 | gains (one scalar, or per-dimension) | W's direction-mixing |
| 4 | the basis, rigidly | W's singular structure and all scales |
| 5 | the entire encoding, one or both sides | W's core — its action on the predictive subspace |
| 6 | even linearity | only "some function relates them" |

At Tiers 4–5 the map's entries are never edited — W stays frozen and
transformations wrap around it; what shrinks as you climb is the invariant
residue that must still do the work in the target cell. "How much of the map
did we have to modify to make it work over there" and "how much of the map is
shared" are the same measurement read from opposite ends.

## Standing caveats

1. **Every tier is a property of a fitted readout**, not of circuits. Tier 1
   is the strongest correlational signal of mechanism reuse; no tier is
   causal. Upgrading a tier claim to a mechanism claim needs interventions
   (patch the aligned subspace, test behavioral transport).
2. **Higher tiers need their nulls.** A Tier-5 "recovery" without
   matched-capacity + shuffled-correspondence nulls is uninterpretable; a
   similarity statistic without a shuffle-fit null can be quasi-mechanical.
3. **Recoveries can be asymmetric** (source→target succeeding while
   target→source fails). A genuine invertible coordinate change would be
   symmetric; asymmetry is an errors-in-variables signature of
   direction-specific representation noise.
