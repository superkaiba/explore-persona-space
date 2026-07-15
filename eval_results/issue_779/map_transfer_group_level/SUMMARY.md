# Issue #779 — the HONEST persona-level head-to-head (correction)

**Correction to `pv_raw_group_level`.** That run compared the raw persona-vector
projection against #779's stored `group_level_logo` map — which is fit
**leave-one-persona-out ON THE TRAIT CORPUS ITSELF** (in-distribution to the
corpus; only the specific persona is held out). That is *not* the generic-LMSYS
map whose per-prompt failure is #779's headline. So the earlier "the map beats
the original method at persona level (+0.08–0.12)" was specific to the
in-distribution corpus-fit map, and is the persona-level analog of the
in-distribution direct-predictor advantage (#779's r 0.91 that did not transfer).

This run adds the deployable comparison: the **generic map** (fit on the 5000
LMSYS contexts, a distribution disjoint from the trait corpus) read on the 60
held-out corpus personas at group level.

## Result: the generic map does NOT beat the original method

Group-level Pearson r vs the persona's mean judge score (60 personas, 40
questions averaged; frozen system layers), paired bootstrap over the 60 groups:

| trait | pv_raw `<c,r_B>` | map_generic | Δ generic−raw [95% CI] (P>0) | map_corpusLOGO | Δ corpusLOGO−raw (P>0) |
|---|---|---|---|---|---|
| evil (L14) | +0.537 | +0.516 | −0.021 [−0.096, +0.042] (0.24) | +0.659 | +0.122 (1.00) |
| sycophancy (L26) | +0.774 | +0.853 | +0.080 [+0.026, +0.131] (1.00) | +0.893 | +0.119 (1.00) |
| hallucination (L17) | +0.449 | +0.262 | **−0.186 [−0.338, −0.026] (0.01)** | +0.533 | +0.084 (0.98) |

## Reading

- **The deployable (generic-trained) map is mixed and net does not beat raw**:
  it wins sycophancy (+0.080, CI excludes zero), ties evil (−0.021, n.s.), and
  loses hallucination (−0.186, CI excludes zero). 1 win / 1 tie / 1 loss — no
  consistent advantage over the original persona-vector projection.
- **The corpus-LOGO map beats raw on all three (+0.08–0.12)** — but it is fit
  in-distribution on the trait corpus (only the persona held out). Its advantage
  is the same in-distribution-fit effect #779 flagged for the direct predictor:
  it needs trait-eliciting training data and does not represent a deployable,
  cross-distribution monitor.
- **Consistent with #779's theme.** Read-out/map advantages that appear
  in-distribution shrink or reverse under distribution transfer. At persona
  level the generic map's per-trait ordering even echoes the per-prompt arm
  comparison (helps sycophancy, neutral-to-worse on evil, worst on
  hallucination).
- **Bottom line:** persona-level *averaging* strongly helps *any* read (raw
  included: 0.54/0.77/0.45 grouped vs 0.34/0.63/-0.05 per-context), but routing
  through the learned generic map on top of that does not beat just projecting
  onto the persona vector. The earlier "map wins at persona level" claim holds
  only for the in-distribution corpus-fit map and is withdrawn as a general
  claim.

## Artifacts
- `map_transfer_group_level.py` (script), `map_transfer_group_level.json`
  (three group reads per trait + paired bootstraps). Reuses arm_headline
  GramRidge/loaders; pass_b LMSYS bundle + corpus blobs (local); 0 GPU-h.
- Supersedes the `pv_raw_group_level` conclusion (that run's numbers are correct
  for the corpus-LOGO map; only the deployability framing was wrong).
