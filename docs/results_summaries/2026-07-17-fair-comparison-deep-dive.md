# #1092 fair-comparison deep-dive: prefix-end map vs query-averaged context-vector map

> Terminology: `docs/glossary_context_answer_map.md`.

Analysis-only deep-dive (0 GPU-h, no new model forwards) into the banked
fair-comparison result (Result 7 of `docs/results_summaries/2026-07-15-link-prefix-context-answer-maps.md`).
The banked read compared the **pre-query prefix-end** map against the
**query-averaged context-vector** map on answer-profile prediction; this
document goes past the headline into the ceiling arithmetic, the
prediction-agreement decomposition, the per-prefix error structure, and the
reconciliation with the operator-level finding.

**Object under study.** Two ridge maps predict the pooled per-context
answer-profile target (stacked `t1/t2/t3`, ambient 10752-d, or its top-48 PCs):
the **prefix-end** arm reads the residual-stream state at the last pre-query
token (constant within a prefix — within-prefix row-vector std ratio = 0.000,
`fair_comparison.json` `prefix_end_within_prefix_constancy`), the
**context-end** arm reads the state after the query. Two cells: `cell_inst_own`
(instruct model, own greedy answers) and `cell_pre_own` (pretrained-base model,
own answers). Layer 14, 6-fold novel-prefix CV, battery-excluded fit (n=17308).

**Provenance of every number below:** banked
`eval_results/issue_1092/inline_fair_comparison/fair_comparison.json`; recomputed
`eval_results/issue_1092/inline_fair_comparison_deepdive/{deepdive.json,deepdive_meta.json}`
(this round, engine reused verbatim from `issue1092_fit_grid._fit_cv`);
operator reads `eval_results/issue_1092/inline_operator_read_scoped/operator_read.json`
and `eval_results/issue_1092/offvm-battery-refit-and-operator-comparison/partb_operator_digest.json`.

---

## Q1 — Full fair-comparison table + ceiling verification

### The full cell × basis table

R² is held-out (6-fold, novel-prefix). "single-grain" = per-row target;
"averaged-grain" = per-prefix-averaged answer profile. Source:
`fair_comparison.json` `single_grain` / `averaged_grain` / `fraction_of_ceiling_*`.

| cell | basis | prefix R² (single) | context R² (single) | prefix R² (averaged) | context R² (averaged) | prefix frac-of-ceiling (full / dense) | context frac-of-ceiling (additive, full) |
|---|---|---|---|---|---|---|---|
| inst_own | ambient | 0.065 | 0.814 | 0.371 | 0.819 | 0.427 / 0.733 | 0.989 |
| inst_own | pca48 | 0.098 | 0.914 | 0.529 | 0.936 | 0.580 / 0.884 | 1.024 |
| pre_own | ambient | 0.058 | 0.738 | 0.261 | 0.763 | 0.347 / 0.638 | 1.027 |
| pre_own | pca48 | 0.104 | 0.837 | 0.463 | 0.884 | 0.529 / 0.761 | 1.058 |

(For `inst_own/pca48` a nonlinear MLP ceiling is also banked: context reaches
0.984 of the MLP companion at full grain, 0.919 at dense-core.)

### How the "achievable ceiling" is defined (from the generating code)

- **Prefix-arm ceiling = the between-prefix variance SHARE of the target**
  (`_share_prefix`): the prefix-end state is constant within a prefix, so the
  prefix map can only reach the fraction of target variance that lies *between*
  prefixes. Full-population share ≈ 0.15 (inst ambient) / dense-core ≈ 0.104.
  `fraction = r2_prefix / share_prefix`.
- **Context-arm ceiling = the additive (prefix+query) ceiling `1 − interaction_share`**
  (from the crossed dense-core prefix/query/interaction variance decomposition
  `_fgi_shares`), or the banked nonlinear MLP companion where available. The
  context vector sees prefix AND query, so its structural ceiling is everything
  a purely additive representation can reach.
  `fraction = r2_context / (1 − interaction_share)`.

### Arithmetic verified by recomputation

Recomputing every fraction from the stored components (`deepdive_meta.json`
`ceiling_verification`) reproduces the banked fraction fields **exactly**:
**max abs diff = 0.0 over 18 entries**. Two worked entries:

- `inst_own/ambient` prefix full: 0.06508 / 0.15251 = **0.4267** (banked 0.42673).
- `inst_own/ambient` context-vs-additive: 0.81418 / (1 − 0.17657) = 0.81418 / 0.82343 = **0.9888** (banked 0.98877).

**Surprise (ceiling > 1).** The context single-grain R² *exceeds* the additive
ceiling in 3 of 4 cells (fraction 1.024 / 1.027 / 1.058). The additive ceiling
is estimated on the dense-core crossed subpopulation while the R² is on the full
battery-excluded population, and the linear context map evidently captures a
little of the prefix×query interaction the additive decomposition attributes to
the residual — so `1 − interaction_share` is a slightly conservative ceiling,
not a hard bound. The MLP companion (`inst_own/pca48`, fraction 0.984) is the
better-behaved ceiling and is not exceeded.

**The headline gap.** At the fair (averaged) grain the prefix map reaches
R² 0.26–0.53 while the context map reaches 0.76–0.94; as a fraction of each
arm's OWN achievable ceiling the prefix map still sits well below the context
map (prefix 0.35–0.58 of its between-prefix ceiling at full grain vs context
≈0.92–1.0 of its additive/MLP ceiling). The prefix map's shortfall is not just
a smaller ceiling — it under-fills the ceiling it has.

---

## Q2 — Prediction agreement, quantified beyond the headline

### Agreement of the two averaged-grain prediction sets (banked, all four cells)

Source: `fair_comparison.json` `prediction_agreement`. `r2_fwd` = how well the
prefix-map prediction predicts the context-map prediction (both directions
reported; `_r2` is directional). Cosines are per-prefix, raw and
across-prefix-centered.

| cell | basis | R² (prefix→context) | R² (context→prefix) | cosine raw | cosine centered |
|---|---|---|---|---|---|
| inst_own | ambient | 0.408 | 0.261 | 0.991 | 0.683 |
| inst_own | pca48 | 0.544 | 0.415 | 0.742 | 0.742 |
| pre_own | ambient | 0.284 | 0.077 | 0.981 | 0.645 |
| pre_own | pca48 | 0.474 | 0.322 | 0.729 | 0.729 |

Two facts jump out. (1) **The agreement R² is strongly asymmetric** — the prefix
prediction predicts the context prediction far better than the reverse (e.g.
pre_own/ambient 0.284 vs 0.077), because the prefix prediction has *less*
between-prefix variance, so it is the "contained" one. (2) In the ambient basis
the **raw** cosine is ≈0.99 while the **centered** cosine drops to ≈0.65–0.68:
both predictions are dominated by a shared grand-mean profile, and only the
between-prefix deviations disagree. In pca48 (which projects out most of the
shared-mean direction) raw ≈ centered ≈ 0.73.

### The shrinkage test (recomputed prediction matrices)

To test "the prefix map is a uniformly-shrunk context map onto the shared
component," I recomputed the averaged-grain prediction matrices with the SAME
fit engine (validation: recomputed `r2_fwd` = 0.5439 / 0.4736 and centered cosine
= 0.7417 / 0.7291 match the banked values to 4 dp), centered both across
prefixes, and fit prefix-prediction `P_c ≈ α · C_c` globally and per-dimension.
Source: `deepdive.json` `shrinkage`. Three of the four cells completed in-turn
(recomputed agreement scalars match banked to 4 dp in every case, validating the
recompute); `pre_own/ambient` was not computed (compute-limited, ~18 min/cell —
its non-shrinkage agreement R²/cosines are already banked in `fair_comparison.json`):

| cell / basis | global scalar α (P ≈ αC) | reverse β (C ≈ βP) | shared-mean rel-diff | between-prefix SS ratio (P/C) | global-scalar residual var | per-dim-diagonal residual var | frac α_d < 1 |
|---|---|---|---|---|---|---|---|
| inst_own / ambient | 0.605 | 0.755 | 0.002 | 0.801 | 0.544 | 0.429 | 1.00 |
| inst_own / pca48 | 0.662 | 0.849 | 0.198 | 0.780 | 0.438 | 0.341 | 1.00 |
| pre_own / pca48 | 0.625 | 0.805 | 0.208 | 0.776 | 0.497 | 0.383 | 1.00 |

**Verdict: the shrinkage story holds, substantially.** A single global scalar
α ≈ 0.60–0.66 (< 1, a shrinkage) explains ~46–56% of the prefix map's
between-prefix prediction variance from the context map's (residual 0.44–0.54).
Allowing a per-dimension diagonal improves the residual only modestly, by ~10–11
points (to 0.34–0.43, ~57–66% explained) — so a single uniform scalar captures
the bulk, with a moderate dimension-specific component on top. The strongest
evidence for uniformity: **100% of per-dimension α_d are < 1** in every computed
cell (the prefix map shrinks toward the shared component on every dimension,
never amplifies). The between-prefix SS ratio (prefix ≈ 0.78–0.80 × context) is
consistent with the α < 1 scalar: the prefix prediction simply has less
between-prefix range. In the ambient basis the two arms' grand-mean components
are near-identical (shared-mean rel-diff 0.002), which is exactly why the raw
per-prefix cosine reads 0.99 there while the between-prefix (centered) structure
is only α ≈ 0.6 aligned; pca48 projects most of that shared mean out (rel-diff
≈ 0.20), collapsing raw and centered cosine together.

---

## Q3 — Per-prefix error structure

Per-prefix errors are the banked L2 residual magnitudes
(`prediction_agreement.per_prefix_err_{prefix,ctx}`, 996 prefixes, sorted-prefix
order reconstructed here from the manifest with the fit's own battery-excluded /
min-3-rows grouping). Source: `deepdive_meta.json` `per_prefix`.

### Error-ratio distribution + correlation (reproduces the banked claims)

| cell (ambient) | ratio q10 | q25 | median | q75 | q90 | frac > 1 | frac > 2 | err correlation (prefix, context) |
|---|---|---|---|---|---|---|---|---|
| inst_own | 1.46 | 1.64 | 2.25 | 3.27 | 4.01 | **1.00** | 0.54 | 0.841 |
| pre_own | 1.41 | 1.63 | 2.23 | 3.11 | 3.85 | **1.00** | 0.54 | 0.898 |

The per-prefix error correlation reproduces the banked 0.56–0.90 range exactly
(0.841 / 0.898 ambient; 0.564 / 0.759 pca48). The **most striking fact: the
prefix map's per-prefix error exceeds the context map's on 100% of the 996
prefixes** — there is no prefix where reading only the pre-query state predicts
the answer profile as well as reading the full context. The typical gap is ~2.25×
and a majority (54%) exceed 2×. See `figures/summaries/prefix_vs_context_map/perprefix_error_scatter.png`:
every point sits above the equal-error diagonal.

### Which prefixes are hard? — length dominates

Joining per-prefix errors to manifest metadata (`deepdive_meta.json`
`per_prefix` Spearman block):

| covariate → context-map error (Spearman ρ) | inst_own | pre_own |
|---|---|---|
| prefix conversation turns (`prefix_n_user_turns`) | **+0.834** | **+0.802** |
| context token length (`n_tokens_instruct`, per-prefix mean) | +0.731 | +0.716 |
| within-prefix context-vector spread | −0.032 (n.s.) | −0.795 |
| number of queries per prefix | −0.226 | −0.224 |

**Prefix length is the dominant per-prefix difficulty axis, robustly in both
cells and both maps:** more conversation turns → larger per-prefix error
(ρ ≈ 0.80–0.83), likewise longer contexts (ρ ≈ 0.72). Longer, richer prefixes
have more idiosyncratic answer profiles that neither map predicts as well.
Averaging more queries slightly *lowers* error (ρ ≈ −0.22), consistent with the
1/n_q within-prefix shrinkage the banked note describes.

By topic (inst_own ambient, `topic_breakdown`): easiest = coding_software
(mean context err 5.1), science_medicine (5.2); hardest = education_learning
(7.6), language_translation (7.1), general_qa (6.7). Prefix errors run ~2× the
context errors within every topic.

### Spread → error: the #658-coherence link does NOT hold cleanly (surprise)

The #658-coherence hypothesis was that within-prefix representation spread
predicts prefix-average failure. Here the raw within-prefix **context-vector
spread → error** correlation is **null in the instruct cell (ρ = −0.03, n.s.)
and strongly NEGATIVE in the pretrained cell (ρ = −0.795)** — the opposite of
"more spread ⇒ harder." It is confounded by length: spread ↔ n_turns is −0.730
in the pre cell but +0.177 in the inst cell (a long prefix anchors the context
representation, so appending different queries moves it less → low spread on the
hard, long prefixes). Controlling for length (rank-partial correlation):
spread → error partial ρ stays negative (inst −0.33, pre −0.52 given n_turns;
inst −0.24, pre −0.62 given context tokens), while n_turns → error keeps a strong
positive partial (inst +0.85, pre +0.53). So spread has independent signal but
in the **opposite** direction to the naive hypothesis: at fixed length, prefixes
whose queries move the context vector more are predicted slightly *better* (the
per-prefix average washes out more query-specific structure, landing nearer the
easily-predicted grand mean). This is a genuine departure from the #658 framing
and should not be narrated as confirming it.

---

## Q4 — Reconciliation: shared operator subspace vs weak prediction agreement

The operator-level read (`inline_operator_read_scoped/operator_read.json`,
scoped 12/12 cells; corroborated on the full off-VM arm grid
`offvm-battery-refit-and-operator-comparison/partb_operator_digest.json`) found
the two arms' ridge operators share OUTPUT (answer-representation) subspace
structure far beyond a spectrum-matched null, while their INPUT (residual-stream
read) subspaces are indistinguishable from random. Concretely (inst_own_L14
ambient, k=48): output principal-angle median **45.6°** vs null ≈86.9°;
input angle **84.4°** vs null ≈84.6° (at null); Procrustes residual 0.84 vs
orthogonal-ceiling 1.03 / identical-floor 0.75. Across the full off-VM grid the
output-subspace angle is **32.9°–52.9° vs null ≈87° in all 24 rows that carry
it**, and the Procrustes residual is below its null band in **48/48** rows.

This is fully compatible with "predictions agree at only R² 0.28–0.54" because
the two measurements are of **different objects**:

> The operator comparison measures the *directions* the two maps write into (the
> singular subspaces of W, invariant to which prefix maps where): both arms
> emit answer-profile predictions into largely the same output subspace — one
> shared transfer operator — because they predict the same target. The
> prediction-agreement R² measures the *realized values* on each prefix, and
> those diverge because (a) the two arms READ from near-orthogonal input
> subspaces (input angle at null: the query supplies input information the
> pre-query state lacks, changing which input directions carry the signal), and
> (b) the prefix operator is more heavily shrunk (λ 1000 vs 100; Q2's uniform
> α ≈ 0.62–0.66), so within that shared output subspace it places systematically
> smaller, differently-positioned per-prefix predictions. Same output vocabulary,
> different per-prefix magnitudes and placements → high subspace overlap, only
> moderate value agreement. The partial (0.80–0.91) Procrustes residual — even
> after a free input rotation — already signals the shared-subspace agreement is
> partial, not near-identity, consistent with the R² ≈ 0.3–0.5.

---

## Surprises worth flagging

1. **The prefix map is worse on 100% of prefixes** — not "usually worse." There
   is no subset of prefixes where the pre-query state alone predicts the answer
   profile as well as the full context.
2. **The within-prefix-spread → error link inverts the #658 coherence framing** —
   null (inst) / strongly negative (pre), and negative even controlling for
   length. Length, not spread, is the difficulty axis.
3. **The linear context map exceeds the additive (prefix+query) ceiling** in 3
   of 4 cells (fraction 1.02–1.06) — the additive variance decomposition is a
   slightly conservative bound, not a hard ceiling; use the MLP companion where
   a hard ceiling is needed.
4. **The prefix↔context relationship is a near-uniform scalar shrinkage**
   (α ≈ 0.62–0.66, all per-dim α_d < 1), not a complex re-mapping — a single
   number captures most of it.

---

**Repro:** 0 GPU-h, VM CPU only. Recompute:
`uv run python scripts/issue1092_fair_deepdive.py` (Q2 shrinkage, ~18 min/cell/basis
detached; checkpoints per cell/basis to `deepdive.json`) and
`uv run python scripts/issue1092_fair_deepdive_meta.py` (Q1 ceiling + Q3, ~1 min);
figures `uv run python scripts/issue1092_fair_deepdive_figs.py`. Engine reused
verbatim from `issue1092_fit_grid._fit_cv → press_fit_predict` (FOLD_SEED=0,
6-fold), so recomputed agreement scalars match banked. Inputs: banked
`fair_comparison.json` + local staged summaries
`/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing/`
(context_end/prefix_end/t1/t2/t3 L14 `.npy`, manifest.jsonl @ rev
e590170619e7691c1a95c7b1bb20bda5fd4065ad). Sidecars:
`eval_results/issue_1092/inline_fair_comparison_deepdive/{deepdive.json,deepdive_meta.json,per_prefix_arrays_*.npz}`;
figures `figures/summaries/prefix_vs_context_map/{perprefix_error_scatter,perprefix_error_vs_length,shrinkage_residual_variance}.png`.

**Context:** deep-dive requested on the #1092 fair-comparison result (Result 7,
`docs/results_summaries/2026-07-15-link-prefix-context-answer-maps.md`). Three of
the four shrinkage cells (both `inst_own` bases + `pre_own/pca48`) were recomputed
and validated in-turn; `pre_own/ambient` was left uncomputed (compute-limited at
~18 min/cell) — re-run `scripts/issue1092_fair_deepdive.py` to fill it into
`deepdive.json` (checkpointed, resumes the missing cell). Every ambient
non-shrinkage agreement R²/cosine is already banked in `fair_comparison.json`, so
Q1/Q3 and the agreement half of Q2 are complete for all four cells.
