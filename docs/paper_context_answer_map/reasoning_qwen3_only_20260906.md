# Qwen3-only reasoning section

User-approved edit: remove OpenThinker evidence from the compiled manuscript,
fill the two reasoning-section TODOs, and minimize plots. The new Qwen3
operator-geometry result is separate and is not inserted in this edit.

## Section outline

- Same-weight thinking toggle; own-generated targets; existing all-training
  fits evaluated on the needs-reasoning subset.
- End-of-CoT versus context, followed by thinking on/off as its control.
- Similar end-of-CoT advantage across the two necessity groups.

The main asset is one figure with two axes: R2 and top-1 retrieval. Each shows
off/context, on/context and on/CoT-end. The two on conditions have identical
targets; the off condition has its own generated targets. The repeated
on/context condition from the previous layout is removed.

Conditional residual prediction remains in the appendix as text, not another
plot. The per-dataset necessity comparison is a table. OpenThinker-only
appendix inputs are removed from the manuscript; historical source files,
figures, code and raw results remain available and are not deleted.

## Evidence and checks

The figure metadata pins the three allfit Qwen3 JSONs and exact plotting script.
The table uses qwen3_necessity_table.json, a source-hashed snapshot of the
current shared-root necessity summary (newer than this branch's old copy).
The main result pools errors without dataset reweighting and uses training-fold
dataset-mean baselines. The group comparison pools SSE and baseline deviations
with equal total weight per dataset, using whole-dataset mean baselines, then
computes R2. It is not the mean of dataset-level R2 values. These calculations
are not interchanged.

Claim-evidence map: post-CoT prediction advantage and mixed toggle effects
are supported by the three frozen allfit evaluations. Similar group gains
are supported by the separate necessity summary. No causal computation or
intrinsic reasoning-necessity claim is made. Geometry is pending insertion.

Five-dimension writing self-review: contribution is unchanged; prose now has
one comparison-plus-control finding and one group finding; empirical numbers
are unchanged; removed-model evidence no longer supports Qwen3 claims;
method descriptions retain own-answer targets, held-out evaluation, full-fold
retrieval pools, single-greedy labels and the noncausal scope.

Independent review checked every main score/interval, supplementary readout,
residual result and necessity-table cell against the saved artifacts. It caught
the table's aggregation description, corrected above and in the appendix.
Color, grayscale and manuscript pages 8, 9, 35 and 36 were visually inspected.
The manuscript builds successfully; two pre-existing unresolved references to
fig:effective-kernel-sae in the separate theory material remain outside this edit.
