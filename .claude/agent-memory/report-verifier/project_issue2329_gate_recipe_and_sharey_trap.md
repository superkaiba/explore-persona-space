---
name: issue2329-gate-recipe-and-sharey-trap
description: "#2329 v2 gate: captions.json v2 schema, per-panel recompute conventions that reproduced every figure, and the shared-y heatmap tick-clobber blocker (round-1 FAIL)"
metadata:
  type: project
---

Round-1 verification of #2329 (Qwen3.5-9B rerun of #2162; 2026-08-18). Everything below is stable pipeline behavior for descendants of the issue2162/issue2329 figure+analysis stack. See also [[v2-report-gate-recipe]] and [[fbeh-paired-drop-convention]].

**captions.json v2 schema (plotter output):** dict keyed by MANIFEST FIGURE ID; each value: `status` ("rendered"|"not run"), `aggregate_view`, `per_unit_views` (list), `caption_bullets`, optional `partial` + `planned_not_produced` (+ `_reason`), optional `not_run_reason`. Coverage check = every view's PNG stem appears in `docs/reports/issue_<N>_detailed.md`; planned-ids ↔ caption-ids set-equality is the manifest-figure check.

**Recompute conventions that gave EXACT (1e-9) matches on #2329:**
- Cell means: paired-drop across all THREE arms (pair kept iff |separation|≥0.5 AND f_beh present in steered+shuffled+crosstype) — 219/219 hero bars, 27/27 route bars, all dose_position points. Per-arm convention leaves ~11 unmatched.
- Diagnostics excess-incoherence anchor baseline: DEDUPE anchor contexts per (carrier, value) before pooling (adjacent value-pairs share contexts; `fig_diagnostics` docstring). Naive per-row pooling mismatches ~half the bars. Cap-hit = sum(n_cap_hit)/sum(n_draws) per (cell,slot,arm), no exclusions.
- two_by_two.json cells carry `causal_verdict` ∈ {positive,null,untestable-causal} + `probe_verdict` ∈ {positive,null,missing} — quadrant counts recompute from those; plotted set drops max_auc-None and untestable-without-steered-F cells.
- transfer_scatter sidecar has `transfer_stats` (rho/p/CI/n) and 4 extra "identity (perfect transfer)" construction points; Spearman over the 31 shared-P1 points reproduces rho exactly; parent x-values recompute from `git show 2af3e898…:eval_results/issue_2162/f_metrics/f_cells.jsonl` under the plain per-arm |separation|≥0.5 mean.
- mapshift: `fresh_fit_diagnostics.json per_layer[L].context_grain.r2_map_ctx / r2_idbias_ctx / knn` ↔ mapshift_r2; `shift_summary.json views.survivors["fresh|L<k>|steered"].mean_cos_over_cells` ↔ shift-prediction summary curve, `ctxshift|…` ↔ raw-context-shift curve; `dv3_ext.json per_config["freshce|L<k>|span"].pooled.cosine.acc` ↔ 2AFC curve.

**THE ROUND-1 BLOCKER — shared-y heatmap tick clobber (check this shape on EVERY multi-panel heatmap with unequal row counts):** `fig_layer_profile` companion heatmap used `plt.subplots(1,2, sharey=True)` + per-panel `imshow(extent=…)` + `set_yticks/set_yticklabels` with 39 ce rows vs 37 pe rows. The SECOND panel's ticks+extent clobber the shared axis: ce rows ≥22 rendered under the pe label set (off-by-two — query_content|ce's dark band landed on the "recency_fact_user_name_d3" label), and ce rows 37–38 (user_expertise, verbosity) were CROPPED silently. Detection recipe: sidecar `text.axes[*].yticklabels` count vs source row count (37 captured vs 39 claimed), then zoom-crop the PNG (PIL crop → Read) and match band signatures against per-row AUC thirds. A row whose data is constant 1.0 rendering with a teal band is proof, not style.
**How to apply:** on any body heatmap, compare sidecar yticklabels count against the source-table row count for that panel BEFORE trusting caption row counts; unequal-row shared-y panels are presumptively broken.

**Minor recurring notes:** plotter captions may say "per-type-cell curves" for point-marker renders (`_kind: scatter`) — note, not bounce; "y-axis spans roughly A to B" can overstate rendered limits by a few units — note. Sidecar y-key is per-axes (`"y"` on unlabeled panels) — value-match per point's own key (#2162 lesson, held here too).
