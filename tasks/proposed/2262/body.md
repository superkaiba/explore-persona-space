---
title: paper_plots._extract_scatters drops genuine (0,0) single-point scatters from
  figure sidecars
kind: infra
tags: []
created_at: '2026-08-13T01:48:20Z'
has_clean_result: false
origin_prompt: 'report-verifier finding, #2162 persona-specificity-ladder fold round
  1'
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/analysis/paper_plots.py
problem: `_extract_scatters` skips ANY single-point PathCollection whose lone offset is exactly (0,0) as "matplotlib's default for empty collections" (lines 852-854). Per-point `ax.scatter([x],[y])` loops — the standard per-carrier/per-pair companion pattern — therefore silently drop every genuine (0,0) datum from the sidecar `points` AND `total_points` (issue 2162 ladder_percarrier: 252 captured vs 264 rendered), corrupting the exact surface report-verifier recomputes against and nearly producing a spurious FAIL in the #2162 ladder round (report-verified round 1, 2026-08-13).
fix_sketch: discriminate a real point from the empty-collection default via artist properties instead of coordinates — e.g. skip only when `offsets.shape[0]==1 and not np.any(offsets) and coll.get_sizes().size == 0` (a real `scatter(...,s=...)` carries a nonempty sizes array), or when facecolors are unset; add a regression test rendering `ax.scatter([0.0],[0.0], s=14)` and asserting the sidecar captures it.
confidence: high
<!-- /workflow-fix-candidate -->
