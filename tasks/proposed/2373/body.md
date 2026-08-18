---
title: 'Mechanize the shared-y heatmap tick-clobber check: sidecar image row counts
  + a verify_report gate'
kind: infra
tags: []
created_at: '2026-08-18T14:40:08Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'report-verifier surfaced during #2329 Step 7d round 1: a sharey=True
  heatmap with unequal row counts silently clobbered tick labels and cropped rows;
  the defect survived the plotter, the methodology-critic, and a 20/20 verify_report
  --mode generation PASS'
workflow: v1
---
# Mechanize the shared-y heatmap tick-clobber check: sidecar row counts + a verify_report gate

## Provenance

Surfaced by the `report-verifier` during #2329 Step 7d round 1, which found a reader-facing figure defect that survived the plotter, the `methodology-critic` (170 claims traced, zero untraceable), AND a 20/20 `verify_report.py --mode generation` PASS. Filed per `.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups trigger the same auto-file + spawn as a formal candidate block).

## The defect class

`scripts/issue2329_figures.py` `fig_layer_profile` built its companion heatmap with:

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
...
im = ax.imshow(..., extent=(-half, 1.0 + half, len(rows) - 0.5, -0.5))
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["cell"] for r in rows], fontsize=5.5)
```

with UNEQUAL row counts per panel — 39 ce rows vs 37 pe rows (`eval_results/issue_2329/f_metrics/probe.json`). Because the y-axis is SHARED and the pe panel is drawn second, the pe panel's `extent` and `set_yticklabels` win:

- the ce panel rendered 37 rows carrying pe's label list;
- the two ce-only cells (`persona_prompted`, `persona_role_header`) sort MID-list, so every ce label below the insertion point shifted by two;
- ce rows 37-38 (`user_expertise`, `verbosity`) fell outside the visible y-range and were cropped silently.

Reader-facing consequence: `query_content|ce` — the only ce probe-null cell, AUC 0.310-0.521 — rendered under the `recency_fact_user_name_d3` label, while the real `recency_instr_format_d3/d5|ce` rows (~0.97-1.00) were displaced. A reader got INVERTED decodability reads on exactly the informative rows, from a figure whose caption asserted "39 for slot ce".

## Why the existing gates all missed it

- **Figure-sanity read (orchestrator + plotter):** the duty checks for empty axes, missing series, and implausible ranges. A 39-row heatmap rendering 37 correctly-formatted rows with plausible colour bands passes every one of those. The defect is a LABEL-CORRESPONDENCE bug, invisible to a visual sanity read.
- **`methodology-critic`:** traces prose claims to ground truth. The caption's "39 for slot ce" traces correctly to `probe.json` — the caption is true of the DATA and false only of the RENDER, which is outside that gate's remit.
- **`verify_report.py --mode generation`:** verifies pin well-formedness and blob identity (`git hash-object` vs `git rev-parse <sha>:<path>`), not axis-label semantics. It returned 20/20 PASS on the broken render.
- Only per-row numerical cross-referencing of band positions against the source table caught it.

## Proposed fix (two mechanizable arms)

**(a) Sidecar capture — `savefig_paper` (`src/explore_persona_space/analysis/paper_plots.py`).** For each axes, record the `imshow`/`pcolormesh` array row count alongside the already-captured `text.axes[*].yticklabels`. The sidecar today captures the labels but NOT the array shape, so the mismatch is undetectable from the sidecar alone — the round-1 detection required reading the source table separately.

**(b) Gate — `scripts/verify_report.py`.** FAIL when, on an axes with explicitly-set ticks, the captured image row count != `len(yticklabels)`; and FAIL when `sharey=True` sibling panels carry unequal heatmap row counts. Both are cheap, purely mechanical, and read from the sidecar with no re-render.

A plot-time assert in the shared save helper is the stricter alternative to (b) (fail at render rather than at verify). #2329's own round-3 fix adds an issue-local per-axes assert in `fig_layer_profile`; this task is the GENERALIZABLE version, so the next dense multi-panel heatmap anywhere in the repo cannot ship the same silent clobber.

## Acceptance criteria

1. `savefig_paper` records per-axes image array row/col counts in the `.meta.json` sidecar.
2. `verify_report.py` FAILs on (i) `n_image_rows != len(yticklabels)` for an axes with explicit ticks, and (ii) unequal-row heatmap panels under a shared y-axis.
3. A regression test reproducing the #2329 shape: two `sharey=True` panels, 39-row and 37-row `imshow`, per-panel `set_yticklabels` — asserted RED before the fix and GREEN after.
4. The check is WARN-or-FAIL calibrated so it cannot wedge the fleet on pre-existing figures: run it against the existing `figures/issue_*/**.meta.json` corpus first and report how many current sidecars would trip it. If the count is non-trivial, the new check FAILs only for freshly-assembled reports (generation mode) and WARNs at promote, mirroring the existing mode-split degrade ladders in that script. A no-flags-lint-red or fleet-wide-FAIL posture is the #1388 wedge shape and is explicitly out of bounds.
5. Sidecars that predate arm (a) must not FAIL for lacking the new field — absent row-count data degrades to a skip-with-note, never a FAIL.

## Non-goals

- Re-rendering historical figures.
- Any change to `#2329`'s own figures (handled in that task's round 3).

## Reference

- Detection recipe + the full trap write-up: `.claude/agent-memory/report-verifier/project_issue2329_gate_recipe_and_sharey_trap.md` @ `93f9d99f7ee33944fb7c2c99818eb4f0b7724e00`
- The `epm:report-verified v1` FAIL marker on #2329 (round 1) carries the numeric evidence.
